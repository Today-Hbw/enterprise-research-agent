from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from typing import Any

from app.models import AccessContext, Source, SourceType, ToolCall, ToolPermission, ToolResult
from app.tools.base import BaseTool

_WORKER = r"""
import ast, json, sys
payload = json.loads(sys.stdin.read())
expression = payload["expression"]
variables = payload.get("variables", {})
if not isinstance(expression, str) or len(expression) > 2000:
    raise ValueError("expression must be a string of at most 2000 characters")
if not isinstance(variables, dict) or len(variables) > 32:
    raise ValueError("variables must be an object with at most 32 entries")
allowed_names = {}
for key, value in variables.items():
    if not isinstance(key, str) or not key.isidentifier() or key.startswith("_"):
        raise ValueError("invalid variable name")
    if not isinstance(value, (int, float, bool, list, tuple)):
        raise ValueError("variable values must be scalar or numeric sequences")
    allowed_names[key] = value
node = ast.parse(expression, mode="eval")
allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
    ast.Constant, ast.Name, ast.List, ast.Tuple, ast.Load, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.FloorDiv, ast.Mod, ast.USub, ast.UAdd, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
if sum(1 for _ in ast.walk(node)) > 200:
    raise ValueError("expression is too complex")
for item in ast.walk(node):
    if not isinstance(item, allowed):
        raise ValueError("expression contains a forbidden operation")
    if isinstance(item, ast.Name) and item.id not in allowed_names:
        raise ValueError("expression references an unknown variable")
result = eval(compile(node, "<isolated-expression>", "eval"), {"__builtins__": {}}, allowed_names)
if not isinstance(result, (int, float, bool, str, list, tuple, type(None))):
    raise ValueError("result type is not supported")
print(json.dumps({"result": result}, ensure_ascii=False))
"""


class IsolatedPythonTool(BaseTool):
    name = "python_execute"
    description = (
        "Evaluate a restricted deterministic expression in an isolated, network-free worker."
    )
    input_schema = {
        "type": "object",
        "properties": {
            "expression": {"type": "string", "minLength": 1, "maxLength": 2000},
            "variables": {"type": "object"},
        },
        "required": ["expression"],
    }
    permission = ToolPermission.MEDIUM
    is_stub = False

    def __init__(self, timeout_seconds: float, max_output_bytes: int) -> None:
        super().__init__(timeout_seconds)
        self.max_output_bytes = max_output_bytes

    async def execute(
        self, call: ToolCall, access_context: AccessContext | None = None
    ) -> ToolResult:
        del access_context
        payload = {
            "expression": call.arguments.get("expression"),
            "variables": call.arguments.get("variables", {}),
        }
        try:
            completed = await asyncio.to_thread(self._run, payload)
        except subprocess.TimeoutExpired:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Python worker timed out.",
                error=f"Worker timed out after {self.timeout_seconds}s",
            )
        if completed.returncode != 0:
            error = completed.stderr.decode("utf-8", "replace").strip().splitlines()[-1:]
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Python worker rejected the request.",
                error=error[0] if error else "Worker failed",
            )
        if len(completed.stdout) > self.max_output_bytes:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Python worker output exceeded the limit.",
                error="Output limit exceeded",
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return ToolResult(
                call_id=call.call_id,
                tool_name=self.name,
                success=False,
                summary="Python worker returned invalid output.",
                error="Invalid worker JSON",
            )
        return ToolResult(
            call_id=call.call_id,
            tool_name=self.name,
            success=True,
            summary="Isolated deterministic expression evaluated.",
            data=result,
            sources=[
                Source(
                    source_type=SourceType.PYTHON,
                    title="Isolated Python calculation",
                    content_snippet=(
                        "Restricted expression executed without filesystem or network access."
                    ),
                )
            ],
        )

    def _run(self, payload: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
        with tempfile.TemporaryDirectory(prefix="research-agent-python-") as directory:
            return subprocess.run(
                [sys.executable, "-I", "-S", "-c", _WORKER],
                input=json.dumps(payload).encode(),
                capture_output=True,
                cwd=directory,
                timeout=self.timeout_seconds,
                check=False,
            )
