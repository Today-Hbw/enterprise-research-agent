const state = {
  conversationId: null,
  running: false,
  sources: new Map(),
};

const elements = {
  form: document.querySelector("#chat-form"),
  query: document.querySelector("#query"),
  send: document.querySelector("#send"),
  messages: document.querySelector("#messages"),
  conversations: document.querySelector("#conversation-list"),
  trace: document.querySelector("#trace-list"),
  summary: document.querySelector("#run-summary"),
  sources: document.querySelector("#source-list"),
  sourceCount: document.querySelector("#source-count"),
  dashboard: document.querySelector("#dashboard-content"),
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function addMessage(role, content, pending = false) {
  document.querySelector(".empty-state")?.remove();
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <div class="message-label">${role === "user" ? "You" : "Research Agent"}</div>
    <div class="message-content ${pending ? "thinking" : ""}">${escapeHtml(content)}</div>`;
  elements.messages.append(wrapper);
  elements.messages.scrollTop = elements.messages.scrollHeight;
  return wrapper.querySelector(".message-content");
}

function addTrace(title, description, type = "decision", meta = "") {
  const item = document.createElement("div");
  item.className = `trace-item ${type}`;
  item.innerHTML = `<strong>${escapeHtml(title)}</strong><p>${escapeHtml(description)}</p>${meta ? `<small>${escapeHtml(meta)}</small>` : ""}`;
  elements.trace.append(item);
}

function addSources(sources) {
  for (const source of sources || []) {
    if (state.sources.has(source.source_id)) continue;
    state.sources.set(source.source_id, source);
    const card = document.createElement("article");
    card.className = "source-card";
    card.innerHTML = `
      <span class="source-type">${escapeHtml(source.source_type)}</span>
      <h3>${escapeHtml(source.title)}</h3>
      <p>${escapeHtml(source.content_snippet)}</p>
      ${source.url ? `<a href="${escapeHtml(source.url)}" target="_blank" rel="noreferrer">Open source ↗</a>` : ""}`;
    elements.sources.append(card);
  }
  elements.sourceCount.textContent = state.sources.size;
}

function resetRun() {
  state.sources.clear();
  elements.sources.innerHTML = "";
  elements.sourceCount.textContent = "0";
  elements.trace.innerHTML = "";
  elements.summary.classList.remove("muted");
  elements.summary.textContent = "Preparing research run…";
}

function formatCost(value) {
  return value === null || value === undefined ? "Not configured" : `$${Number(value).toFixed(6)}`;
}

function budgetLabel(run) {
  const limits = [];
  if (run.budget.token_limit !== null) {
    limits.push(`${run.metrics.token_usage.toLocaleString()} / ${run.budget.token_limit.toLocaleString()} tokens`);
  }
  if (run.budget.cost_limit !== null) {
    limits.push(`${formatCost(run.metrics.estimated_cost)} / ${formatCost(run.budget.cost_limit)}`);
  }
  return limits.length ? limits.join(" · ") : "No run budget configured";
}

function renderRunSummary(run) {
  const budgetState = run.metrics.budget_exhausted ? "Budget exhausted" : "Within budget";
  elements.summary.innerHTML = `
    <div class="run-summary-heading">
      <strong>${escapeHtml(run.status.toUpperCase())}</strong>
      <span>${escapeHtml(run.run_id)}</span>
    </div>
    <div class="metric-grid">
      <div><span>Tokens</span><strong>${run.metrics.token_usage.toLocaleString()}</strong></div>
      <div><span>Estimated cost</span><strong>${formatCost(run.metrics.estimated_cost)}</strong></div>
      <div><span>Decisions</span><strong>${run.metrics.llm_call_count}</strong></div>
      <div><span>Tool calls</span><strong>${run.metrics.tool_call_count}</strong></div>
      <div><span>Input</span><strong>${run.metrics.input_tokens.toLocaleString()}</strong></div>
      <div><span>Output</span><strong>${run.metrics.output_tokens.toLocaleString()}</strong></div>
    </div>
    <div class="budget-line ${run.metrics.budget_exhausted ? "exhausted" : ""}">
      <strong>${budgetState}</strong>
      <span>${escapeHtml(budgetLabel(run))}</span>
    </div>`;
}

function renderStoredRun(run) {
  resetRun();
  renderRunSummary(run);
  let decisionNumber = 0;
  for (const step of run.trace) {
    if (!step.tool_name) decisionNumber += 1;
    addTrace(
      step.tool_name || `Agent decision · step ${decisionNumber}`,
      step.tool_output_summary || step.summary,
      step.tool_name ? "tool" : "decision",
      step.tool_name ? `${step.duration_ms}ms · ${step.status.toUpperCase()}` : "",
    );
  }
  addSources(run.sources);
}

async function openConversation(conversationId) {
  if (state.running) return;
  const response = await fetch(`/api/conversations/${encodeURIComponent(conversationId)}`);
  if (!response.ok) return;
  const conversation = await response.json();
  state.conversationId = conversation.conversation_id;
  elements.messages.innerHTML = "";
  for (const message of conversation.messages) addMessage(message.role, message.content);

  const latestRunId = conversation.run_ids.at(-1);
  if (latestRunId) {
    const runResponse = await fetch(`/api/runs/${encodeURIComponent(latestRunId)}`);
    if (runResponse.ok) renderStoredRun(await runResponse.json());
  } else {
    resetRun();
    elements.summary.textContent = "No completed run";
  }
  await loadConversations();
}

async function parseEventStream(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";
    for (const frame of frames) {
      const dataLine = frame.split("\n").find((line) => line.startsWith("data: "));
      if (dataLine) onEvent(JSON.parse(dataLine.slice(6)));
    }
  }
}

function handleEvent(payload, assistantNode) {
  const { event, data } = payload;
  if (event === "run_started") {
    state.conversationId = data.conversation_id;
    elements.summary.textContent = `${payload.run_id}\n${data.model}\nRUNNING`;
    addTrace("Run started", "Runtime limits and tool registry loaded.", "decision", payload.run_id);
  } else if (event === "agent_decision") {
    addTrace(`Agent decision · step ${data.step}`, data.summary);
  } else if (event === "tool_started") {
    addTrace(data.tool_name, "Executing deterministic placeholder tool…", "tool", data.call_id);
  } else if (event === "tool_completed") {
    addTrace(`${data.tool_name} · complete`, data.summary, "tool", `${data.duration_ms}ms · STUB`);
    addSources(data.sources);
  } else if (event === "assistant_delta") {
    assistantNode.classList.remove("thinking");
    assistantNode.textContent = data.content;
  } else if (event === "run_completed") {
    const run = data.run;
    renderRunSummary(run);
    loadDashboard();
  } else if (event === "run_failed") {
    assistantNode.classList.remove("thinking");
    assistantNode.textContent = `Run failed: ${data.error}`;
    if (data.run) {
      renderRunSummary(data.run);
    } else {
      elements.summary.textContent = `FAILED\n${data.error}`;
    }
    loadDashboard();
  }
}

async function submitQuery(query) {
  if (state.running || !query.trim()) return;
  state.running = true;
  elements.send.disabled = true;
  resetRun();
  addMessage("user", query.trim());
  const assistantNode = addMessage("assistant", "Planning and selecting tools…", true);
  elements.query.value = "";
  elements.query.style.height = "auto";

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query.trim(), conversation_id: state.conversationId }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || response.statusText);
    await parseEventStream(response, (payload) => handleEvent(payload, assistantNode));
    await loadConversations();
  } catch (error) {
    assistantNode.classList.remove("thinking");
    assistantNode.textContent = `Request failed: ${error.message}`;
  } finally {
    state.running = false;
    elements.send.disabled = false;
    elements.query.focus();
  }
}

async function loadConversations() {
  const response = await fetch("/api/conversations");
  if (!response.ok) return;
  const conversations = await response.json();
  elements.conversations.innerHTML = conversations.map((conversation) => `
    <button class="conversation-item ${conversation.conversation_id === state.conversationId ? "active" : ""}" data-id="${conversation.conversation_id}">
      <strong>${escapeHtml(conversation.title)}</strong>
      <span>${conversation.messages.length} messages · ${new Date(conversation.updated_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</span>
    </button>`).join("");
}

async function loadDashboard() {
  const response = await fetch("/api/runs?limit=20");
  if (!response.ok) {
    elements.dashboard.textContent = "Run metrics unavailable";
    return;
  }
  const dashboard = await response.json();
  const completed = dashboard.status_counts.completed || 0;
  const failed = dashboard.status_counts.failed || 0;
  const recentRuns = dashboard.recent_runs.map((run) => `
    <button class="dashboard-run" data-conversation-id="${escapeHtml(run.conversation_id)}">
      <span class="dashboard-run-status ${escapeHtml(run.status)}">${escapeHtml(run.status)}</span>
      <strong>${escapeHtml(run.user_query)}</strong>
      <small>${run.metrics.token_usage.toLocaleString()} tokens · ${formatCost(run.metrics.estimated_cost)}</small>
    </button>`).join("");
  elements.dashboard.classList.remove("muted");
  elements.dashboard.innerHTML = `
    <div class="dashboard-grid">
      <div><span>Recent runs</span><strong>${dashboard.total_runs}</strong></div>
      <div><span>Completed</span><strong>${completed}</strong></div>
      <div><span>Failed</span><strong>${failed}</strong></div>
      <div><span>Total tokens</span><strong>${dashboard.total_tokens.toLocaleString()}</strong></div>
      <div><span>Estimated cost</span><strong>${formatCost(dashboard.total_estimated_cost)}</strong></div>
      <div><span>Avg duration</span><strong>${dashboard.average_duration_ms}ms</strong></div>
    </div>
    <h3 class="dashboard-heading">Latest activity</h3>
    <div class="dashboard-runs">${recentRuns || '<p class="muted">No runs yet</p>'}</div>`;
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuery(elements.query.value);
});
elements.query.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.form.requestSubmit();
  }
});
elements.query.addEventListener("input", () => {
  elements.query.style.height = "auto";
  elements.query.style.height = `${Math.min(elements.query.scrollHeight, 170)}px`;
});
document.querySelector(".prompt-chip").addEventListener("click", (event) => submitQuery(event.currentTarget.dataset.prompt));
document.querySelector("#new-chat").addEventListener("click", () => {
  state.conversationId = null;
  elements.messages.innerHTML = `<div class="empty-state"><span class="empty-icon">⌁</span><h2>New research run</h2><p>Start a fresh in-memory conversation.</p></div>`;
  resetRun();
  elements.summary.textContent = "No active run";
  elements.summary.classList.add("muted");
  loadConversations();
});
elements.conversations.addEventListener("click", (event) => {
  const item = event.target.closest(".conversation-item");
  if (item) openConversation(item.dataset.id);
});
elements.dashboard.addEventListener("click", (event) => {
  const item = event.target.closest(".dashboard-run");
  if (item) openConversation(item.dataset.conversationId);
});
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab, .inspector-pane").forEach((item) => item.classList.remove("active"));
  tab.classList.add("active");
  document.querySelector(`#${tab.dataset.tab}-pane`).classList.add("active");
  if (tab.dataset.tab === "dashboard") loadDashboard();
}));

loadConversations();
loadDashboard();
