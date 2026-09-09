[CmdletBinding()]
param(
    [string]$KeyPath = $env:ERA_SSH_KEY,
    [string]$Server = $env:ERA_SERVER,
    [string]$PublicUrl = $env:ERA_PUBLIC_URL,
    [int]$Port = 22,
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$openSshRoot = Join-Path $env:SystemRoot "System32\OpenSSH"
$ssh = Join-Path $openSshRoot "ssh.exe"
$scp = Join-Path $openSshRoot "scp.exe"

if (-not $KeyPath -or -not (Test-Path -LiteralPath $KeyPath)) {
    throw "Pass -KeyPath or set ERA_SSH_KEY to the SSH private-key path."
}
if (-not $Server) {
    throw "Pass -Server or set ERA_SERVER to the SSH target (for example, user@host)."
}
if (-not (Test-Path -LiteralPath $ssh) -or -not (Test-Path -LiteralPath $scp)) {
    throw "Windows OpenSSH client is required."
}

Push-Location $repoRoot
try {
    if (-not $SkipTests) {
        $testEnvironment = @{
            APP_ENV = "test"
            LLM_PROVIDER = "deterministic"
            OPENAI_API_KEY = ""
            DOUBAO_API_KEY = ""
            KNOWLEDGE_BACKEND = "memory"
            RAG_PLATFORM_API_KEY = ""
            WEB_SEARCH_BACKEND = "stub"
            BRAVE_SEARCH_API_KEY = ""
            HTTP_FETCH_BACKEND = "stub"
            SQL_BACKEND = "stub"
            POSTGRES_DSN = ""
            PYTHON_BACKEND = "stub"
            BROWSER_BACKEND = "stub"
            MCP_SERVERS_JSON = ""
            STATE_BACKEND = "memory"
            STATE_POSTGRES_DSN = ""
            REDIS_URL = ""
        }
        $savedEnvironment = @{}
        try {
            foreach ($entry in $testEnvironment.GetEnumerator()) {
                $savedEnvironment[$entry.Key] = [Environment]::GetEnvironmentVariable($entry.Key, "Process")
                [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
            }
            & uv run --locked --extra dev pytest -q
            if ($LASTEXITCODE -ne 0) { throw "Tests failed." }
        }
        finally {
            foreach ($entry in $savedEnvironment.GetEnumerator()) {
                [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, "Process")
            }
        }
    }

    $revision = (& git rev-parse --short=12 HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw "Unable to resolve Git revision." }
    $dirtySuffix = if (& git status --porcelain) { "-worktree" } else { "" }
    $release = "$(Get-Date -Format yyyyMMddHHmmss)-$revision$dirtySuffix"
    $remoteRelease = "/mnt/enterprise-research-agent/docker/releases/$release"
    $image = "enterprise-research-agent:$release"
    $archive = Join-Path ([IO.Path]::GetTempPath()) "$release.tar"
    $archiveName = Split-Path -Leaf $archive
    $packagePaths = @(
        "app",
        "scripts/deploy-server.sh",
        ".dockerignore",
        "docker-compose.server.yml",
        "Dockerfile",
        "pyproject.toml",
        "uv.lock",
        "README.md",
        "LICENSE"
    )

    try {
        & tar.exe -cf $archive @packagePaths
        if ($LASTEXITCODE -ne 0) { throw "Unable to create deployment archive." }

        & $ssh -i $KeyPath -p $Port $Server "mkdir -p '$remoteRelease'"
        if ($LASTEXITCODE -ne 0) { throw "Unable to create remote release directory." }

        $remoteArchive = "$remoteRelease/$archiveName"
        & $scp -i $KeyPath -P $Port $archive "${Server}:$remoteArchive"
        if ($LASTEXITCODE -ne 0) { throw "Unable to upload deployment archive." }

        $remoteCommand = "tar -xf '$remoteArchive' -C '$remoteRelease' && rm '$remoteArchive' && sed -i 's/`r$//' '$remoteRelease/scripts/deploy-server.sh' && bash '$remoteRelease/scripts/deploy-server.sh' '$remoteRelease' '$image'"
        & $ssh -i $KeyPath -p $Port $Server $remoteCommand
        if ($LASTEXITCODE -ne 0) { throw "Remote Docker deployment failed." }
    }
    finally {
        if (Test-Path -LiteralPath $archive) {
            Remove-Item -LiteralPath $archive
        }
    }

    Write-Host "Deployed $image"
    if ($PublicUrl) {
        Write-Host "Open $PublicUrl"
    }
}
finally {
    Pop-Location
}
