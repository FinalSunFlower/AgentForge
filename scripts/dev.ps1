$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Error "Create .venv first: py -3.12 -m venv .venv && .\.venv\Scripts\python.exe -m pip install -e `".[dev]`""
}
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Copied .env.example to .env"
}

$env:PYTHONPATH = (Get-Location).Path
Write-Host "Core API  http://localhost:8100"
Write-Host "Runtime   http://localhost:8101"
Write-Host "Console   cd apps/web && npm run dev  (http://localhost:3000)"
Write-Host "Evals/tools/architecture need no LLM key. Playground runs do."

Start-Process -NoNewWindow -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "uvicorn", "services.core_api.app.main:app", "--port", "8100"
Start-Sleep -Seconds 1
& .\.venv\Scripts\python.exe -m uvicorn services.agent_runtime.app.main:app --port 8101
