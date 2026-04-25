# ══════════════════════════════════════════════════════════════════
#  CMS — Quick-start setup script (PowerShell)
#
#  USAGE (run from the repo root — TDPCL\):
#    .\setup.ps1
#
#  What it does:
#    1. Creates a Python virtual environment in backend\.venv
#    2. Installs all pinned core requirements
#    3. Copies .env.example → backend\.env
#    4. Installs frontend Node packages
#    5. Prints the commands to start both servers
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Condition Monitoring System — Setup                ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Python venv ────────────────────────────────────────────
Write-Host "► Step 1/4  Creating Python virtual environment …" -ForegroundColor Yellow
Set-Location backend
python -m venv .venv

# ── Step 2: Activate + install ────────────────────────────────────
Write-Host "► Step 2/4  Installing Python packages …" -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip --quiet
pip install -r requirements.txt

# ── Step 3: .env file ─────────────────────────────────────────────
Write-Host "► Step 3/4  Creating backend\.env from template …" -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    Copy-Item ..\.env.example .env
    Write-Host "   Created .env  (edit ANTIGRAVITY_ENABLED to enable AG mode)" -ForegroundColor Gray
} else {
    Write-Host "   .env already exists — skipping" -ForegroundColor Gray
}

# ── Step 4: Frontend ──────────────────────────────────────────────
Write-Host "► Step 4/4  Installing frontend Node packages …" -ForegroundColor Yellow
Set-Location ..\frontend
npm install --silent

Set-Location ..

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║   ✅  Setup complete!                                 ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "Start the backend (new terminal):" -ForegroundColor Cyan
Write-Host "  cd backend" -ForegroundColor White
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  uvicorn main:app --reload --port 8000" -ForegroundColor White
Write-Host ""
Write-Host "Start the frontend (another terminal):" -ForegroundColor Cyan
Write-Host "  cd frontend" -ForegroundColor White
Write-Host "  npm run dev" -ForegroundColor White
Write-Host ""
Write-Host "API docs: http://localhost:8000/docs" -ForegroundColor DarkCyan
Write-Host "Dashboard: http://localhost:5173" -ForegroundColor DarkCyan
Write-Host ""
