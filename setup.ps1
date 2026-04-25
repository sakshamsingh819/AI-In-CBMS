# ══════════════════════════════════════════════════════════════════
#  CMS — Quick-start setup script (PowerShell)
#  USAGE (run from repo root TDPCL\):   .\setup.ps1
# ══════════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Condition Monitoring System -- Setup               ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Step 1 -- Python venv
Write-Host "Step 1/4  Creating Python virtual environment ..." -ForegroundColor Yellow
Set-Location backend
python -m venv .venv

# Step 2 -- Activate and install
Write-Host "Step 2/4  Installing Python packages ..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Step 3 -- .env file
Write-Host "Step 3/4  Creating backend\.env from template ..." -ForegroundColor Yellow
if (-Not (Test-Path ".env")) {
    Copy-Item ..\.env.example .env
    Write-Host "   Created .env" -ForegroundColor Gray
}

# Step 4 -- Frontend
Write-Host "Step 4/4  Installing frontend Node packages ..." -ForegroundColor Yellow
Set-Location ..\frontend
npm install

Set-Location ..

Write-Host ""
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "To generate scores and results now:" -ForegroundColor Cyan
Write-Host "  cd backend"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python scripts\demo_results.py"
Write-Host ""
Write-Host "To start backend:"
Write-Host "  uvicorn main:app --reload --port 8000"
Write-Host ""
Write-Host "To start frontend (separate terminal):"
Write-Host "  cd frontend"
Write-Host "  npm run dev"
Write-Host ""
