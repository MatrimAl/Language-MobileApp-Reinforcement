# 🚀 reinFORCING_the_people - Quick Start Guide

Write-Host "🎓 reinFORCING_the_people - RL Language Learning Setup" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "✅ Checking Python..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Python not found! Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📦 Setting up Backend..." -ForegroundColor Yellow
Set-Location backend

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Green
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Green
.\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
pip install -r requirements.txt

# Create .env if not exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Green
    Copy-Item .env.example .env
}

Write-Host ""
Write-Host "✅ Backend setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Start MongoDB:" -ForegroundColor White
Write-Host "   docker run -d -p 27017:27017 --name mongodb mongo:latest" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Seed sample data:" -ForegroundColor White
Write-Host "   python seed_data.py" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Start backend server:" -ForegroundColor White
Write-Host "   python main.py" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Initialize RL model (in browser):" -ForegroundColor White
Write-Host "   http://localhost:8000/docs -> POST /api/rl/initialize" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Start dashboard (in new terminal):" -ForegroundColor White
Write-Host "   cd ../dashboard" -ForegroundColor Gray
Write-Host "   pip install -r requirements.txt" -ForegroundColor Gray
Write-Host "   streamlit run app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "🎉 Happy Coding!" -ForegroundColor Cyan
