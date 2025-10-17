# Pre-Commit Security Check Script
# Run this before pushing to GitHub

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "  SECURITY CHECK STARTED" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

$issues = 0

# Check 1: .gitignore exists
Write-Host "[1/5] Checking .gitignore..." -ForegroundColor Yellow
if (Test-Path ".gitignore") {
    Write-Host "  PASS: .gitignore exists" -ForegroundColor Green
} else {
    Write-Host "  FAIL: .gitignore NOT FOUND!" -ForegroundColor Red
    $issues++
}

# Check 2: .env not in git
Write-Host ""
Write-Host "[2/5] Checking for .env files in git..." -ForegroundColor Yellow
try {
    $envInGit = git ls-files | Select-String "\.env$"
    if ($envInGit) {
        Write-Host "  FAIL: .env file found in git!" -ForegroundColor Red
        $envInGit | ForEach-Object { Write-Host "    - $_" -ForegroundColor Red }
        $issues++
    } else {
        Write-Host "  PASS: No .env files tracked" -ForegroundColor Green
    }
} catch {
    Write-Host "  SKIP: Git not initialized" -ForegroundColor Yellow
}

# Check 3: Check staged files
Write-Host ""
Write-Host "[3/5] Checking staged files..." -ForegroundColor Yellow
try {
    $staged = git diff --cached --name-only 2>$null
    $sensit
