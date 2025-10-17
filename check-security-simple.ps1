# Security Check Before GitHub Push
# Simple version

Write-Host ""
Write-Host "==================================="  -ForegroundColor Cyan
Write-Host "   SECURITY CHECK" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

$hasIssues = $false

# Check 1
Write-Host "[CHECK 1] .gitignore exists..." -ForegroundColor Yellow
if (Test-Path ".gitignore") {
    Write-Host "  PASS" -ForegroundColor Green
} else {
    Write-Host "  FAIL - Create .gitignore!" -ForegroundColor Red
    $hasIssues = $true
}

# Check 2
Write-Host ""
Write-Host "[CHECK 2] .env file not in repository..." -ForegroundColor Yellow
if (Test-Path ".env") {
    Write-Host "  WARNING: .env exists locally (this is OK if in .gitignore)" -ForegroundColor Yellow
}
$content = Get-Content ".gitignore" -Raw
if ($content -match "\.env") {
    Write-Host "  PASS - .env is in .gitignore" -ForegroundColor Green
} else {
    Write-Host "  FAIL - Add .env to .gitignore!" -ForegroundColor Red
    $hasIssues = $true
}

# Check 3
Write-Host ""
Write-Host "[CHECK 3] venv/ directory excluded..." -ForegroundColor Yellow
if ($content -match "venv/") {
    Write-Host "  PASS" -ForegroundColor Green
} else {
    Write-Host "  FAIL - Add venv/ to .gitignore!" -ForegroundColor Red
    $hasIssues = $true
}

# Check 4
Write-Host ""
Write-Host "[CHECK 4] __pycache__/ excluded..." -ForegroundColor Yellow
if ($content -match "__pycache__") {
    Write-Host "  PASS" -ForegroundColor Green
} else {
    Write-Host "  WARNING - Add __pycache__/ to .gitignore" -ForegroundColor Yellow
}

# Check 5
Write-Host ""
Write-Host "[CHECK 5] node_modules/ excluded..." -ForegroundColor Yellow
if ($content -match "node_modules") {
    Write-Host "  PASS" -ForegroundColor Green
} else {
    Write-Host "  WARNING - Add node_modules/ to .gitignore" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
if ($hasIssues) {
    Write-Host "  FAILED - Fix issues above!" -ForegroundColor Red
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host ""
    exit 1
} else {
    Write-Host "  ALL CHECKS PASSED" -ForegroundColor Green
    Write-Host "  Safe to push!" -ForegroundColor Green
    Write-Host "===================================" -ForegroundColor Cyan
    Write-Host ""
    exit 0
}
