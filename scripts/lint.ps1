#!/usr/bin/env pwsh
# Lint script for CHAMPPy project
# Runs Black and Flake8 code quality checks

$ErrorActionPreference = "Continue"
$ExitCode = 0

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  CHAMPPy Code Quality Checks" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Black
Write-Host "Running Black (code formatter check)..." -ForegroundColor Yellow
black -l 120 --check ./src/champpy/ ./tests/ ./scripts/
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Black found formatting issues!" -ForegroundColor Red
    Write-Host "  Fix with: black -l 120 ./src/champpy/ ./tests/" -ForegroundColor Gray
    $ExitCode = 1
} else {
    Write-Host "[PASS] Black check passed!" -ForegroundColor Green
}

Write-Host ""

# Check Flake8
Write-Host "Running Flake8 (linting)..." -ForegroundColor Yellow
flake8 --max-line-length=120 --ignore=E501,E712,E203,W503 ./src/champpy/ ./tests/ ./scripts/ --statistics
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Flake8 found linting issues!" -ForegroundColor Red
    $ExitCode = 1
} else {
    Write-Host "[PASS] Flake8 check passed!" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan

if ($ExitCode -eq 0) {
    Write-Host "[SUCCESS] All checks passed!" -ForegroundColor Green
} else {
    Write-Host "[FAILED] Some checks failed!" -ForegroundColor Red
}

Write-Host "========================================" -ForegroundColor Cyan
exit $ExitCode
