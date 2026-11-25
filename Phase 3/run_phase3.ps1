# FedZT-Edge Phase 3 PowerShell Execution Script
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " FedZT-Edge Phase 3 Research Implementation & Evaluation Runner" -ForegroundColor Cyan
Write-Host " Authors: Mohid Abbas (24i-0074) & Ali Khan (24i-0069)" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

Write-Host "`n[1/3] Running Unit Test Suite..." -ForegroundColor Yellow
python -m unittest discover -s tests
if ($LASTEXITCODE -ne 0) {
    Write-Error "Unit tests failed! Exiting."
    exit $LASTEXITCODE
}

Write-Host "`n[2/3] Verifying / Preparing Datasets..." -ForegroundColor Yellow
python data\download_and_prepare_datasets.py

Write-Host "`n[3/3] Running Full Experimental Benchmark Suite..." -ForegroundColor Yellow
python experiments\run_all_experiments.py

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host " Execution Complete! All Figures and Tables have been generated." -ForegroundColor Green
Write-Host " Figures: Phase 3\results\figures\" -ForegroundColor Green
Write-Host " Tables:  Phase 3\results\tables\" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
