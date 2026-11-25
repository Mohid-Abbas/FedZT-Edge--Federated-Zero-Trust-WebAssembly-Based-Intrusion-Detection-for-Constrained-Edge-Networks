@echo off
echo ======================================================================
echo  FedZT-Edge Phase 3 Research Implementation & Evaluation Runner
echo  Authors: Mohid Abbas (24i-0074) & Ali Khan (24i-0069)
echo ======================================================================

echo [1/3] Running Unit Test Suite...
python -m unittest discover -s tests
if %ERRORLEVEL% NEQ 0 (
    echo Unit tests failed! Exiting.
    exit /b %ERRORLEVEL%
)

echo [2/3] Verifying / Preparing Datasets...
python data\download_and_prepare_datasets.py

echo [3/3] Running Full Experimental Benchmark Suite...
python experiments\run_all_experiments.py

echo ======================================================================
echo  Execution Complete! All Figures and Tables have been generated.
echo  Figures: Phase 3\results\figures\
echo  Tables:  Phase 3\results\tables\
echo ======================================================================
pause
