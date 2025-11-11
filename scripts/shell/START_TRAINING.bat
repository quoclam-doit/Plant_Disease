@echo off
echo ============================================
echo Starting MambaTSR Training
echo ============================================
echo.
echo Training will run in background...
echo Check progress in: G:\Dataset\training.log
echo.

wsl bash -c "cd /mnt/g/Dataset && nohup /mnt/g/Dataset/.venv_wsl/bin/python train_mambatsr_plantvillage.py > training.log 2>&1 &"

timeout /t 5 > nul

echo.
echo Training started!
echo.
echo To check progress, run:
echo   wsl bash -c "tail -f /mnt/g/Dataset/training.log"
echo.
echo To check if running:
echo   wsl bash -c "ps aux | grep train_mambatsr"
echo.
pause
