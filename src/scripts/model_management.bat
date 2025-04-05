@echo off

set root=../
set PYTHONPATH-%PYTHONPATH%;%root%

python model_management.py %*
if %errorlevel% neq 0 (
    echo.
    echo Error: model_management.py failed with error code %errorlevel%.
    exit /b %errorlevel%
)

echo.
echo model_management.py completed successfully.
exit /b 0
REM End of script
