@echo off
setlocal

set SCRIPT=apple-future-price.py
set MODEL=gemma3

echo ============================================================
echo  ARIMA Optimizer using Ollama (%MODEL%)
echo ============================================================
echo.
echo Choose an option:
echo   1. Suggest optimal (p,d,q) parameters for AAPL ARIMA model
echo   2. Generate improved Python code
echo.
set /p CHOICE="Enter 1 or 2: "

if "%CHOICE%"=="1" goto suggest_params
if "%CHOICE%"=="2" goto improve_code
echo Invalid choice. Exiting.
goto end

:suggest_params
echo.
echo Asking %MODEL% for optimal ARIMA (p,d,q) parameters...
echo ------------------------------------------------------------
(
  echo You are a quantitative finance expert. Given an ARIMA model applied to AAPL daily closing prices from 2015 to 2024, suggest the optimal (p,d,q) parameters. Explain what each parameter means and why your chosen values are appropriate for stock price data. Be concise and direct.
) | ollama run %MODEL%
goto end

:improve_code
echo.
echo Sending current script to %MODEL% for improvement...
echo ------------------------------------------------------------
set PROMPT_FILE=%TEMP%\arima_prompt.txt

(
  echo You are an expert Python developer and quantitative analyst. Here is a Python script that forecasts AAPL stock prices using ARIMA. Please rewrite it with improvements: better (p,d,q) selection using AIC/BIC, confidence intervals on the forecast plot, stationarity check with ADF test, and cleaner output. Return only the improved Python code with no extra explanation.
  echo.
  echo ```python
  type %SCRIPT%
  echo ```
) > %PROMPT_FILE%

ollama run %MODEL% < %PROMPT_FILE%

del %PROMPT_FILE%
goto end

:end
echo.
echo ============================================================
echo Done.
pause
