# AAPL Stock Price Forecasting with ARIMA + Ollama (Gemma3)

A Python project that forecasts Apple (AAPL) stock prices using an ARIMA time series model, with automated parameter optimization powered by the **Gemma3** LLM running locally via [Ollama](https://ollama.com).

---

## Features

- Downloads historical AAPL closing prices via `yfinance`
- Runs an **ADF stationarity test** to validate differencing requirements
- Performs a **grid search over (p, d, q)** using AIC/BIC to select the best ARIMA order automatically
- Generates a **30-day business-day forecast** with proper date indexing
- Plots historical prices alongside forecast with a **95% confidence interval**
- Includes a `.bat` helper that pipes the script to **Gemma3 via Ollama** for AI-assisted parameter suggestions and code improvements

---

## Project Structure

```
ARIMA/
├── apple-future-price.py   # Main forecasting script
├── ollama-improve.bat      # Ollama/Gemma3 AI assistant launcher
└── README.md
```

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

Install Python dependencies:

```bash
pip install yfinance statsmodels pandas matplotlib ollama
```

Pull the Gemma3 model:

```bash
ollama pull gemma3
```

---

## Usage

### Run the forecast

```bash
python apple-future-price.py
```

### Use Gemma3 AI assistant

```bash
ollama-improve.bat
```

Choose from two options:

| Option | Description |
|--------|-------------|
| `1` | Ask Gemma3 to recommend optimal (p, d, q) parameters with explanation |
| `2` | Send the script to Gemma3 for AI-generated code improvements |

---

## How It Works

### 1. Data Download

Historical AAPL daily closing prices from **2015-01-01 to 2024-01-01** are fetched using `yfinance` and resampled to business-day (`B`) frequency.

### 2. ADF Stationarity Test

An **Augmented Dickey-Fuller (ADF)** test is run before fitting the model:

```
ADF Statistic : 0.2438
P-value       : 0.9746
Critical [1%] : -3.4333
Critical [5%] : -2.8628
Critical [10%]: -2.5675
Verdict       : NON-STATIONARY (differencing needed)
```

A p-value > 0.05 confirms the raw price series is non-stationary — the ARIMA `d` parameter handles this via differencing.

### 3. AIC/BIC Grid Search

The script tests all combinations of `p ∈ [0,3]`, `d ∈ [0,1]`, `q ∈ [0,3]` and selects the model with the lowest AIC:

```
  p   d   q          AIC          BIC
--------------------------------------
  3   0   2      9020.84      9060.91   ← Best
  3   0   3      9022.49      9068.29
  2   1   3      9028.96      9063.31
  3   1   2      9029.04      9063.39
  1   1   3      9031.60      9060.22
  ...
```

**Best model: `ARIMA(3, 0, 2)`**

### 4. Forecast Output

30 business-day forecast from 2024-01-01 to 2024-02-09, converging around **~$190.53**:

```
2024-01-01    190.594829
2024-01-02    190.505858
...
2024-02-09    190.562626
```

### 5. Plot

The output chart displays:
- Historical closing prices (last 200 days)
- 30-day forecast line
- 95% confidence interval shading

---

## Gemma3 AI Recommendations

The `ollama-improve.bat` script queries **Gemma3** (Google's open-weight LLM, 3.3 GB, running locally via Ollama) for two types of analysis:

### Parameter Recommendation (Option 1)

Gemma3 recommended **`ARIMA(1, 1, 1)`** as a strong baseline:

| Parameter | Meaning | Gemma3's Reasoning |
|-----------|---------|-------------------|
| `p = 1` (AR) | Autoregressive lag | Stock prices exhibit serial correlation — yesterday's price strongly predicts today's |
| `d = 1` (I) | Differencing order | AAPL's raw price series is non-stationary; one level of differencing stabilizes the mean |
| `q = 1` (MA) | Moving average lag | Accounts for the impact of past forecast errors (market noise/shocks) |

> Gemma3 noted this is a **starting point** and recommended validating with ACF/PACF plots, residual diagnostics, and time-series cross-validation.

### Code Improvements (Option 2)

Gemma3 suggested the following enhancements, all of which were implemented:

- **ADF Test** — confirm stationarity before fitting
- **AIC/BIC grid search** — data-driven (p, d, q) selection instead of hardcoded values
- **95% Confidence Intervals** — visualize forecast uncertainty
- **Proper date index** — use `pd.bdate_range` so forecast x-axis shows real business dates, not integers
- **Warning suppression** — `warnings.filterwarnings("ignore")` for clean terminal output

The AIC grid search ultimately identified **`ARIMA(3, 0, 2)`** as the best-fit model, outperforming Gemma3's suggested `(1, 1, 1)` by AIC score.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Download AAPL historical price data |
| `statsmodels` | ARIMA model fitting, ADF test, AIC/BIC |
| `pandas` | Time series data handling |
| `matplotlib` | Plotting historical + forecast chart |
| `ollama` | Local LLM inference runtime |
| **Gemma3** | Google's open-weight LLM for parameter suggestions and code review |

---

## Notes

- ARIMA is a **univariate statistical model** — it only uses historical price patterns and does not factor in fundamentals, news, or sentiment.
- Forecasts converging to a flat value after a few steps is expected ARIMA behavior for near-random-walk financial data.
- Gemma3 runs **fully locally** — no API keys or internet connection required after the initial model pull.

---

## License

MIT
