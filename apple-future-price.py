import warnings
warnings.filterwarnings("ignore")

import os
import glob
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from datetime import datetime, timezone

TICKER      = "AAPL"
START_DATE  = "2015-01-01"
DATA_DIR    = os.path.join(os.path.dirname(__file__), "data")

# ---------------------------------------------------------------------------
# Data layer — cache to data/ folder, refresh only when stale
# ---------------------------------------------------------------------------

def last_business_day():
    """Return the previous business day (data is available for that day)."""
    today = pd.Timestamp.now().normalize()
    return (today - pd.offsets.BDay(1)).normalize()


def find_latest_cache(ticker):
    """Return the most recently saved CSV path for this ticker, or None."""
    pattern = os.path.join(DATA_DIR, f"{ticker}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_prices(ticker, start):
    last_bday = last_business_day()
    cached    = find_latest_cache(ticker)

    if cached:
        df = pd.read_csv(cached, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index)
        last_cached = df.index[-1].normalize()
        if last_cached >= last_bday:
            print(f"  [cache] Using existing data: {os.path.basename(cached)}")
            print(f"  [cache] Last date in cache : {last_cached.date()}  (up to date)")
            return df["Close"].asfreq("B").dropna()
        else:
            print(f"  [cache] Cache is outdated  : last={last_cached.date()}, expected>={last_bday.date()}")

    # Download fresh data
    print(f"  [download] Fetching {ticker} from {start} ...")
    raw = yf.download(ticker, start=start, progress=False)
    raw.index = pd.to_datetime(raw.index)

    # Flatten multi-level columns from yfinance and keep only Close
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "Close"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(DATA_DIR, f"{ticker}_{timestamp}.csv")
    close.to_csv(filename, header=True)
    print(f"  [download] Saved to: {os.path.basename(filename)}")

    return close.asfreq("B").dropna()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 50)
print("Loading Data")
print("=" * 50)
prices = load_prices(TICKER, START_DATE)
print(f"  Rows loaded: {len(prices)}  ({prices.index[0].date()} to {prices.index[-1].date()})")

# --- Stationarity Check (ADF Test) ---
print("\n" + "=" * 50)
print("ADF Stationarity Test")
print("=" * 50)
result = adfuller(prices)
print(f"  ADF Statistic : {result[0]:.4f}")
print(f"  P-value       : {result[1]:.4f}")
for key, val in result[4].items():
    print(f"  Critical [{key}] : {val:.4f}")
print("  Verdict:", "STATIONARY" if result[1] <= 0.05 else "NON-STATIONARY (differencing needed)")

# --- AIC/BIC Grid Search for best (p,d,q) ---
print("\n" + "=" * 50)
print("AIC/BIC Grid Search for best (p, d, q)")
print("=" * 50)

best_aic     = np.inf
best_order   = None
results_table = []

for p in range(0, 4):
    for d in range(0, 2):
        for q in range(0, 4):
            try:
                m = ARIMA(prices, order=(p, d, q)).fit()
                results_table.append((p, d, q, round(m.aic, 2), round(m.bic, 2)))
                if m.aic < best_aic:
                    best_aic   = m.aic
                    best_order = (p, d, q)
            except Exception:
                pass

results_table.sort(key=lambda x: x[3])
print(f"{'p':>3} {'d':>3} {'q':>3} {'AIC':>12} {'BIC':>12}")
print("-" * 38)
for row in results_table[:10]:
    print(f"{row[0]:>3} {row[1]:>3} {row[2]:>3} {row[3]:>12.2f} {row[4]:>12.2f}")
print(f"\n  Best order: ARIMA{best_order}  (AIC={best_aic:.2f})")

# --- Fit best model and forecast 30 business days ---
print("\n" + "=" * 50)
print(f"Fitting ARIMA{best_order} and forecasting 30 days")
print("=" * 50)

model_fit      = ARIMA(prices, order=best_order).fit()
forecast_result = model_fit.get_forecast(steps=30)
forecast_mean  = forecast_result.predicted_mean
conf_int       = forecast_result.conf_int(alpha=0.05)

future_dates        = pd.bdate_range(start=prices.index[-1] + pd.offsets.BDay(1), periods=30)
forecast_mean.index = future_dates
conf_int.index      = future_dates

print(forecast_mean.to_string())

# --- Plot ---
plt.figure(figsize=(12, 6))
plt.plot(prices[-200:], label="Historical (last 200 days)", color="steelblue")
plt.plot(forecast_mean, label="Forecast (30 days)", color="tomato", linewidth=2)
plt.fill_between(
    future_dates,
    conf_int.iloc[:, 0],
    conf_int.iloc[:, 1],
    color="tomato", alpha=0.2, label="95% Confidence Interval"
)
plt.title(f"AAPL Price Forecast — ARIMA{best_order}")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()
