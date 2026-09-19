"""Sample quant strategy markdown used by ingestion tests (mirrors /tmp fixture)."""
# Bollinger Band Mean Reversion Strategy

## Overview

This strategy trades mean reversion on the 15-minute BTC-USDT chart. It buys
when price touches the lower Bollinger Band and exits when price returns to the
middle band. A trailing stop of 1.5 ATR protects against regime breaks.

## Parameters

| Param                | Value | Bounds     |
| -------------------- | ----- | ---------- |
| lookback             | 20    | [10, 60]   |
| entry_std            | 2.0   | [1.0, 3.0] |
| exit_std             | 0.0   | [0.0, 1.0] |
| stop_loss_atr        | 1.5   | [1.0, 3.0] |
| risk_per_trade_pct   | 1.0   | [0.5, 2.0] |

## Signal

Buy when `close_t < lower_band_t` and RSI(14) < 30. Sell when `close_t >
upper_band_t` or when the 20-period rolling volatility exceeds 4% per bar.

The annualized volatility target is σ = 15%, and position size is scaled by
the inverse of realized volatility:

```text
position = (risk_per_trade_pct * equity) / (atr * price)
```

## Exit Rules

- Exit long when price crosses back above the middle band.
- Hard stop-loss at 1.5 × ATR(14) from entry.
- Daily loss limit: 5% of starting equity; halt trading for 30 minutes.
- Maximum 3 consecutive losing trades before a 60-minute cooldown.

## Data Requirements

Requires Level 3 order book depth for slippage estimation, 1-minute OHLCV
history for the last 2 years, and funding rate history. Execution assumes a
centralized exchange with maker rebates and taker fees of 10 bps.