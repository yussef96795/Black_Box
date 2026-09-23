# VWAP Displacement Intraday Momentum

## Hypothesis

Short-term directional persistence follows price displacement from the
intraday VWAP anchor: when price trades meaningfully above the volume-weighted
average price, the imbalance persists for the next several bars.

## Data Requirements

- BTCUSDT perpetual, 1-minute OHLCV, from 2020.
- Level 2 order book depth for slippage estimation.
- Funding rate history is NOT required.

## Entry Signal

Enter long when the close crosses above the VWAP anchor and cumulative volume
delta is positive for the last 10 minutes. Exit when price crosses below VWAP
or after 120 minutes in trade.

## Sizing & Risk

Position size is 1% nominal risk scaled by the inverse of ATR(14). A trailing
stop of 2 ATR protects against regime breaks. Sessions use the daily open as
the anchor reset point.