# US Equities Open Auction Breakout

## Hypothesis

The first 30 minutes after the primary exchange open resolve institutional
order flow, and breakouts above the opening auction range predict
continuation.

## Data Requirements

- QQQ daily and 1-minute bars from 2015.
- Level 3 order book for execution analysis.

## Entry Signal

Buy above the opening range high with a session-time filter; exit at the
VWAP cross or end-of-session close.

## Sizing & Risk

Risk 0.5% per trade; hard stop at 10 bps of slippage allowance.