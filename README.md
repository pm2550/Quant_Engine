# Quant Engine

A personal quantitative trading research / decision-support pipeline for US
equities and Chinese A-shares.  Multi-factor signal scoring, walk-forward
backtests, news / event monitoring, intraday alerts, fundamentals from SEC
EDGAR, and a LightGBM challenger model that is compared side-by-side with
the rule-based composite.

Built for a single-operator workflow — daily Telegram digest, ad-hoc
chatbot consultation, and 30-day decision review for self-calibration.

## Status

- 314 unit tests passing
- ~20 systemd units for ingest / alerting / reporting
- Public repo, but personal portfolio and credentials are gitignored
- Trains on ~20 years of historical OHLCV; SEC EDGAR fundamentals back to ~2009

## Architecture

```
   yfinance / akshare           SEC EDGAR (XBRL)            FRED / yfinance macro
        OHLCV                  fundamentals JSON                VIX, yields, DXY
          │                          │                                │
          ▼                          ▼                                ▼
     data/prices/*.parquet     data/edgar/CIK*.json              data/macro/*
          │                          │                                │
          └──────────┬───────────────┴────────────────────────────────┘
                     │
            ┌────────┴─────────────────────────────────────────┐
            │                                                  │
            ▼                                                  ▼
   Rule-based 11-factor composite                LightGBM Challenger (Alpha158-style
   (multi_factor.py: technical / events /         + macro + SEC fundamentals,
   sentiment / fundamental / analyst /             161 features, walk-forward IC ≈ 0.05)
   momentum / macro_regime / alt_data /
   rating_change / event_intensity / trade_signals)
            │                                                  │
            └────────────────────┬─────────────────────────────┘
                                 │
                                 ▼
                      Daily report + Telegram digest
                      (recommendations + challenger predictions
                       side-by-side, disagreement flagged)
```

## Layout

| Path | Contents |
|------|----------|
| `quant/` | Core modules (orchestrator, fetcher, signals, recommender, multi_factor, ...) |
| `quant/ml/` | LightGBM challenger: features (Alpha158 port), macro, SEC EDGAR, predict, serve |
| `quant/alt_data/` | Alternative data ingestion (e.g. Bilibili interest as leading indicator for CN gaming) |
| `prompts/` | LLM prompt registry (versioned `.md` files) |
| `config/` | YAML config: portfolio (gitignored), LLM routes, sources, opportunity universe |
| `tests/` | 310+ unit tests + ML feature/macro/edgar tests |
| `systemd/` | Service + timer unit files |
| `scripts/` | Operational scripts (one-offs, backfills) |

## ML Challenger

A walk-forward-validated LightGBM regressor that predicts forward 20-day returns.
Compared with the rule-based composite in every daily digest; disagreement
between the two is surfaced explicitly.

Features (161):
- **144 technical** — pandas port of Microsoft Qlib's `Alpha158DL`: KBAR, price ratios,
  rolling moments (MA / STD / ROC / MAX / MIN / RSV / Bollinger-like quantiles /
  index-of-extreme positions / Aroon-like time-since / volume RSI variants)
- **7 macro context** — VIX percentile rank, VIX trend, 10Y yield change in bps,
  yield-curve regime percentile, DXY / gold / oil 60-day pct change.
  *Only stationary transforms* — raw levels caused calendar-period
  memorization (cross-sectional poison; see commit history for the ablation)
- **10 SEC EDGAR fundamentals** — TTM revenue / EPS, YoY growth, net /
  gross / operating margin, debt-to-equity, OCF-to-revenue, days-since-filing.
  Free SEC Company Facts API; 15+ year depth back to XBRL mandate (2009)

Hyperparameters tuned for ~95-symbol universe (Qlib's CSI300 defaults
over-regularized us and shrunk predictions to population mean).

Out-of-sample (4-fold walk-forward, 1y validation per fold, 20y total):
```
median IC                +0.049
median RankIC            +0.040
median top-decile spread +3.5% (20-day forward return)
```

## Data sources (all free)

| Source | Coverage |
|--------|----------|
| yfinance | US equities + ETFs daily OHLCV, fundamentals snapshot, options expiries |
| akshare | A-share OHLCV + fundamentals + news (东方财富, Sina backup) |
| SEC EDGAR | Quarterly fundamentals XBRL since 2009 (Company Facts API) |
| FRED + yfinance index | Macro (VIX, yields, DXY, commodities) |
| RSS feeds | News (configurable in `config/sources.yaml`) |
| Bilibili public API | Gaming-related interest snapshot (alt data for CN gaming positions) |

## Setup

```bash
# 1. Clone + Python venv (3.10+)
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Copy example configs (personal info is gitignored)
cp config/portfolio.example.yaml config/portfolio.yaml
# fill in your positions + telegram_target chat id

cp config/llm.example.yaml config/llm.yaml
# add your dashscope / Gemini / TG bot token

# 3. Initialize DB
python -c "from quant import db; db.init()"

# 4. Backfill historical prices
python -m quant.fetcher --full-refresh

# 5. (Optional) ML challenger — separate venv to keep prod venv slim
python -m venv qlib_env
qlib_env/bin/pip install lightgbm scikit-learn pandas pyarrow yfinance requests
qlib_env/bin/python -m quant.ml.edgar              # backfill fundamentals (~30s)
qlib_env/bin/python -m quant.ml.macro              # backfill macro
qlib_env/bin/python -m quant.ml.challenger --train-full

# 6. Install systemd units
sudo cp systemd/*.service systemd/*.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now quant-daily.timer
# ... and the rest as needed
```

## Testing

```bash
# Core
source venv/bin/activate && python -m pytest tests/ -x --tb=short -q

# ML (in separate venv)
qlib_env/bin/python -m pytest tests/test_ml_features.py tests/test_ml_macro.py tests/test_ml_edgar.py -x -q
```

## Caveats / known limitations

- Sample universe is ~95 symbols.  Some published quant ML tricks (cross-sectional
  ranking; Qlib's stronger regularization defaults) need 300+ instruments to
  outperform — we tested and disabled them
- A-share fundamentals not yet wired into the challenger (akshare can provide
  these; not done yet)
- News sentiment from FinBERT is *not* used as an ML feature — our news archive
  is too short (started 2026-05-06) for it to add training signal
- Forward analyst estimates are unavailable on free tiers
- The 11-factor composite uses heuristics for some inputs (analyst ratings,
  sentiment, alt-data) that have stronger paid alternatives

## License

MIT. Use at your own risk. This is research code for personal use; nothing
here is investment advice, and walk-forward IC of ~0.05 is the floor for
"signal exists" not "guaranteed profit". Real trading also needs transaction
cost modeling, position sizing, risk controls, and discretion that this
pipeline only partially captures.
