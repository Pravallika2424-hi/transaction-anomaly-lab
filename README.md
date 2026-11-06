# Transaction Anomaly Lab

A small, reproducible anomaly detector for **synthetic** payment transactions. It combines robust amount scoring with unusual transaction velocity. This is a portfolio demonstration, not a fraud decision system and uses no employer data.

## Run

```bash
python app.py
python -m unittest discover -s tests
```

The script prints precision, recall, and F1 on a generated holdout set, plus five flagged examples. Only Python's standard library is required.

## Method

Training uses normal transactions only. The median and median absolute deviation (MAD) of `log1p(amount)` and `transactions_last_hour` produce robust z-scores. A transaction is flagged when either score exceeds a threshold selected on a separate validation set. The final metrics use an untouched test set. Synthetic anomalies are injected with higher amounts or velocities; results measure this toy generator only.

## Resume description

Built a reproducible Python demo for synthetic payment anomaly detection with robust feature scoring, validation threshold selection, and holdout precision/recall/F1 evaluation.

## Source
Imported from https://github.com/gurijalachaitanya483-web/transaction-anomaly-lab.

