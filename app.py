"""Robust anomaly detection on synthetic transactions; no external dependencies."""
import math
import random
from statistics import median


def make_data(seed=42, normal=600, anomalous=60):
    rng = random.Random(seed)
    rows = []
    for _ in range(normal):
        rows.append((round(rng.lognormvariate(3.3, 0.45), 2), rng.randrange(1, 6), 0))
    for i in range(anomalous):
        amount = rng.uniform(250, 900) if i % 2 == 0 else rng.uniform(15, 90)
        velocity = rng.randrange(1, 6) if i % 2 == 0 else rng.randrange(13, 28)
        rows.append((round(amount, 2), velocity, 1))
    rng.shuffle(rows)
    return rows


def fit(rows):
    if not rows or any(label for _, _, label in rows):
        raise ValueError("fit requires nonempty normal-only data")
    columns = ([math.log1p(r[0]) for r in rows], [r[1] for r in rows])
    centers = [median(col) for col in columns]
    scales = [max(median(abs(x - c) for x in col) * 1.4826, 0.1) for col, c in zip(columns, centers)]
    return centers, scales


def score(row, model):
    centers, scales = model
    values = (math.log1p(row[0]), row[1])
    return max((x - c) / s for x, c, s in zip(values, centers, scales))


def metrics(rows, model, threshold):
    tp = fp = fn = 0
    for row in rows:
        predicted = score(row, model) >= threshold
        tp += predicted and row[2] == 1
        fp += predicted and row[2] == 0
        fn += not predicted and row[2] == 1
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return precision, recall, f1


def main():
    train = make_data(10, 600, 0)
    validation = make_data(11, 200, 20)
    test = make_data(12, 200, 20)
    model = fit(train)
    threshold = max((2.5, 3, 3.5, 4, 5, 6), key=lambda t: metrics(validation, model, t)[2])
    p, r, f1 = metrics(test, model, threshold)
    print(f"Validation threshold: {threshold}; test precision={p:.3f}, recall={r:.3f}, F1={f1:.3f}")
    for amount, velocity, label in sorted(test, key=lambda row: score(row, model), reverse=True)[:5]:
        print(f"amount={amount:7.2f} hourly_count={velocity:2d} synthetic_label={label}")


if __name__ == "__main__":
    main()
