"""Remake the random 60/40 dev/test split of the gold questions in eval/gold/gold_60_40.csv.

Questions are split within each format and evidence type (slides text, slides image, documents text,
documents image), so each keeps the same dev share. The out-of-scope questions have no split and stay that way.

    .venv/bin/python eval/make_split.py

Saves eval/gold/gold_60_40.csv and eval/results_60_40/gold_split.csv.
"""
import csv
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "eval" / "gold" / "gold_60_40.csv"
OUT = ROOT / "eval" / "results_60_40"
DEV_SHARE = 0.6
SEED = 469  # chosen before any result on this split was seen, and drawn once


def main():
    with TARGET.open(newline="") as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)

    rng = random.Random(SEED)
    groups = {}
    for row in rows:
        if row["split"]:
            groups.setdefault((row["corpus"], row["evidence_type"]), []).append(row)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda r: r["id"])
        rng.shuffle(group)
        n_dev = round(len(group) * DEV_SHARE)
        for i, row in enumerate(group):
            row["split"] = "dev" if i < n_dev else "test"

    with TARGET.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter()
    for row in rows:
        if row["split"]:
            labels = [("overall", "all"), ("evidence_type", row["evidence_type"]),
                      ("format", "documents" if row["corpus"] == "longdoc" else "slides"),
                      ("format_and_evidence", f"{'documents' if row['corpus'] == 'longdoc' else 'slides'} {row['evidence_type']}"),
                      ("collection", row["collection"])]
            for label in labels:
                counts[(*label, row["split"])] += 1
    OUT.mkdir(exist_ok=True)
    with (OUT / "gold_split.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["group", "value", "dev", "test", "total", "dev_share"])
        for group, value in sorted({key[:2] for key in counts}):
            dev, test = counts[(group, value, "dev")], counts[(group, value, "test")]
            writer.writerow([group, value, dev, test, dev + test, round(dev / (dev + test), 3)])
    print(f"Saved {TARGET} and {OUT / 'gold_split.csv'}")


if __name__ == "__main__":
    main()
