# ML-Based Network Intrusion Detection System

A network intrusion detection system that extracts flow-level features from raw traffic
and classifies them with XGBoost. Detects six classes: **Normal**, **SYN Flood**,
**HTTP Flood**, **C2 (command & control) beaconing**, **Brute Force**, and **SQL Injection**.

## How it works

**1. Flow extraction**
PCAPs are parsed with Scapy and packets are grouped into bidirectional flows by
`(src_ip, dst_ip, src_port, dst_port, protocol)`. A flow is considered complete when
it's idle for 120s, a FIN/RST is seen, or it hits 6 packets. Each flow produces 58 base
features covering timing stats, byte counts, TCP flags, periodicity, payload entropy,
and SQLi indicators.

**2. Feature engineering highlights**
- **Periodicity (C2 detection):** a periodogram over inter-arrival times catches
  fixed-interval beaconing (e.g. a C2 sample beaconing every ~7s) that simple packet
  counts miss.
- **TLS detection by content, not port:** checks whether the first payload byte is
  `0x16` (TLS handshake) instead of assuming TLS only runs on port 443 — catches
  encrypted C2 tunneled over non-standard ports.
- **SQLi payload inspection:** decodes up to 512 bytes of HTTP payload looking for
  injection keywords and time-based blind-SQLi patterns (`sleep()`, `pg_sleep`). A
  minimum keyword count is required before hard-flagging, to avoid false positives from
  benign queries that happen to contain words like "select" or "from".
- **Isolation Forest anomaly score**, trained on normal traffic only, added as an
  extra feature.

**3. Model**
Five classifiers were compared; **XGBoost** was chosen because it kept normal-traffic
recall above 0.98 (false positives are costly at scale) while training fast and
supporting sample weights natively for class imbalance. SQLi/C2 cost weights are
increased since those are the higher-value classes to catch.

**4. Handling class imbalance**
- **SMOTE** for Normal / Brute Force (enough real samples for reliable interpolation).
- **CTGAN** for C2 and SQLi (~300–450 real flows each — too few for SMOTE, which just
  produced near-duplicate flows and caused overfitting).

**5. Post-processing / alerting**
- Per-class confidence thresholds, tuned on a validation set.
- Hard overrides: strong SQLi payload evidence forces the SQLi label; high-confidence
  C2 probability forces the C2 label.
- **Temporal engine:** an alert only fires once the same source IP is seen 3+ times
  within a 10-second window — suppresses one-off/noisy detections that would otherwise
  be false alarms.

## Results

- XGBoost baseline kept normal recall > 0.98 across 5-fold CV.
- Confusion matrix: main confusion is SQLi vs. HTTP Flood (both HTTP-based; resolved
  mostly, not entirely, by the payload hard-flag override).
- On a 1,000-sample held-out test, the temporal engine collapsed raw detections to
  ~895 actionable alerts, suppressing isolated single-occurrence flags as noise.

Full metrics, confusion matrices, and learning curves are in the notebook outputs.

## Project structure

```
.
├── IDS.ipynb      # full pipeline: feature extraction -> training -> evaluation
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

Requires **Python 3.10+**.

Update the dataset path near the top of the notebook to point at your own PCAPs/CSV:

```python
data_path = r"C:\Users\yourname\DT_project\traffic\Dataset"
```

Then run the notebook top to bottom.

## Data

This repo does **not** include the PCAP captures, trained model, or feature CSVs —
only the pipeline. Point `data_path` at your own traffic captures to reproduce results.

## Notes / limitations

- Confidence thresholds and the temporal-engine window (10s / 3 hits) were tuned by
  trial and error on this project's validation set, not derived analytically — retune
  if you use different traffic.
- SQLi vs. HTTP Flood remains the model's weakest boundary; the payload hard-flag helps
  but doesn't fully resolve it.
- This is a research/coursework project, not a production-hardened IDS.
