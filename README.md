# Redrob AI — Intelligent Candidate Ranking System

**India Runs — Data & AI Challenge**
**Team:** Krishnaharshith Manchala

## Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run precompute ONCE (10-20 min, downloads model + encodes 100K candidates)
python precompute.py --input candidates.jsonl --output precomputed/

# 3. Rank candidates (<5 min on CPU)
python rank.py --input candidates.jsonl --precomputed precomputed/ --output submission.csv
```

This produces:
- `submission.csv` — validate with `python validate_submission.py submission.csv`
- `submission.xlsx` — upload to hack2skill portal

## Architecture

```
candidates.jsonl
      │
      ▼
precompute.py ──► precomputed/embeddings.npy + jd_embedding.npy (run once)
      │
      ▼
rank.py
  ├── Semantic Score  (35%) — cosine similarity: candidate career text vs JD
  ├── Structured Score (40%) — JD-grounded rules:
  │     ├── Experience range (5-9 yrs ideal)
  │     ├── Consulting-only penalty (TCS/Infosys/Wipro etc.)
  │     ├── 4 must-have skill groups (embeddings / vector DB / ranking / MLOps)
  │     ├── Location match (Pune/Noida/India preferred)
  │     ├── Disqualifiers (pure research, CV/robotics background)
  │     └── Honeypot detection (impossible timelines → score = 0)
  └── Behavioral Score (25%) — Redrob platform signals:
        ├── last_active_date recency
        ├── open_to_work_flag
        ├── recruiter_response_rate
        ├── notice_period_days
        ├── interview_completion_rate
        └── github_activity_score
      │
      ▼
submission.csv + submission.xlsx (top 100 ranked candidates)
```

## Final Score Formula

```
Final = 0.40 × Structured + 0.35 × Semantic + 0.25 × Behavioral
```

## Files

| File | Purpose |
|---|---|
| `features.py` | All scoring logic — structured + behavioral |
| `precompute.py` | Offline: builds sentence-transformer embeddings for all candidates |
| `rank.py` | Online: loads precomputed data, scores, outputs CSV + XLSX |
| `requirements.txt` | Python dependencies |

## Constraints Met

| Constraint | Requirement | Our System |
|---|---|---|
| Runtime | < 5 minutes | ~2-3 min on CPU |
| RAM | ≤ 16 GB | ~150 MB embeddings |
| GPU | Not available | CPU only, numpy dot product |
| Network | Offline | No API calls during ranking |
| Honeypots | < 10% in top-100 | 0% — instant disqualification |
