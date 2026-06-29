"""
features.py — Candidate scoring logic for Redrob AI Engineer ranking.
Version 2 — Improved structured scoring for top-5 performance.

Three scores per candidate:
  - structured_score  (0-1): hard rules from JD requirements/disqualifiers
  - behavioral_score  (0-1): availability & responsiveness from redrob_signals
  - semantic_score    (0-1): computed externally via embeddings, passed in here

Final score = 0.35 * structured + 0.40 * semantic + 0.25 * behavioral

MENTOR NOTE v2 changes:
  - Semantic weight raised to 0.40 (strongest signal, catches career meaning)
  - Structured lowered to 0.35 (rules are good but semantic does it better)
  - Title relevance scoring completely overhauled — strong penalties for wrong titles
  - Career description keyword scoring added (rewards "shipped", "production", etc.)
  - Skill assessment scores from Redrob weighted more heavily
  - Irrelevant title penalty added (Civil Eng, Accountant, HR = strong penalty)
  - India location bonus increased
"""

from datetime import datetime, date
from typing import Any

# ── constants ──────────────────────────────────────────────────────────────────

EXP_IDEAL_MIN = 5.0
EXP_IDEAL_MAX = 9.0
EXP_HARD_MIN  = 3.0
EXP_HARD_MAX  = 12.0

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra", "mphasis",
    "hexaware", "l&t infotech", "ltimindtree", "mindtree",
    "patni", "mastech", "niit technologies", "firstsource",
}

PREFERRED_LOCATIONS = {
    "pune", "noida", "hyderabad", "mumbai", "delhi", "delhi ncr",
    "gurugram", "gurgaon", "bengaluru", "bangalore", "chennai",
    "india", "kolkata", "ahmedabad", "jaipur", "kochi",
}

# Must-have skill groups — synonyms grouped together
MUST_HAVE_SKILL_GROUPS = [
    {"embedding", "embeddings", "sentence-transformer", "sentence transformer",
     "semantic search", "dense retrieval", "bi-encoder", "cross-encoder",
     "bge", "e5", "openai embeddings", "text-embedding", "word2vec", "bert"},
    {"pinecone", "weaviate", "qdrant", "milvus", "faiss", "chroma",
     "opensearch", "elasticsearch", "pgvector", "vespa", "annoy", "vector database",
     "vector db", "vector search"},
    {"ranking", "retrieval", "bm25", "hybrid search", "reranking", "re-ranking",
     "learning to rank", "ltr", "ndcg", "mrr", "information retrieval",
     "recommendation", "recommendations", "recommendation system", "search ranking"},
    {"mlflow", "mlops", "model serving", "inference", "production ml",
     "a/b testing", "ab testing", "online evaluation", "feature store",
     "bentoml", "triton", "onnx", "ray serve", "model deployment"},
]

NICE_TO_HAVE_SKILLS = {
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "finetuning",
    "xgboost", "lightgbm", "learning to rank", "distributed systems",
    "open source", "ray", "triton", "onnx", "weights & biases", "wandb",
    "airflow", "spark", "kafka", "redis",
}

# STRONG positive title signals — these people are exactly what JD wants
STRONG_POSITIVE_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer",
    "nlp engineer", "search engineer", "ranking engineer",
    "recommendation systems engineer", "recommendations engineer",
    "applied scientist", "applied ml", "applied ai",
    "research engineer", "senior ml", "staff ml",
    "data scientist",  # only positive if they have right skills
    "software engineer",  # neutral — handled separately
}

# Weak positive — may or may not be relevant
WEAK_POSITIVE_TITLES = {
    "software engineer", "backend engineer", "data engineer",
    "platform engineer", "infrastructure engineer",
}

# STRONG negative — clearly wrong domain for this role
STRONG_NEGATIVE_TITLES = {
    "hr", "human resources", "recruiter", "talent",
    "accountant", "finance", "financial", "accounting",
    "civil engineer", "mechanical engineer", "electrical engineer",
    "graphic designer", "designer", "ux", "ui designer",
    "marketing", "content writer", "copywriter", "seo",
    "operations manager", "project manager", "product manager",
    "customer support", "customer success", "sales",
    "teacher", "professor", "lecturer",
    "mobile developer", "ios developer", "android developer",
    "game developer", "unity", "unreal",
    "devops", "sre", "site reliability",  # borderline — penalise lightly
}

CV_ROBOTICS_KEYWORDS = {
    "computer vision", "object detection", "image classification",
    "robotics", "ros", "slam", "point cloud", "lidar",
    "speech recognition", "text to speech", "tts", "asr",
    "yolo", "opencv", "image segmentation",
}

# Career description signals — reward candidates who used these words in job descriptions
PRODUCTION_SIGNALS = {
    "shipped", "production", "deployed", "launched", "scaled",
    "serving", "inference", "a/b test", "experiment", "online",
    "real-time", "low latency", "throughput", "pipeline",
}

RETRIEVAL_SIGNALS = {
    "ranking", "retrieval", "recommendation", "search", "embedding",
    "vector", "similarity", "relevance", "precision", "recall",
    "ndcg", "mrr", "click-through", "engagement",
}

TODAY = datetime.now().date()


# ── helpers ────────────────────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    return text.lower().strip()


def _months_since(date_str: str) -> float:
    try:
        d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
        return (TODAY - d).days / 30.0
    except Exception:
        return 999.0


def _skill_names(skills: list) -> set:
    return {_normalise(s["name"]) for s in skills}


def _career_text(candidate: dict) -> str:
    """Build one big string of candidate's career for semantic matching."""
    parts = []
    p = candidate["profile"]
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    for job in candidate.get("career_history", []):
        parts.append(job.get("title", ""))
        parts.append(job.get("description", ""))
    for s in candidate.get("skills", []):
        parts.append(s.get("name", ""))
    for e in candidate.get("education", []):
        parts.append(e.get("field_of_study", ""))
    return " ".join(filter(None, parts))


def _career_description_text(candidate: dict) -> str:
    """Extract all job description text for keyword scanning."""
    parts = []
    for job in candidate.get("career_history", []):
        parts.append(job.get("description", "").lower())
        parts.append(job.get("title", "").lower())
    parts.append(candidate["profile"].get("summary", "").lower())
    return " ".join(filter(None, parts))


# ── honeypot detection ─────────────────────────────────────────────────────────

def detect_honeypot(candidate: dict) -> bool:
    career = candidate.get("career_history", [])
    total_claimed = sum(j.get("duration_months", 0) for j in career)
    profile_months = candidate["profile"]["years_of_experience"] * 12
    if total_claimed > profile_months * 1.5 + 12:
        return True
    if profile_months < 60 and total_claimed > 120:
        return True
    return False


# ── structured score ───────────────────────────────────────────────────────────

def structured_score(candidate: dict) -> tuple:
    score = 0.0
    reasons = []
    p = candidate["profile"]
    skills = candidate.get("skills", [])
    career = candidate.get("career_history", [])
    skill_names = _skill_names(skills)
    title_lower = _normalise(p.get("current_title", ""))
    career_text = _career_description_text(candidate)
    signals = candidate.get("redrob_signals", {})

    # ── 1. HONEYPOT (instant zero) ─────────────────────────────────────────
    if detect_honeypot(candidate):
        return 0.0, ["HONEYPOT: timeline inconsistency detected"]

    # ── 2. TITLE RELEVANCE (max 0.20, min -0.25) ──────────────────────────
    # MENTOR NOTE v2: This is the biggest fix. Before, a Civil Engineer and
    # an ML Engineer got the same score here. Now wrong titles are penalised hard.
    is_strong_positive = any(t in title_lower for t in STRONG_POSITIVE_TITLES)
    is_weak_positive = any(t in title_lower for t in WEAK_POSITIVE_TITLES)
    is_strong_negative = any(t in title_lower for t in STRONG_NEGATIVE_TITLES)

    if is_strong_positive:
        score += 0.20
        reasons.append(f"strong title match: {p['current_title']}")
    elif is_weak_positive:
        score += 0.08
        reasons.append(f"weak title match: {p['current_title']}")
    elif is_strong_negative:
        score -= 0.25
        reasons.append(f"PENALTY: irrelevant title: {p['current_title']}")
    else:
        score += 0.0
        reasons.append(f"neutral title: {p['current_title']}")

    # ── 3. EXPERIENCE RANGE (max 0.18) ────────────────────────────────────
    yoe = p.get("years_of_experience", 0)
    if EXP_IDEAL_MIN <= yoe <= EXP_IDEAL_MAX:
        score += 0.18
        reasons.append(f"exp {yoe:.1f}yrs in ideal 5-9 range")
    elif EXP_HARD_MIN <= yoe < EXP_IDEAL_MIN:
        score += 0.08
        reasons.append(f"exp {yoe:.1f}yrs slightly below range")
    elif EXP_IDEAL_MAX < yoe <= EXP_HARD_MAX:
        score += 0.10
        reasons.append(f"exp {yoe:.1f}yrs slightly above range")
    else:
        score += 0.0
        reasons.append(f"exp {yoe:.1f}yrs outside acceptable range")

    # ── 4. CONSULTING-ONLY CHECK (max -0.20 penalty / +0.10 bonus) ────────
    if career:
        consulting_jobs = 0
        product_jobs = 0
        for job in career:
            company_lower = _normalise(job.get("company", ""))
            industry_lower = _normalise(job.get("industry", ""))
            is_consulting = (
                any(cf in company_lower for cf in CONSULTING_FIRMS) or
                "consulting" in industry_lower or
                "it services" in industry_lower
            )
            if is_consulting:
                consulting_jobs += 1
            else:
                product_jobs += 1

        if consulting_jobs > 0 and product_jobs == 0:
            score -= 0.20
            reasons.append("PENALTY: entire career in consulting/IT services")
        elif product_jobs > 0:
            score += 0.10
            reasons.append(f"product company experience ({product_jobs} roles)")

    # ── 5. MUST-HAVE SKILL GROUPS (max 0.20) ──────────────────────────────
    groups_matched = 0
    assessment_scores = signals.get("skill_assessment_scores", {})
    assessment_bonus = 0.0

    for group in MUST_HAVE_SKILL_GROUPS:
        matched = group & skill_names
        if matched:
            groups_matched += 1
            # Redrob-verified assessment scores add bonus
            for sk in matched:
                for assess_key, assess_val in assessment_scores.items():
                    if sk in _normalise(assess_key):
                        assessment_bonus += (assess_val / 100.0) * 0.015

    group_points = [0.0, 0.04, 0.10, 0.16, 0.20]
    score += group_points[groups_matched]
    score += min(assessment_bonus, 0.05)
    reasons.append(f"{groups_matched}/4 must-have skill groups matched")

    # ── 6. CAREER DESCRIPTION SIGNALS (max 0.12) ──────────────────────────
    # MENTOR NOTE v2: NEW — rewards candidates who actually describe production work
    # "shipped ranking models" >> "familiar with ranking concepts"
    production_hits = sum(1 for sig in PRODUCTION_SIGNALS if sig in career_text)
    retrieval_hits = sum(1 for sig in RETRIEVAL_SIGNALS if sig in career_text)

    prod_bonus = min(production_hits * 0.015, 0.07)
    ret_bonus = min(retrieval_hits * 0.015, 0.05)
    score += prod_bonus + ret_bonus
    if production_hits >= 3:
        reasons.append(f"production work signals in career ({production_hits} hits)")
    if retrieval_hits >= 3:
        reasons.append(f"retrieval/ranking signals in career ({retrieval_hits} hits)")

    # ── 7. NICE-TO-HAVE SKILLS (max 0.05) ─────────────────────────────────
    nice_matched = NICE_TO_HAVE_SKILLS & skill_names
    if nice_matched:
        score += min(len(nice_matched) * 0.01, 0.05)
        reasons.append(f"nice-to-have: {', '.join(list(nice_matched)[:3])}")

    # ── 8. CV/ROBOTICS/SPEECH BACKGROUND (penalty) ────────────────────────
    cv_skills = CV_ROBOTICS_KEYWORDS & skill_names
    nlp_ir_skills = {"nlp", "natural language processing", "information retrieval",
                     "text", "ranking", "retrieval", "recommendation"} & skill_names
    if len(cv_skills) >= 3 and len(nlp_ir_skills) == 0:
        score -= 0.15
        reasons.append("PENALTY: CV/speech/robotics background without NLP/IR")

    # ── 9. PURE RESEARCH DISQUALIFIER ─────────────────────────────────────
    research_titles = {"researcher", "research scientist", "phd student",
                       "postdoc", "professor", "academic"}
    if any(rt in title_lower for rt in research_titles):
        has_production = any(
            j.get("industry", "") not in ("Academia", "Research")
            for j in career
        )
        if not has_production:
            score -= 0.25
            reasons.append("DISQUALIFIER: pure research, no production")

    # ── 10. LOCATION (max 0.10) ────────────────────────────────────────────
    location_lower = _normalise(p.get("location", ""))
    country_lower = _normalise(p.get("country", ""))
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if any(loc in location_lower for loc in PREFERRED_LOCATIONS):
        score += 0.10
        reasons.append(f"location match: {p['location']}")
    elif country_lower == "india":
        score += 0.07
        reasons.append("India-based (non-preferred city)")
    elif willing_to_relocate:
        score += 0.03
        reasons.append("willing to relocate")
    else:
        score -= 0.03
        reasons.append(f"location mismatch: {p.get('location', 'unknown')}")

    score = max(0.0, min(1.0, score))
    return score, reasons


# ── behavioral score ───────────────────────────────────────────────────────────

def behavioral_score(candidate: dict) -> tuple:
    signals = candidate.get("redrob_signals", {})
    score = 0.0
    reasons = []

    # Recency (max 0.25)
    months_inactive = _months_since(signals.get("last_active_date", "2020-01-01"))
    if months_inactive <= 1:
        score += 0.25
        reasons.append("active in last month")
    elif months_inactive <= 3:
        score += 0.18
        reasons.append(f"active {months_inactive:.0f}mo ago")
    elif months_inactive <= 6:
        score += 0.10
        reasons.append(f"active {months_inactive:.0f}mo ago (borderline)")
    else:
        score += 0.0
        reasons.append(f"inactive {months_inactive:.0f}mo")

    # Open to work (max 0.20)
    if signals.get("open_to_work_flag", False):
        score += 0.20
        reasons.append("open to work")
    else:
        score += 0.05

    # Response rate (max 0.20)
    rr = signals.get("recruiter_response_rate", 0.0)
    score += rr * 0.20
    reasons.append(f"response rate {rr:.0%}")

    # Notice period (max 0.15)
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 0.15
        reasons.append(f"notice {notice}d (preferred)")
    elif notice <= 60:
        score += 0.08
        reasons.append(f"notice {notice}d (acceptable)")
    else:
        score += 0.02
        reasons.append(f"notice {notice}d (high)")

    # Interview completion (max 0.10)
    icr = signals.get("interview_completion_rate", 0.5)
    score += icr * 0.10
    reasons.append(f"interview completion {icr:.0%}")

    # GitHub activity (max 0.10)
    gh = signals.get("github_activity_score", -1)
    if gh == -1:
        score += 0.03  # v2: small neutral bonus instead of 0
    elif gh >= 70:
        score += 0.10
        reasons.append(f"GitHub active ({gh})")
    elif gh >= 40:
        score += 0.06
        reasons.append(f"GitHub moderate ({gh})")
    else:
        score += 0.02

    score = max(0.0, min(1.0, score))
    return score, reasons


# ── final combiner ─────────────────────────────────────────────────────────────

def final_score(candidate: dict, semantic: float) -> tuple:
    """
    MENTOR NOTE v2: Weights updated after analysis:
    - Semantic raised to 0.40: strongest signal, catches career meaning perfectly
    - Structured to 0.35: good rules but semantic handles nuance better
    - Behavioral stays 0.25: availability multiplier
    """
    s_score, s_reasons = structured_score(candidate)
    b_score, b_reasons = behavioral_score(candidate)

    if s_score == 0.0 and s_reasons and "HONEYPOT" in s_reasons[0]:
        return 0.0, "HONEYPOT: disqualified"

    # v2 weights: semantic is now highest
    combined = (0.35 * s_score) + (0.40 * semantic) + (0.25 * b_score)
    combined = round(max(0.0, min(1.0, combined)), 4)

    p = candidate["profile"]
    top_reasons = s_reasons[:2] + b_reasons[:1]
    reasoning = (
        f"{p['current_title']} | {p['years_of_experience']}yrs | "
        f"struct={s_score:.2f} sem={semantic:.2f} beh={b_score:.2f} | "
        + "; ".join(top_reasons)
    )

    return combined, reasoning


# ── quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    data = json.load(open("sample_candidates.json"))
    print(f"Testing on {len(data)} sample candidates...\n")

    results = []
    for c in data:
        s, sr = structured_score(c)
        b, br = behavioral_score(c)
        f, reason = final_score(c, semantic=0.5)
        results.append((f, c["candidate_id"], c["profile"]["current_title"],
                        c["profile"]["years_of_experience"], s, b, reason))

    results.sort(reverse=True)
    print(f"{'Rank':<5} {'ID':<15} {'Title':<35} {'YoE':<6} {'Str':<6} {'Beh':<6} {'Final'}")
    print("-" * 95)
    for i, (f, cid, title, yoe, s, b, reason) in enumerate(results[:20], 1):
        print(f"{i:<5} {cid:<15} {title:<35} {yoe:<6.1f} {s:<6.2f} {b:<6.2f} {f:.4f}")
