"""
Reconstructs, for a single incoming question, the exact same feature pipeline
used during training. Every feature here has a direct counterpart in the
Kaggle notebook. See the comments for which training step each one mirrors.
"""

import math
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .model_loader import ModelBundle
from .schemas import QuestionInput

CODE_FENCE_RE = re.compile(r"```")
HTML_CODE_TAG_RE = re.compile(r"<code>", re.IGNORECASE)


def detect_code_block(body: str) -> int:
    """
    Training data stored question bodies as Stack Overflow's HTML, where a
    code block is a <code> tag. A live user pasting a draft question is more
    likely to use markdown-style ``` fences or indented blocks, so we check
    for all three common conventions.
    """
    if HTML_CODE_TAG_RE.search(body):
        return 1
    if CODE_FENCE_RE.search(body):
        return 1
    lines = body.split("\n")
    indented_lines = sum(1 for line in lines if line.startswith("    ") or line.startswith("\t"))
    if indented_lines >= 2:
        return 1
    return 0


def assign_bucket_index(value: float, edges: List[float]) -> int:
    """
    Mirrors pd.qcut(..., include_lowest=True) bucket assignment: bucket i
    covers (edges[i], edges[i+1]], with the first bucket also including the
    exact minimum. Values outside the observed training range are clipped
    into the nearest edge bucket rather than raising an error, since a live
    question can legitimately have a body length outside anything seen in
    training (e.g. unusually long).
    """
    n_buckets = len(edges) - 1
    if value <= edges[1]:
        return 0
    for i in range(1, n_buckets):
        if edges[i] < value <= edges[i + 1]:
            return i
    return n_buckets - 1


def compute_tag_target_score(tags: List[str], bundle: ModelBundle) -> float:
    tag_rates: Dict[str, float] = bundle.tag_rates["tag_rates"]
    fallback = bundle.tag_rates["global_fallback_rate"]
    cleaned = [t.strip().lower() for t in tags if t.strip()]
    if not cleaned:
        return fallback
    scores = [tag_rates.get(t, fallback) for t in cleaned]
    return float(np.mean(scores))


def build_raw_row(question: QuestionInput, bundle: ModelBundle) -> dict:
    """Computes every engineered value once; both feature dataframes below pull from this dict."""
    now = question.posted_at or datetime.now(timezone.utc)

    title_length = len(question.title)
    body_length = len(question.body)
    has_code_block = detect_code_block(question.body)
    cleaned_tags = [t.strip().lower() for t in question.tags if t.strip()]
    num_tags = len(cleaned_tags)

    edges = bundle.body_length_bucket_edges
    categories = bundle.body_length_bucket_categories
    bucket_idx = assign_bucket_index(body_length, edges)
    body_length_bucket = categories[bucket_idx]

    hour = now.hour
    post_hour_sin = math.sin(2 * math.pi * hour / 24)
    post_hour_cos = math.cos(2 * math.pi * hour / 24)

    asker = question.asker
    reputation = asker.reputation if asker else None
    upvotes = asker.upvote_count if asker else None
    downvotes = asker.downvote_count if asker else None
    account_created = asker.account_created if asker else None

    reputation_missing = int(reputation is None)
    upvotes_missing = int(upvotes is None)
    downvotes_missing = int(downvotes is None)

    if account_created is not None:
        account_age_days = (now - account_created).total_seconds() / 86400
        account_age_missing = 0
    else:
        account_age_days = -1
        account_age_missing = 1

    reputation_val = reputation if reputation is not None else -1
    upvotes_val = upvotes if upvotes is not None else -1
    downvotes_val = downvotes if downvotes is not None else -1

    # log1p mirrors training exactly; sentinel -1 values pass through log1p(max(x, 0))
    # so the "missing" case always maps to log1p(0) = 0, matching training behaviour
    # where the -1 fill was log-transformed the same way.
    asker_reputation_log = np.log1p(max(reputation_val, 0))
    asker_upvote_count_log = np.log1p(max(upvotes_val, 0))
    asker_downvote_count_log = np.log1p(max(downvotes_val, 0))
    asker_account_age_days_log = np.log1p(max(account_age_days, 0))

    is_new_asker = int(reputation is not None and reputation <= 1)

    tag_target_score = compute_tag_target_score(cleaned_tags, bundle)

    return {
        "title_length": title_length,
        "body_length": body_length,
        "body_length_bucket": body_length_bucket,
        "has_code_block": has_code_block,
        "num_tags": num_tags,
        "post_hour_sin": post_hour_sin,
        "post_hour_cos": post_hour_cos,
        "asker_reputation_log": asker_reputation_log,
        "asker_upvote_count_log": asker_upvote_count_log,
        "asker_downvote_count_log": asker_downvote_count_log,
        "asker_account_age_days_log": asker_account_age_days_log,
        "is_new_asker": is_new_asker,
        "asker_reputation_missing": reputation_missing,
        "asker_upvote_count_missing": upvotes_missing,
        "asker_downvote_count_missing": downvotes_missing,
        "asker_account_age_days_missing": account_age_missing,
        "tag_target_score": tag_target_score,
        "recent_platform_answer_rate": bundle.platform_context["recent_platform_answer_rate"],
        "recent_platform_median_speed": bundle.platform_context["recent_platform_median_speed_hours"],
    }


def build_classification_df(row: dict, bundle: ModelBundle) -> pd.DataFrame:
    df = pd.DataFrame([row])[bundle.feature_cols]
    df["body_length_bucket"] = pd.Categorical(
        df["body_length_bucket"], categories=bundle.body_length_bucket_categories
    )
    return df


def build_regression_df(row: dict, bundle: ModelBundle) -> pd.DataFrame:
    df = pd.DataFrame([row])[bundle.feature_cols_reg]
    df["body_length_bucket"] = pd.Categorical(
        df["body_length_bucket"], categories=bundle.body_length_bucket_categories
    )
    return df
