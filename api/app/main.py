import math

import numpy as np
from fastapi import FastAPI

from .features import build_classification_df, build_regression_df, build_raw_row
from .model_loader import get_model_bundle
from .schemas import PredictionResponse, QuestionInput, TimeEstimate, TopFactor

app = FastAPI(
    title="Stack Overflow Answer Predictor",
    description="Predicts whether a draft question is likely to be answered, "
    "an expected response time range, and the key factors driving the prediction.",
    version="1.0.0",
)

# Human-readable descriptions for each feature, used to turn raw SHAP output
# into plain-language explanations. Kept separate from the model itself so the
# wording can be tuned freely without touching any prediction logic.
FEATURE_LABELS = {
    "tag_target_score": "the historical answer rate for these tags",
    "recent_platform_answer_rate": "how often questions have been getting answered recently",
    "recent_platform_median_speed": "how quickly questions have been getting answered recently",
    "asker_reputation_log": "your reputation level",
    "asker_upvote_count_log": "your upvote history",
    "asker_downvote_count_log": "your downvote history",
    "asker_account_age_days_log": "how long your account has existed",
    "is_new_asker": "being a very new account",
    "has_code_block": "including a code block",
    "body_length": "the length of your question body",
    "body_length_bucket": "the length category of your question body",
    "title_length": "the length of your title",
    "num_tags": "the number of tags used",
    "post_hour_sin": "the time of day posted",
    "post_hour_cos": "the time of day posted",
    "asker_reputation_missing": "not providing reputation info",
    "asker_upvote_count_missing": "not providing upvote info",
    "asker_downvote_count_missing": "not providing downvote info",
    "asker_account_age_days_missing": "not providing account age info",
}


@app.on_event("startup")
def _warm_up_models():
    # Forces the model bundle (and its SHAP explainer) to load once at startup
    # rather than on the first incoming request, so the first real user isn't
    # the one who pays the load-time cost.
    get_model_bundle()


def _format_hours(hours: float) -> str:
    if hours < 1:
        minutes = max(int(round(hours * 60)), 1)
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    if hours < 24:
        rounded = max(int(round(hours)), 1)
        return f"{rounded} hour{'s' if rounded != 1 else ''}"
    if hours < 168:
        days = max(int(round(hours / 24)), 1)
        return f"{days} day{'s' if days != 1 else ''}"
    weeks = max(int(round(hours / 168)), 1)
    return f"{weeks} week{'s' if weeks != 1 else ''}"


def _explain_prediction(shap_row, feature_values: dict, top_n: int = 4):
    contributions = sorted(
        zip(shap_row, feature_values.keys(), feature_values.values()),
        key=lambda x: abs(x[0]),
        reverse=True,
    )
    factors = []
    for shap_val, feature, value in contributions[:top_n]:
        label = FEATURE_LABELS.get(feature, feature)
        direction = "increases" if shap_val > 0 else "decreases"
        factors.append(
            TopFactor(
                feature=feature,
                plain_language=label,
                direction=direction,
                impact=round(float(shap_val), 4),
            )
        )
    return factors


@app.post("/predict", response_model=PredictionResponse)
def predict(question: QuestionInput) -> PredictionResponse:
    bundle = get_model_bundle()

    raw_row = build_raw_row(question, bundle)

    # --- Classification: will this be answered? ---
    clf_df = build_classification_df(raw_row, bundle)
    proba_answered = float(bundle.clf_model.predict(clf_df)[0])
    will_likely_be_answered = proba_answered >= bundle.final_threshold

    shap_values = bundle.clf_explainer.shap_values(clf_df)
    shap_row = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    top_factors = _explain_prediction(shap_row, raw_row)

    # --- Regression: how long might it take? ---
    reg_df = build_regression_df(raw_row, bundle)
    log_lower = bundle.reg_lower.predict(reg_df)[0]
    log_median = bundle.reg_median.predict(reg_df)[0]
    log_upper = bundle.reg_upper.predict(reg_df)[0]

    hours_lower = float(np.expm1(log_lower))
    hours_median = float(np.expm1(log_median))
    hours_upper = float(np.expm1(log_upper))
    # Guard against rare quantile-crossing (independently trained models can
    # occasionally disagree on ordering) - clip to keep the range sensible.
    hours_lower = min(hours_lower, hours_median)
    hours_upper = max(hours_upper, hours_median)

    time_estimate = TimeEstimate(
        likely_time=_format_hours(hours_median),
        range_low=_format_hours(hours_lower),
        range_high=_format_hours(hours_upper),
        summary=(
            f"If answered, typically within {_format_hours(hours_median)} "
            f"(likely between {_format_hours(hours_lower)} and {_format_hours(hours_upper)})"
        ),
    )

    if will_likely_be_answered:
        summary = (
            f"This question looks likely to be answered "
            f"({proba_answered:.0%} estimated probability)."
        )
    else:
        summary = (
            f"This question may struggle to get answered "
            f"({proba_answered:.0%} estimated probability). "
            f"Consider the factors below."
        )

    return PredictionResponse(
        probability_answered=round(proba_answered, 4),
        will_likely_be_answered=will_likely_be_answered,
        decision_threshold=bundle.final_threshold,
        expected_time_to_answer=time_estimate,
        top_factors=top_factors,
        summary=summary,
    )


@app.get("/health")
def health():
    return {"status": "ok"}
