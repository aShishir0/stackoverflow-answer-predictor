"""
Streamlit landing page for the Stack Overflow Answer Predictor.

Calls the deployed FastAPI service's /predict endpoint and renders the
result (answer probability, expected time range, and top contributing
factors) in a readable, non-technical format.
"""

import os
from datetime import datetime, timezone

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "https://stackoverflow-answer-predictor.onrender.com")

st.set_page_config(
    page_title="Will My Question Get Answered?",
    page_icon="❓",
    layout="centered",
)

st.title("❓ Will My Question Get Answered?")
st.write(
    "Paste a draft Stack Overflow question below to get a prediction of "
    "whether it's likely to be answered, roughly how long that might take, "
    "and what's driving the prediction."
)

with st.form("question_form"):
    title = st.text_input(
        "Title",
        placeholder="How to filter a pandas dataframe by multiple conditions?",
        max_chars=300,
    )
    body = st.text_area(
        "Body",
        placeholder=(
            "Describe your problem in detail. Include a code block "
            "(``` fences or 4-space indentation) if relevant - it makes "
            "a real difference to your odds of getting answered."
        ),
        height=220,
    )
    tags_input = st.text_input(
        "Tags (comma-separated)",
        placeholder="python, pandas",
    )

    st.markdown("**Your account info** _(optional - leave blank if you'd rather stay anonymous)_")
    col1, col2, col3 = st.columns(3)
    with col1:
        reputation = st.number_input("Reputation", min_value=0, value=None, step=1)
    with col2:
        upvote_count = st.number_input("Upvotes given", min_value=0, value=None, step=1)
    with col3:
        downvote_count = st.number_input("Downvotes given", min_value=0, value=None, step=1)
    account_created = st.date_input("Account created on", value=None)

    submitted = st.form_submit_button("Predict", use_container_width=True)


def build_payload():
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]

    asker = None
    if reputation is not None or upvote_count is not None or downvote_count is not None or account_created is not None:
        asker = {
            "reputation": int(reputation) if reputation is not None else None,
            "upvote_count": int(upvote_count) if upvote_count is not None else None,
            "downvote_count": int(downvote_count) if downvote_count is not None else None,
            "account_created": (
                datetime.combine(account_created, datetime.min.time(), tzinfo=timezone.utc).isoformat()
                if account_created else None
            ),
        }

    return {
        "title": title,
        "body": body,
        "tags": tags,
        "asker": asker,
    }


def render_result(result: dict):
    proba = result["probability_answered"]
    will_answer = result["will_likely_be_answered"]

    st.divider()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Answer probability", f"{proba:.0%}")
    with col2:
        if will_answer:
            st.success(result["summary"])
        else:
            st.warning(result["summary"])

    st.progress(proba)

    st.subheader("⏱ Expected response time")
    time_info = result["expected_time_to_answer"]
    st.write(time_info["summary"])

    st.subheader("🔍 What's driving this prediction")
    for factor in result["top_factors"]:
        arrow = "🟢 increases" if factor["direction"] == "increases" else "🔴 decreases"
        st.markdown(f"- **{factor['plain_language'].capitalize()}** {arrow} the likelihood of an answer")

    with st.expander("Raw prediction data"):
        st.json(result)


if submitted:
    if not title.strip() or not body.strip():
        st.error("Please fill in both a title and a body before predicting.")
    else:
        payload = build_payload()
        try:
            with st.spinner("Contacting prediction service... (may take up to a minute on first request)"):
                response = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=90)
            if response.status_code == 200:
                render_result(response.json())
            else:
                st.error(f"API returned an error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the prediction service: {e}")

st.divider()
st.caption(
    "This tool provides an estimate based on historical Stack Overflow data "
    "(2020-2022) and is not a guarantee. See the project README for details "
    "on how predictions are generated and their known limitations."
)
