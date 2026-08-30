# Stack Overflow Answer Predictor

This project predicts whether a draft Stack Overflow question is likely to be answered, and roughly how long that might take. It's built on a genuinely relational SQL data source (Google BigQuery's public Stack Overflow dataset), a full exploratory and statistical analysis, two trained LightGBM models with SHAP explainability, and a deployed FastAPI backend with a Streamlit frontend.

## Live demo

- Try it live (Streamlit): https://stackoverflow-answer-predictor.streamlit.app
- API docs (Swagger): https://stackoverflow-answer-predictor.onrender.com/docs
- Kaggle notebooks:
  - EDA and statistical testing notebook: [analysis_notebook](https://www.kaggle.com/code/shishiradh/stack-overflow-analysis)
  - Classification and Regression model training notebook: [prediction_notebook](https://www.kaggle.com/code/shishiradh/stack-overflow-prediction)

The API is hosted on Render's free tier, which spins down after a period of inactivity. The first request after some idle time can take 30 to 60 seconds while it wakes back up, so don't worry if that first prediction feels slow.

## What this project does

A large share of developer problem solving happens on Q&A platforms like Stack Overflow, but a meaningful fraction of questions never get answered at all. This project looks at two questions before a question is even posted:

1. Will this question get answered?
2. If so, roughly how long might that take?

The same idea applies to plenty of other settings beyond Stack Overflow itself: an internal support desk, a product forum, a company wiki. "Will this ticket get resolved, and how fast" is a genuinely useful question wherever people are asking other people for help.

The full analysis, methodology, and findings are written up in [FINDINGS.md](FINDINGS.md).

## Headline results

| Model | Metric | Result |
|---|---|---|
| Classification (will it be answered?) | ROC-AUC | 0.711 |
| Classification | Macro-F1 (at tuned threshold 0.56) | 0.645 |
| Regression (time to answer, quantile) | 80% interval coverage | 81.4%, well calibrated |
| Regression | Median absolute error | 2.5 hours |

The reasoning behind these numbers, including the statistical testing and feature importance work that led to them, is in [FINDINGS.md](FINDINGS.md).

## How it's put together

```
BigQuery (bigquery-public-data.stackoverflow)
        |  SQL: CTEs, joins, walk-forward target encoding (sql/)
        v
Kaggle notebooks: EDA, statistical testing, feature engineering,
                   model training (notebooks/, mirrored on Kaggle)
        |
        v
Trained artifacts: LightGBM models (.txt) plus JSON metadata (models/)
        |
        v
FastAPI service (api/), Dockerized, deployed on Render
        |
        v
Streamlit frontend (streamlit_app/), deployed on Streamlit Cloud
```

## Tech stack

| Layer | Tools |
|---|---|
| Data source | Google BigQuery (public Stack Overflow dataset) |
| Analysis and SQL | google-cloud-bigquery, pandas, CTEs and window-function-style queries |
| Statistics | scipy, statsmodels (chi-square, Cochran-Armitage, Kruskal-Wallis, logistic regression) |
| Modeling | LightGBM (binary classification and quantile regression) |
| Explainability | SHAP |
| Backend | FastAPI, Pydantic, Docker |
| Frontend | Streamlit |
| Deployment | Render for the API, Streamlit Community Cloud for the frontend |

## Repository structure

```
stackoverflow-answer-predictor/
├── notebooks/          EDA and model training notebooks (also on Kaggle, see links above)
├── src/                 Reusable scripts (query runner, dry-run cost checker)
├── sql/                 Versioned BigQuery queries
├── docs                 Graphs and Plots fo EDA
├── models/              Trained model files and JSON metadata (committed, see below)
├── api/                 FastAPI prediction service, see api/README.md
├── streamlit_app/       Frontend, see streamlit_app/README.md
├── FINDINGS.md           Full EDA and statistical testing writeup
├── requirements.txt      Training and analysis environment
└── README.md             This file
```

## Reproducing this project

### 1. Data and training environment

```bash
git clone https://github.com/yourusername/stackoverflow-answer-predictor.git
cd stackoverflow-answer-predictor
python -m venv soflow-env
soflow-env\Scripts\activate.bat        # Windows
pip install -r requirements.txt
```

This requires a Google Cloud project with the BigQuery API enabled (`gcloud auth application-default login`). The public dataset itself is free to query within the 1TB per month free tier. See `sql/` for the exact queries and `notebooks/` (or the Kaggle links above) for the full pipeline.

### 2. Trained models

Trained model files and metadata are already committed in `models/`, so there's no need to retrain anything just to run the API. If you do want to retrain from scratch, the notebooks walk through the full pipeline: feature engineering, walk-forward tag encoding, LightGBM training, threshold tuning, SHAP, and export.

### 3. Run the API

See [api/README.md](api/README.md) for local and Docker instructions.

### 4. Run the frontend

See [streamlit_app/README.md](streamlit_app/README.md) for local instructions.

## Known limitations

- The dataset is a static BigQuery snapshot covering 2008 to 2022. The final roughly 90 days of that window were excluded from training and evaluation because of right-censoring, explained in FINDINGS.md.
- There's a genuine, externally corroborated decline in Stack Overflow activity that began around September 2021. The model reflects that later, lower-answer-rate period rather than treating it as noise to smooth over. This is documented in detail in FINDINGS.md.
- Regression predictions are presented as calibrated ranges rather than single point estimates, given the model's modest point-estimate R-squared of around 0.09. The reasoning for this choice is in FINDINGS.md.
- Deployment-specific limitations, such as the static rolling-context snapshot and the heuristics used to detect code blocks in live input, are documented in [api/README.md](api/README.md).

## License

MIT
