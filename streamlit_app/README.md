# Streamlit Frontend

The user-facing landing page. Calls the deployed FastAPI service's
`/predict` endpoint and renders the result.

## Run locally

```bash
cd streamlit_app
python -m venv venv
venv\Scripts\activate.bat        # Windows
pip install -r requirements.txt
streamlit run app.py
```

By default this points at the live Render deployment
(`https://stackoverflow-answer-predictor.onrender.com`). To point at your
local API instead (e.g. while testing changes before redeploying):

```bash
set API_BASE_URL=http://127.0.0.1:8000
streamlit run app.py
```

## Deploy (Streamlit Community Cloud)

1. Push this folder to your GitHub repo (already done if it's part of the
   main project repo).
2. Go to share.streamlit.io, sign in with GitHub.
3. New app -> select the repo, branch `main`, and set **Main file path** to
   `streamlit_app/app.py`.
4. Under **Advanced settings -> Secrets**, no secrets are required unless
   you want to override `API_BASE_URL` (e.g. if the Render URL changes) -
   in that case add:
   ```
   API_BASE_URL = "https://your-actual-url.onrender.com"
   ```
   and change the code to read `st.secrets.get("API_BASE_URL", ...)` instead
   of (or in addition to) `os.getenv(...)`.
5. Deploy. Streamlit Cloud gives you a public `*.streamlit.app` URL.

## Notes

- Render's free tier spins down after inactivity; the first prediction
  request after idle time can take 30-60 seconds while it wakes up. The
  spinner text in the app mentions this so it doesn't look broken.
- Asker fields are all optional - if none are filled in, the request is
  sent with `asker: null`, which the API treats the same way it treated
  deleted/anonymous accounts during training.
