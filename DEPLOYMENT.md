# Deployment Guide — Streamlit Community Cloud

This app is built to run on **Streamlit Community Cloud** with no local model hosting:
the LLM runs on the Groq Cloud API and embeddings run on CPU via a small
`all-MiniLM-L6-v2` model. The only secret you need is a free `GROQ_API_KEY`.

## Prerequisites

- A GitHub account with this repository pushed to it.
- A free Groq API key from <https://console.groq.com/keys>.
- A free Streamlit Community Cloud account: <https://share.streamlit.io>.

## Deployment steps

1. **Push the repo to GitHub** (branch `main`).
2. Go to <https://share.streamlit.io> and click **Create app → Deploy from GitHub**.
3. Select:
   - **Repository:** `satyamshivam13/PDF_RAG_Chatbot`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
4. Open **Advanced settings → Secrets** and paste:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
   (Optionally add `LLM_MODEL` / `LLM_BASE_URL` — see `.streamlit/secrets.toml.example`.)
5. Click **Deploy**. First boot installs `requirements.txt` and downloads the
   embedding model (~90 MB), so the initial cold start takes ~1–3 minutes.
6. Copy the public URL Streamlit assigns (e.g. `https://<app>.streamlit.app`).

## Post-deploy validation checklist

Run through this on the live URL before publishing the demo or adding a badge:

- [ ] App loads without a stack trace (missing-key shows a friendly warning, not a crash)
- [ ] Upload a `.pdf` — "File embedded and stored successfully!" appears
- [ ] Ask a question grounded in the uploaded file — answer streams token-by-token
- [ ] Embeddings build (first query after upload returns relevant context)
- [ ] Response-time indicator renders after the answer
- [ ] (FastAPI backend only) DuckDuckGo fallback `/search` returns results
- [ ] Re-uploading the same file shows "already processed" (hash de-dup works)

## Measurements to record (fill in after validation)

| Metric | Value |
|---|---|
| Cold start (first load) | _e.g. ~2 min_ |
| Warm inference latency (per query) | _e.g. ~1–2 s_ |
| Peak memory | _Streamlit Cloud free tier ≈ 1 GB limit_ |
| Live URL | _https://<app>.streamlit.app_ |

> Only add the "Live Demo" badge to the README **after** every checklist item passes
> on the live URL. Do not publish a badge for an unverified deployment.

## Notes & limits (free tier)

- The app sleeps after inactivity; the first request after sleeping re-runs the
  cold start. This is expected on the free tier.
- Groq's free tier has a daily token cap; heavy demo use can exhaust it until reset.
- `streamlit_app.py` is the Cloud entrypoint. The FastAPI backend (`rag_api.py`) and
  `index.html` are for local/self-hosted use and are not deployed by this flow.
- The ChromaDB store (`db/`) is ephemeral on Cloud — uploads persist only for the
  life of the running container.
