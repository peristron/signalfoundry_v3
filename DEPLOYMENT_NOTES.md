# Signal Foundry v2 Insight Iteration

## What Changed

- Added an **Insight Engine** tab that creates evidence-backed sensemaking cards.
- Captures bounded local excerpts during scanning so themes can point back to representative evidence.
- Adds signal categories: pain/friction, need/request, blocker/constraint, aspiration/opportunity, risk/concern, decision/tradeoff, contradiction/tension, and absence/weak signal.
- Adds confidence, interpretation, and follow-up questions for each insight card.
- Feeds structured insight-card context into the AI Analyst.
- Cleans leftover copy/paste boundary markers from the delivered app file.
- Fixes a small Excel row-index safety issue.

## Streamlit Secrets

The app expects this password secret:

```toml
auth_password = "your-password-here"
```

Optional AI provider keys:

```toml
deepseek_api_key = "your-key-here"
openai_api_key = "your-key-here"
xai_api_key = "your-key-here"
```

Use placeholders only in GitHub. Configure real values in Streamlit Community Cloud secrets.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run mainapp.py
```
