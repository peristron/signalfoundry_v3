# Signal Foundry

Signal Foundry is a Streamlit-based unstructured text analytics app for turning messy text sources into fast, inspectable signals.

It can analyze PDFs, CSVs, Excel files, transcripts, PowerPoints, pasted text, URLs, and offline harvester sketches. It surfaces recurring language, key phrases, entities, themes, relationships, maturity signals, and evidence-backed insight cards.

Signal Foundry is best understood as a **sensemaking tool**: it helps a human analyst see what deserves attention faster.

## What It Does

Signal Foundry helps answer questions such as:

- What is this corpus really about?
- What terms and phrases dominate?
- What ideas cluster together?
- What seems distinctive rather than merely frequent?
- What tensions, blockers, needs, risks, or opportunities appear?
- What concepts are missing or weak?
- Which people, systems, organizations, or entities keep appearing?
- How does language change across time or groups?
- What maturity signals appear in the text?

## Core Features

### Executive Signal Dashboard

After scanning data, the app shows a first-pass dashboard with:

- corpus size
- token volume
- evidence snippet count
- detected insight cards
- available groups and dates
- strongest signals
- signal-type mix
- suggested next analytical steps
- corpus fingerprint: top terms, phrases, and entities

This is the recommended starting point after every scan.

### Insight Engine

The Insight Engine creates evidence-backed cards with:

- signal name
- signal type
- evidence strength
- distinctiveness
- confidence
- representative evidence
- interpretation
- follow-up question

Signal types include:

- Pain / Friction
- Need / Request
- Blocker / Constraint
- Aspiration / Opportunity
- Risk / Concern
- Decision / Tradeoff
- Contradiction / Tension
- Absence / Weak Signal

### Word Cloud and Frequency Tables

Use these to confirm that the scan captured the right content and that boilerplate or noise is not dominating the analysis.

### Theme and Keyphrase Analysis

The app includes:

- theme evidence cards
- TF-IDF keyphrases
- sticky phrase detection using NPMI
- frequency vs. distinctiveness quadrant
- expected signal checks
- contrastive analysis across categories
- temporal drift when date data is available

### Network Graph

The graph shows relationships between frequently connected terms. It is useful for exploring clusters and conceptual neighborhoods.

For stability, graph rendering is intentionally bounded. Use:

- Min Link Frequency
- Max Nodes
- Max Links
- Physics on/off

For dense corpora, download the `.gexf` graph file and inspect it in external graph tools such as Gephi.

### Maturity Models

The app includes several maturity lenses, including:

- EdTech & LMS Ops
- General Business Ops
- Policy & Governance
- TAM 12-Domain Maturity Model

Maturity scoring is based on language signals. It should be treated as a structured conversation aid, not a final audit.

### AI Analyst

When configured, the AI Analyst can interpret the statistical sketch and insight-card context.

Privacy note: the AI Analyst is designed to use summary signals rather than reading full raw documents directly.

## Supported Inputs

The app supports:

- `.csv`
- `.xlsx`
- `.vtt`
- `.txt`
- `.json`
- `.pdf`
- `.pptx`
- pasted text
- public URLs
- offline harvester sketches

## Recommended Workflow

1. Upload one or more files.
2. Keep **Clear previous data** enabled unless intentionally combining scans.
3. Scan the content.
4. Start with the **Executive Signal Dashboard**.
5. Review the **Insight Engine** cards.
6. Check the Word Cloud and Frequency Tables for data quality.
7. Use Themes and Keyphrases for deeper interpretation.
8. Use the Network Graph only after the first-pass outputs make sense.
9. Use Maturity when the source material fits the selected maturity model.
10. Use AI Analyst last, once the visible evidence looks reasonable.

## Important Usage Notes

### Clear Previous Data

Leave **Clear previous data** enabled for most tests.

Turn it off only when you intentionally want to merge new material into the current corpus. Otherwise, repeated scans can duplicate data and distort results.

### Large Files

The app may allow large uploads, but Streamlit Community Cloud can still hit memory or runtime limits during parsing, graphing, topic modeling, or rendering.

For very large files, use `harvester.py` offline and upload the generated sketch.

### Offline Harvester

The offline harvester creates a `.json` sketch that can be loaded into the app.

Basic use:


python harvester.py --input data.csv --col text --output sketch.json

With date and category support:

python harvester.py \
  --input data.csv \
  --col text \
  --date-col date \
  --category-col team \
  --output sketch.json

For privacy-sensitive workflows, disable evidence excerpts:

python harvester.py --input data.csv --col text --output sketch.json --no-evidence
Streamlit Secrets

At minimum, configure:

auth_password = "your-password-here"

Optional AI provider keys:

deepseek_api_key = "your-key-here"
openai_api_key = "your-key-here"
xai_api_key = "your-key-here"

Do not commit real API keys to GitHub. Add real secrets in Streamlit Community Cloud settings.

Local Run
pip install -r requirements.txt
streamlit run mainapp_signalfoundry_v3.py

If your app file is named differently, use that filename instead.

Deployment

For Streamlit Community Cloud:

Push the repo to GitHub.
Confirm requirements.txt is at the repo root.
Set the app entry point to the correct Python file.
Add Streamlit secrets.
Deploy.
Test with a small file before using larger corpora.
Interpreting Results Safely

Signal Foundry surfaces evidence and patterns. It does not prove intent, causality, maturity, or truth by itself.

Use it to:

find leads
identify recurring signals
compare groups
detect missing concepts
generate better follow-up questions
support human interpretation

Do not treat outputs as final conclusions without reviewing representative evidence.
