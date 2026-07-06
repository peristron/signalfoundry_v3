# Signal Foundry

Signal Foundry is a Streamlit-based text analysis and computational sensemaking app. It turns unstructured or semi-structured text into interpretable signals: recurring terms, distinctive phrases, hidden topics, evidence cards, entity patterns, relationship graphs, maturity indicators, and optional AI-assisted synthesis.

The app is designed to help a human analyst quickly understand the shape of a resource or corpus. It does not replace expert judgment. Instead, it provides a structured first pass: what appears often, what stands out, what clusters together, what may be missing, what is qualified or contested, and what deserves closer human review.

## What Signal Foundry Does

Signal Foundry can process text from sources such as:

- PDF files
- TXT files
- CSV and Excel files
- PowerPoint files
- VTT or transcript-style files
- pasted text
- URLs
- offline harvester sketches

After scanning, it produces several layers of analysis:

- an Executive Signal Dashboard
- Signal Compass directional read
- Resource Shape synthesis
- Supporting Insight Cards
- Word Cloud and corpus statistics
- theme evidence cards
- frequency vs. distinctiveness quadrant
- keyphrase extraction
- named-entity-style extraction
- topic modeling
- network graph analysis
- temporal and category comparisons when metadata exists
- maturity scoring when a matching maturity lens is selected
- optional AI Analyst output based on a statistical context brief
- calibration export packages for repeatable testing

## High-Level Purpose

Signal Foundry is meant for exploratory analysis of text-heavy materials where the user wants to ask:

- What is this resource mostly about?
- What language is repeated?
- What language is distinctive rather than merely frequent?
- What ideas travel together?
- What themes, tensions, risks, or constraints are emerging?
- What concepts are missing or weak?
- Which excerpts should a human inspect first?
- What is the overall shape of the resource?

It is especially useful for reports, transcripts, research papers, policy documents, client notes, strategy documents, feedback collections, and other messy text sources.

## Analytical Foundations

Signal Foundry uses a layered analytical pipeline. The goal is to combine familiar natural language processing methods with practical interpretive heuristics.

### 1. Text Cleaning and Normalization

The app first converts uploaded material into analyzable text. Depending on settings, it can remove chat artifacts, strip URLs or HTML, drop numbers, preserve or remove hyphens, apply stopwords, and optionally lemmatize terms.

This step matters because every later output depends on the quality of the cleaned text. If boilerplate, headers, footers, or transcript artifacts dominate the scan, the downstream outputs will also be distorted.

### 2. Tokenization and Frequency Analysis

The app breaks text into tokens, counts word frequencies, and tracks document or row-level units. This provides the basic statistical foundation for the word cloud, top terms, lexical diversity, and several downstream analyses.

Frequency answers a simple but important question:

> What appears most often?

Frequency alone is not enough, but it is a useful baseline.

### 3. Phrase Significance with NPMI

Signal Foundry uses Normalized Pointwise Mutual Information, or NPMI, to find word pairs that appear together more strongly than chance.

This helps distinguish meaningful phrases from ordinary adjacent words. For example, a phrase that appears together repeatedly and unusually may be more analytically useful than two words that are merely common on their own.

NPMI helps answer:

> Which phrases are unusually sticky?

### 4. Keyphrase Extraction with TF-IDF

Signal Foundry uses TF-IDF, or Term Frequency - Inverse Document Frequency, to identify terms that are distinctive to the uploaded corpus.

TF-IDF does not simply reward common words. It rewards words that are important in this corpus relative to how broadly they appear across document chunks.

TF-IDF helps answer:

> What makes this resource specific?

### 5. Topic Modeling with LDA and NMF

For larger or chunked corpora, Signal Foundry can use topic modeling methods such as:

- Latent Dirichlet Allocation (LDA)
- Non-Negative Matrix Factorization (NMF)

These methods attempt to identify hidden topic structures in a collection of documents or text chunks.

LDA treats topics as probability distributions over words. It is useful for discovering broad latent themes in larger mixed corpora.

NMF factorizes a document-term matrix into additive components. It often works well for cleaner corpora where topics are more compact and interpretable.

Topic modeling helps answer:

> What hidden topic groups may be present beneath the surface?

### 6. Signal Taxonomy

Signal Foundry classifies candidate signals into analytical categories such as:

- Evidence / Experiment
- Risk / Concern
- Risk / Failure Mode
- Blocker / Constraint
- Need / Request
- Decision / Tradeoff
- Infrastructure / System Dependence
- Institutional Structure / Social Design
- Authority / Legitimacy
- Contradiction / Tension
- Motif / Image Pattern
- Source / Boilerplate
- Low-Specificity Signal
- Absence / Weak Signal

These categories are heuristic. They help organize interpretation, but they are not final truth labels.

### 7. Interpretive Lift and Insight Cards

The app ranks candidate insight cards using a directional score called Interpretive Lift.

Interpretive Lift considers factors such as:

- evidence strength
- phrase distinctiveness
- confidence
- semantic fit
- phrase quality
- signal role
- whether the signal appears to be direct, qualified, or contrastive

The purpose is to bring the most useful analytical leads toward the top, while still preserving supporting or diagnostic signals for review.

### 8. Qualification and Contrast Detection

Signal Foundry includes a calibration layer that tries to distinguish direct claims from qualified or contrastive claims.

For example, a document may mention a concept in order to reject it, limit it, compare against it, or clarify that it is not the main point.

The app looks for contextual cues such as:

- not
- does not
- rather than
- instead
- only as comparison
- upper bound
- not intended for
- should not be confused with

This helps prevent the app from over-promoting phrases that appear in the text but are not actually central claims.

### 9. Signal Compass and Resource Shape

The Signal Compass gives a directional read of the resource. It summarizes which analytical forces are pulling the text most strongly.

Resource Shape then turns top-ranked signals into a short synthesis of what the corpus appears to be about.

These views are designed to be read first. They are not final conclusions. They are a map for human review.

### 10. Network Graph Analysis

Signal Foundry can build a network graph from co-occurring terms. Nodes represent terms. Edges represent relationships between terms that appear together.

The graph can help reveal:

- central concepts
- clusters of related terms
- dense or disconnected topics
- possible conceptual structure

Graph rendering is capped for browser stability on Streamlit Community Cloud. Large or dense corpora may be better reviewed through exported graph files.

### 11. Maturity Scoring

When selected, maturity models compare the text against domain vocabularies and staged capability language.

Maturity scoring is directional. It measures language present in the source material, not actual organizational reality.

It is most useful when the source material fits the selected maturity lens.

### 12. AI Analyst

The optional AI Analyst does not receive the full raw source document by default. Instead, it receives a privacy-conscious context brief built from:

- corpus statistics
- top terms and phrases
- entities
- Signal Compass
- Resource Shape
- insight cards
- signal roles
- graph/community summary when available
- maturity results when available

This design lets the AI help interpret the analytical sketch without requiring the full document text to be sent to the model.

The AI Analyst is best used after reviewing the visible dashboard and evidence cards.

## Privacy and Data Handling

Signal Foundry is designed for analysis of user-uploaded materials in a Streamlit app environment.

Important guidance:

- Do not upload sensitive or confidential material unless you are authorized to do so.
- For Streamlit Community Cloud deployments, anonymize source material where appropriate.
- The AI Analyst uses a summarized context brief rather than full raw documents by default.
- Full diagnostic exports may include evidence excerpts and should be shared carefully.
- Safe calibration exports omit representative evidence from the insight-card CSV.

## Calibration Export

The admin-only calibration export creates a ZIP package containing files such as:

- insight cards
- Resource Shape summary
- Resource Shape weighting diagnostics
- signal type distribution
- signal role distribution
- top terms
- top bigrams
- NPMI phrases
- TF-IDF keyphrases
- evidence snippets when full export is selected
- run summary metadata

This is useful for:

- comparing repeated runs
- tuning stopwords
- testing calibration changes
- documenting analysis snapshots
- auditing why a signal ranked where it did

## Recommended Workflow

1. Configure stopwords, cleaning, and scan settings in the sidebar.
2. Upload or paste source material in Workspace.
3. Scan the material.
4. Start with the Executive Signal Dashboard.
5. Read Signal Compass and Resource Shape.
6. Inspect Supporting Insight Cards.
7. Check Word Cloud and top terms for noise or boilerplate.
8. Review Themes, Keyphrases, Entities, and Graphs as needed.
9. Use Maturity only when the source fits the selected maturity lens.
10. Use the AI Analyst last, after the visible outputs look reasonable.
11. Use Calibration Export when testing or comparing runs.

## Interpreting Results Safely

Signal Foundry is best understood as a signal amplifier and evidence organizer.

It can help surface:

- repeated language
- distinctive phrases
- hidden topics
- relationships between terms
- possible tensions
- missing or weak concepts
- qualified or rejected claims
- candidate evidence for human review

It cannot guarantee:

- author intent
- factual truth
- causal explanation
- completeness
- legal, medical, scientific, or policy correctness

Treat the outputs as structured leads. Human interpretation remains essential.

## Deployment Notes

Signal Foundry is designed to run as a Streamlit app, including on Streamlit Community Cloud.

General deployment expectations:

- Python 3.10 or higher is recommended.
- Required packages should be listed in `requirements.txt`.
- API keys should be stored in Streamlit secrets, not hardcoded.
- Large files may run into memory or processing constraints on free hosted environments.
- Graph rendering is intentionally capped for browser stability.

## Suggested Short Description

Signal Foundry is a computational sensemaking tool for unstructured text. It combines NLP, statistical phrase scoring, TF-IDF, topic modeling, graph analysis, signal taxonomy, maturity scoring, and optional AI synthesis to help analysts surface the shape, themes, tensions, and evidence patterns within complex text resources.

