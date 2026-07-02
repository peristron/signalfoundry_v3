# Harvester Insight Update

These files update the offline harvesting workflow for the Signal Foundry v3 Insight Engine.

## Basic Use

```bash
python harvester.py --input data.csv --col text --output sketch.json
```

## Better Insight Engine Support

Use date and category columns when available:

```bash
python harvester.py \
  --input data.csv \
  --col text \
  --date-col date \
  --category-col team \
  --output sketch.json
```

The generated sketch includes:

- word counts
- bigrams
- sampled topic documents
- temporal counts
- category counts
- lightweight entities
- bounded representative evidence snippets
- a small dashboard summary

## Privacy Option

Evidence snippets are useful for the Insight Engine, but they do include short excerpts from source text.

Disable excerpt storage with:

```bash
python harvester.py --input data.csv --col text --output sketch.json --no-evidence
```

## Large Files

For very large CSV files, tune chunk size:

```bash
python harvester.py --input data.csv --col text --chunksize 25000 --output sketch.json
```
