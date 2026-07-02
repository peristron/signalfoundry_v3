import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from typing import Dict, Iterable, List, Optional

import pandas as pd

import text_processor as tp


SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".xlsm")
DEFAULT_MAX_TOPIC_DOCS = 50000
DEFAULT_MAX_EVIDENCE_DOCS = 10000
DEFAULT_EVIDENCE_CHARS = 700


def setup_args():
    parser = argparse.ArgumentParser(
        description=(
            "Harvester: large-scale offline sketch generator for Signal Foundry. "
            "Produces counts, bigrams, optional temporal/category counts, sampled "
            "topic docs, and bounded evidence snippets for the Insight Engine."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV/XLSX file or folder")
    parser.add_argument("--col", required=True, help="Name of the text column to analyze")
    parser.add_argument("--output", default="sketch.json", help="Output path for the sketch file")
    parser.add_argument("--chunksize", type=int, default=50000, help="Rows per CSV chunk")
    parser.add_argument("--date-col", default=None, help="Optional date column for temporal analysis")
    parser.add_argument("--category-col", default=None, help="Optional category/group column for comparison")
    parser.add_argument("--encoding", default="utf-8", help="CSV encoding, for example utf-8 or latin-1")
    parser.add_argument("--min-word-len", type=int, default=2, help="Minimum token length")
    parser.add_argument("--keep-hyphens", action="store_true", help="Preserve hyphens inside tokens")
    parser.add_argument("--keep-apostrophes", action="store_true", help="Preserve apostrophes inside tokens")
    parser.add_argument("--no-bigrams", action="store_true", help="Disable bigram calculation")
    parser.add_argument("--no-evidence", action="store_true", help="Do not store representative excerpts")
    parser.add_argument("--max-evidence-docs", type=int, default=DEFAULT_MAX_EVIDENCE_DOCS)
    parser.add_argument("--evidence-chars", type=int, default=DEFAULT_EVIDENCE_CHARS)
    parser.add_argument("--max-topic-docs", type=int, default=DEFAULT_MAX_TOPIC_DOCS)
    parser.add_argument("--doc-batch-size", type=int, default=5, help="Rows grouped into one sampled topic doc")
    parser.add_argument(
        "--stopwords",
        default="",
        help="Optional comma-separated stopwords or phrases to remove before analysis",
    )
    parser.add_argument(
        "--no-generic-stopwords",
        action="store_true",
        help="Keep generic prepositions/filler words instead of removing them",
    )
    return parser.parse_args()


def resolve_column(header: List[str], requested: Optional[str]) -> Optional[str]:
    if not requested:
        return None
    if requested in header:
        return requested
    lowered = {str(name).lower(): name for name in header}
    return lowered.get(str(requested).lower())


def stream_csv_rows(
    file_path: str,
    chunksize: int,
    text_col: str,
    date_col: Optional[str],
    category_col: Optional[str],
    encoding: str,
) -> Iterable[Dict[str, Optional[str]]]:
    preview = pd.read_csv(file_path, nrows=0, encoding=encoding, on_bad_lines="skip", engine="python")
    header = list(preview.columns)
    resolved_text = resolve_column(header, text_col)
    resolved_date = resolve_column(header, date_col)
    resolved_category = resolve_column(header, category_col)
    if not resolved_text:
        raise ValueError(f"Text column '{text_col}' not found in {file_path}")

    usecols = [resolved_text]
    for optional_col in [resolved_date, resolved_category]:
        if optional_col and optional_col not in usecols:
            usecols.append(optional_col)

    for chunk in pd.read_csv(
        file_path,
        usecols=usecols,
        chunksize=chunksize,
        on_bad_lines="skip",
        encoding=encoding,
        engine="python",
        dtype=str,
    ):
        for _, row in chunk.iterrows():
            text = row.get(resolved_text)
            if text is None:
                continue
            text = str(text).strip()
            if not text or text.lower() in {"nan", "none", "null"}:
                continue
            yield {
                "text": text,
                "date": row.get(resolved_date) if resolved_date else None,
                "category": row.get(resolved_category) if resolved_category else None,
            }


def stream_excel_rows(
    file_path: str,
    chunksize: int,
    text_col: str,
    date_col: Optional[str],
    category_col: Optional[str],
) -> Iterable[Dict[str, Optional[str]]]:
    if tp.openpyxl is None:
        raise ImportError("openpyxl is required for Excel harvesting")

    wb = tp.openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    first = next(rows_iter, None)
    if first is None:
        wb.close()
        return

    header = tp.make_unique_header(list(first))
    resolved_text = resolve_column(header, text_col)
    resolved_date = resolve_column(header, date_col)
    resolved_category = resolve_column(header, category_col)
    if not resolved_text:
        wb.close()
        raise ValueError(f"Text column '{text_col}' not found in {file_path}")

    name_to_idx = {name: idx for idx, name in enumerate(header)}
    text_idx = name_to_idx[resolved_text]
    date_idx = name_to_idx.get(resolved_date) if resolved_date else None
    category_idx = name_to_idx.get(resolved_category) if resolved_category else None

    yielded = 0
    for row in rows_iter:
        text = row[text_idx] if text_idx < len(row) else None
        if text is None:
            continue
        text = str(text).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        yielded += 1
        yield {
            "text": text,
            "date": row[date_idx] if (date_idx is not None and date_idx < len(row)) else None,
            "category": row[category_idx] if (category_idx is not None and category_idx < len(row)) else None,
        }
        if yielded % chunksize == 0:
            print(f"   Excel stream checkpoint: {yielded:,} rows")

    wb.close()


def stream_file(
    file_path: str,
    chunksize: int,
    text_col: str,
    date_col: Optional[str],
    category_col: Optional[str],
    encoding: str,
) -> Iterable[Dict[str, Optional[str]]]:
    ext = os.path.splitext(file_path.lower())[1]
    if ext == ".csv":
        yield from stream_csv_rows(file_path, chunksize, text_col, date_col, category_col, encoding)
    elif ext in {".xlsx", ".xlsm"}:
        yield from stream_excel_rows(file_path, chunksize, text_col, date_col, category_col)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")


def collect_input_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(f"Input path not found: {input_path}")
    files = [
        os.path.join(input_path, filename)
        for filename in sorted(os.listdir(input_path))
        if filename.lower().endswith(SUPPORTED_EXTENSIONS)
    ]
    if not files:
        raise FileNotFoundError(f"No supported files found in folder: {input_path}")
    return files


def main():
    args = setup_args()

    phrases, single_stopwords = tp.parse_user_stopwords(args.stopwords)
    phrase_pattern = tp.build_phrase_pattern(phrases)
    stopwords = tp.build_default_stopwords(remove_prepositions=not args.no_generic_stopwords)
    stopwords.update(single_stopwords)
    trans_map = tp.build_punct_translation(
        keep_hyphens=args.keep_hyphens,
        keep_apostrophes=args.keep_apostrophes,
    )

    word_counter = Counter()
    bigram_counter = Counter()
    doc_freqs = Counter()
    entity_counts = Counter()
    temporal_counts = defaultdict(Counter)
    category_counts = defaultdict(Counter)
    topic_docs: List[Counter] = []
    evidence_docs = []
    evidence_limit_reached = False
    total_rows = 0
    total_kept_rows = 0

    batch_accum = Counter()
    batch_rows = 0

    files = collect_input_files(args.input)
    print("--- HARVESTER STARTED ---")
    print(f"Input: {args.input}")
    print(f"Files: {len(files)}")
    print(f"Text column: {args.col}")
    if args.date_col:
        print(f"Date column: {args.date_col}")
    if args.category_col:
        print(f"Category column: {args.category_col}")

    for file_idx, file_path in enumerate(files, start=1):
        print(f">>> Processing file {file_idx}/{len(files)}: {file_path}")
        file_rows = 0
        try:
            row_iter = stream_file(
                file_path,
                args.chunksize,
                args.col,
                args.date_col,
                args.category_col,
                args.encoding,
            )
            for row in row_iter:
                total_rows += 1
                file_rows += 1
                raw_text = row["text"]
                tokens = tp.clean_and_tokenize(
                    raw_text,
                    remove_chat=True,
                    remove_html=True,
                    unescape=True,
                    remove_urls=True,
                    trans_map=trans_map,
                    stopwords=stopwords,
                    phrase_pattern=phrase_pattern,
                    min_len=args.min_word_len,
                    drop_int=True,
                )
                if not tokens:
                    continue

                total_kept_rows += 1
                line_counts = Counter(tokens)
                word_counter.update(tokens)
                doc_freqs.update(line_counts.keys())
                entity_counts.update(tp.extract_entities_regex(raw_text, stopwords))

                if not args.no_bigrams and len(tokens) > 1:
                    bigram_counter.update(pairwise(tokens))

                date_key = tp.clean_date_str(row.get("date"))
                category_key = tp.clean_category(row.get("category"))
                if date_key:
                    temporal_counts[date_key].update(tokens)
                if category_key:
                    category_counts[category_key].update(tokens)

                if not args.no_evidence and len(evidence_docs) < args.max_evidence_docs:
                    evidence = tp.build_evidence_doc(
                        len(evidence_docs) + 1,
                        raw_text,
                        tokens,
                        date_key,
                        category_key,
                        max_chars=args.evidence_chars,
                    )
                    if evidence:
                        evidence_docs.append(evidence)
                elif not args.no_evidence:
                    evidence_limit_reached = True

                if len(topic_docs) < args.max_topic_docs:
                    batch_accum.update(line_counts)
                    batch_rows += 1
                    if batch_rows >= max(1, args.doc_batch_size):
                        topic_docs.append(batch_accum)
                        batch_accum = Counter()
                        batch_rows = 0

                if total_rows % args.chunksize == 0:
                    print(
                        f"   Processed {total_rows:,} rows "
                        f"({total_kept_rows:,} with usable tokens)"
                    )
        except Exception as exc:
            print(f"WARNING: skipped {file_path}: {exc}", file=sys.stderr)
            continue

        print(f"   File rows scanned: {file_rows:,}")

    if batch_accum and len(topic_docs) < args.max_topic_docs:
        topic_docs.append(batch_accum)

    serializable_bigrams = {f"{pair[0]}|{pair[1]}": count for pair, count in bigram_counter.items()}
    serializable_topic_docs = [dict(counter) for counter in topic_docs]
    serializable_temporal = {key: dict(counter) for key, counter in temporal_counts.items()}
    serializable_category = {key: dict(counter) for key, counter in category_counts.items()}

    dashboard_summary = tp.summarize_dashboard(
        word_counter,
        bigram_counter,
        category_counts,
        temporal_counts,
        entity_counts,
        total_rows,
    )

    sketch_data = {
        "schema_version": "2.0",
        "total_rows": total_rows,
        "usable_rows": total_kept_rows,
        "counts": dict(word_counter),
        "bigrams": serializable_bigrams,
        "topic_docs": serializable_topic_docs,
        "limit_reached": len(topic_docs) >= args.max_topic_docs,
        "temporal_counts": serializable_temporal,
        "category_counts": serializable_category,
        "entity_counts": dict(entity_counts),
        "doc_freqs": dict(doc_freqs),
        "evidence_docs": evidence_docs,
        "evidence_limit_reached": evidence_limit_reached,
        "dashboard_summary": dashboard_summary,
        "metadata": {
            "source": args.input,
            "files": files,
            "text_col": args.col,
            "date_col": args.date_col,
            "category_col": args.category_col,
            "generated_by": "Harvester Insight v3.0",
            "privacy_note": (
                "This sketch includes bounded representative excerpts unless "
                "--no-evidence was used."
            ),
        },
    }

    out_path = args.output
    if not out_path.lower().endswith(".json"):
        out_path += ".json"

    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            json.dump(sketch_data, output_file, ensure_ascii=False)
        print("Sketch saved successfully.")
        print(f"Output: {out_path}")
        print(f"Rows scanned: {total_rows:,}")
        print(f"Usable rows: {total_kept_rows:,}")
        print(f"Unique words: {len(word_counter):,}")
        print(f"Bigrams: {len(bigram_counter):,}")
        print(f"Topic docs sampled: {len(topic_docs):,}")
        print(f"Evidence snippets: {len(evidence_docs):,}")
        print("Upload this JSON file to the Offline Analysis section in the main app.")
    except Exception as exc:
        print(f"ERROR: could not save sketch: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
