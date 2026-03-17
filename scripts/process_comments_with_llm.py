#!/usr/bin/env python3

import argparse
import importlib.util
import json
import logging
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd


GLOBAL_PORTKEY_CLIENT = None


def read_api_key(api_key_path: str) -> str:
    with open(api_key_path, "r") as file:
        return file.read().strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LLM labeling on comment parquet files with periodic progress logging."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/comment_exports/final_topic_comments.parquet",
        help="Input parquet file containing comments.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output parquet path. Defaults to <input_stem>_<prompt>_<model>.parquet.",
    )
    parser.add_argument(
        "--inp-col",
        type=str,
        default="content",
        help="Input column name containing comment text.",
    )
    parser.add_argument(
        "--out-col",
        type=str,
        default=None,
        help="Output column name for LLM responses. Defaults to <prompt>_<model>.",
    )
    parser.add_argument(
        "--prompt-name",
        type=str,
        required=True,
        help="Prompt name key from configs/labeling_prompts.json.",
    )
    parser.add_argument(
        "--llm-name",
        type=str,
        default="gpt-4o-mini",
        help="LLM name.",
    )
    parser.add_argument(
        "--api-key-path",
        type=str,
        default="portkey_api_key.txt",
        help="Path to API key file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of worker processes (upper bound).",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=10000,
        help="Chunk size and reporting interval in number of processed comments.",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=500,
        help="Number of requests sent per API sub-batch.",
    )
    parser.add_argument(
        "--rpm-limit",
        type=int,
        default=1500,
        help="Target requests-per-minute rate limit.",
    )
    parser.add_argument(
        "--min-batch-seconds",
        type=float,
        default=0.0,
        help="Minimum seconds between API sub-batch starts; 0 means sleep only if RPM limit requires it.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Run sequentially instead of multiprocessing.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Path to incremental checkpoint CSV. Defaults to <output-file>.checkpoint.csv.",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from checkpoint file if it exists.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Do not load any existing checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every-chunks",
        type=int,
        default=1,
        help="Append checkpoint rows every N processed chunks.",
    )
    parser.set_defaults(resume=True)

    args = parser.parse_args()
    args.api_key = read_api_key(args.api_key_path)
    return args


def load_config_and_prompt(args: argparse.Namespace) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "constants_labeling.py"

    spec = importlib.util.spec_from_file_location("constants_labeling", str(config_path))
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    prompt_path = repo_root / getattr(config, "PROMPT_PATH", "configs/labeling_prompts.json")
    with open(prompt_path, "r") as file:
        prompt_templates = json.load(file)

    if args.prompt_name not in prompt_templates:
        raise ValueError(f"Unknown prompt name: {args.prompt_name}")

    args.prompt_template = prompt_templates[args.prompt_name]
    if args.out_col is None:
        args.out_col = f"{args.prompt_name}_{args.llm_name}"

    if args.output_file is None:
        input_path = Path(args.input_file)
        args.output_file = str(input_path.with_name(f"{input_path.stem}_{args.out_col}.parquet"))

    return args


def _initialize_portkey_client(api_key: str):
    global GLOBAL_PORTKEY_CLIENT
    from portkey_ai import Portkey

    GLOBAL_PORTKEY_CLIENT = Portkey(api_key=api_key)


def _append_checkpoint_rows(checkpoint_file: str, row_indices, responses, out_col: str):
    checkpoint_df = pd.DataFrame({"row_index": row_indices, out_col: responses})
    checkpoint_path = Path(checkpoint_file)
    checkpoint_df.to_csv(
        checkpoint_file,
        mode="a",
        header=not checkpoint_path.exists(),
        index=False,
    )


def _load_checkpoint_rows(checkpoint_file: str, out_col: str):
    checkpoint_path = Path(checkpoint_file)
    if not checkpoint_path.exists():
        return None

    checkpoint_df = pd.read_csv(checkpoint_file)
    if "row_index" not in checkpoint_df.columns or out_col not in checkpoint_df.columns:
        return None

    checkpoint_df = checkpoint_df.dropna(subset=["row_index"])
    checkpoint_df["row_index"] = checkpoint_df["row_index"].astype(int)
    checkpoint_df = checkpoint_df.drop_duplicates(subset=["row_index"], keep="last")
    return checkpoint_df


def _worker(worker_args):
    sentence, prompt_template, model_name, api_key, prompt_name = worker_args
    try:
        global GLOBAL_PORTKEY_CLIENT
        if GLOBAL_PORTKEY_CLIENT is None:
            from portkey_ai import Portkey

            GLOBAL_PORTKEY_CLIENT = Portkey(api_key=api_key)

        client = GLOBAL_PORTKEY_CLIENT
        prompt = [
            {"role": "system", "content": "You are helping in scientific analysis, so please be precise."},
            {"role": "user", "content": prompt_template.format(sentence=sentence)},
        ]

        if "eval" in prompt_name and "Mistral" in model_name:
            max_tokens = 5
        elif "eval" in prompt_name:
            max_tokens = 1
        else:
            max_tokens = 1000

        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=max_tokens,
            top_p=1.0,
            messages=prompt,
        )
        return response.choices[0].message.content
    except Exception:
        return ""


def call_llm_on_sentences(sentences, args: argparse.Namespace, batch_size: int = 32):
    worker_input = [
        (
            sentence,
            args.prompt_template,
            args.llm_name,
            args.api_key,
            args.prompt_name,
        )
        for sentence in sentences
    ]

    api_batch_size = max(1, int(args.api_batch_size))
    rpm_limit = max(1, int(args.rpm_limit))
    min_batch_seconds = max(0.0, float(args.min_batch_seconds))

    def _wait_if_needed(current_batch_size: int, batch_duration_seconds: float):
        rpm_interval_seconds = (current_batch_size / rpm_limit) * 60.0
        target_interval_seconds = max(min_batch_seconds, rpm_interval_seconds)
        wait_seconds = max(0.0, target_interval_seconds - batch_duration_seconds)
        if wait_seconds > 0:
            logging.info(
                "Rate limit pacing: batch=%d, rpm_limit=%d, target_interval=%.2fs, "
                "batch_duration=%.2fs, sleeping=%.2fs",
                current_batch_size,
                rpm_limit,
                target_interval_seconds,
                batch_duration_seconds,
                wait_seconds,
            )
            time.sleep(wait_seconds)

    if args.debug_mode:
        _initialize_portkey_client(args.api_key)
        results = []
        for batch_start in range(0, len(worker_input), api_batch_size):
            batch_end = min(batch_start + api_batch_size, len(worker_input))
            batch_items = worker_input[batch_start:batch_end]
            batch_t0 = time.time()
            for item in batch_items:
                results.append(_worker(item))
            batch_duration = time.time() - batch_t0
            _wait_if_needed(len(batch_items), batch_duration)
        return results

    processes = min(cpu_count(), batch_size)
    results = []
    with Pool(processes=processes, initializer=_initialize_portkey_client, initargs=(args.api_key,)) as pool:
        for batch_start in range(0, len(worker_input), api_batch_size):
            batch_end = min(batch_start + api_batch_size, len(worker_input))
            batch_items = worker_input[batch_start:batch_end]
            batch_t0 = time.time()
            batch_results = pool.map(_worker, batch_items)
            results.extend(batch_results)
            batch_duration = time.time() - batch_t0
            _wait_if_needed(len(batch_items), batch_duration)

    return results


def process_comments_with_llm(comments_df: pd.DataFrame, args: argparse.Namespace, batch_size: int = 32):
    sentences = comments_df[args.inp_col].tolist()
    responses = call_llm_on_sentences(sentences, args, batch_size=batch_size)
    comments_df[args.out_col] = responses
    return comments_df


def run(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    start_time = time.time()
    logging.info("Loading input parquet: %s", args.input_file)
    comments_df = pd.read_parquet(args.input_file)

    if args.inp_col not in comments_df.columns:
        candidate_columns = ["content", "Comments", "comment", "text", "body"]
        detected_col = next((col for col in candidate_columns if col in comments_df.columns), None)
        if detected_col is not None:
            logging.warning(
                "Input column '%s' not found. Auto-detected '%s' from available columns.",
                args.inp_col,
                detected_col,
            )
            args.inp_col = detected_col
        else:
            available = ", ".join(map(str, comments_df.columns.tolist()))
            raise ValueError(
                f"Input column '{args.inp_col}' not found in input parquet. "
                f"Available columns: {available}"
            )

    valid_mask = comments_df[args.inp_col].notna()
    total_valid = int(valid_mask.sum())

    if args.out_col not in comments_df.columns:
        comments_df[args.out_col] = ""

    checkpoint_file = args.checkpoint_file or f"{args.output_file}.checkpoint.csv"
    if args.resume:
        checkpoint_df = _load_checkpoint_rows(checkpoint_file, args.out_col)
        if checkpoint_df is not None and len(checkpoint_df) > 0:
            valid_checkpoint_df = checkpoint_df[
                checkpoint_df["row_index"].isin(comments_df.index)
            ]
            comments_df.loc[valid_checkpoint_df["row_index"], args.out_col] = valid_checkpoint_df[args.out_col].values
            logging.info(
                "Loaded %d checkpointed responses from %s",
                len(valid_checkpoint_df),
                checkpoint_file,
            )

    logging.info("Rows loaded: %d", len(comments_df))
    logging.info("Rows with non-null '%s': %d", args.inp_col, total_valid)
    logging.info("Model: %s | Prompt: %s | Output column: %s", args.llm_name, args.prompt_name, args.out_col)

    if total_valid == 0:
        comments_df[args.out_col] = ""
        comments_df.to_parquet(args.output_file, index=False)
        logging.info("No valid comments to process. Wrote output to %s", args.output_file)
        return

    filled_mask = comments_df[args.out_col].fillna("").astype(str).str.strip().ne("")
    already_processed_mask = valid_mask & filled_mask
    already_processed = int(already_processed_mask.sum())
    remaining_indices = comments_df.index[valid_mask & ~filled_mask].tolist()

    logging.info("Already processed from checkpoint/output: %d", already_processed)
    logging.info("Remaining comments to process: %d", len(remaining_indices))

    if len(remaining_indices) == 0:
        comments_df.to_parquet(args.output_file, index=False)
        logging.info("All valid comments already processed. Saved output parquet: %s", args.output_file)
        return

    processed = already_processed
    report_every = max(1, args.report_every)
    checkpoint_every_chunks = max(1, args.checkpoint_every_chunks)
    chunks_since_checkpoint = 0

    for start in range(0, len(remaining_indices), report_every):
        end = min(start + report_every, len(remaining_indices))
        chunk_indices = remaining_indices[start:end]
        chunk_df = comments_df.loc[chunk_indices].copy()
        chunk_result = process_comments_with_llm(chunk_df, args, batch_size=args.batch_size)
        chunk_responses = chunk_result[args.out_col].tolist()
        comments_df.loc[chunk_indices, args.out_col] = chunk_responses
        processed = already_processed + end
        chunks_since_checkpoint += 1

        if chunks_since_checkpoint >= checkpoint_every_chunks:
            _append_checkpoint_rows(checkpoint_file, chunk_indices, chunk_responses, args.out_col)
            logging.info("Checkpoint appended: %s (%d rows)", checkpoint_file, len(chunk_indices))
            chunks_since_checkpoint = 0

        chunk_non_empty = sum(1 for value in chunk_responses if str(value).strip() != "")
        chunk_size = len(chunk_responses)
        chunk_empty = chunk_size - chunk_non_empty
        if chunk_size > 0:
            logging.info(
                "Chunk quality: non-empty=%d/%d (%.2f%%), empty=%d",
                chunk_non_empty,
                chunk_size,
                100.0 * chunk_non_empty / chunk_size,
                chunk_empty,
            )

        if chunk_size > 0 and chunk_non_empty == 0:
            raise RuntimeError(
                "All responses in the latest chunk are empty. "
                "Stopping early to avoid writing an all-empty output."
            )

        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0.0
        logging.info("Processed %d/%d comments (%.2f%%) | %.2f comments/s", processed, total_valid, 100.0 * processed / total_valid, rate)

    if chunks_since_checkpoint > 0:
        last_chunk_start = max(0, len(remaining_indices) - chunks_since_checkpoint * report_every)
        pending_indices = remaining_indices[last_chunk_start:]
        pending_responses = comments_df.loc[pending_indices, args.out_col].tolist()
        _append_checkpoint_rows(checkpoint_file, pending_indices, pending_responses, args.out_col)
        logging.info("Final checkpoint append: %s (%d rows)", checkpoint_file, len(pending_indices))

    comments_df.to_parquet(args.output_file, index=False)

    total_elapsed = time.time() - start_time
    logging.info("Completed processing in %.2f seconds.", total_elapsed)
    logging.info("Saved output parquet: %s", args.output_file)


if __name__ == "__main__":
    cli_args = parse_arguments()
    cli_args = load_config_and_prompt(cli_args)
    run(cli_args)
