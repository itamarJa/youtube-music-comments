"""
Get comments from the parquet file by channel or video ID.
Sample videos per channel and/or comments per video.
Streams data in chunks to avoid loading entire file into memory.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import os
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

parser = argparse.ArgumentParser(description="Get comments from parquet file")
parser.add_argument('--channels_csv', type=str, help='Path to CSV file containing channel IDs')
parser.add_argument('--channel_col', type=str, default='channel_id', help='Column name for channel IDs in CSV')
parser.add_argument('--get_all', action='store_true', help='Get all comments (disable sampling and <100 filter)')
parser.add_argument('--output_file', type=str, help='Path to save the output CSV file (overrides default naming)')
args = parser.parse_args()

# Input file
comment_f = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/youtube_comments_music.parquet'

# Configuration - modify these as needed
get_all_comments = args.get_all

if args.channels_csv:
    channels_df = pd.read_csv(args.channels_csv)
    channels = channels_df[args.channel_col].dropna().astype(str).tolist()
else:
    channels = [
        "UCq-Fj5jknLsUf-MWSy4_brA",
        "UC5Z010wx1Yy_sQA6wNji93w",
        "UCEocdCKuIKLlcd_bwBnbetg",
        "UCf2G6jDbJI3s0d5U-N9fqzg",
        "UChEYVadfkMCfrKUi6qr3I1Q",
        "UCPKWE1H6xhxwPlqUlKgHb_w",
        "UC1SqP7_RfOC9Jf9L_GRHANg"
    ]  # List of channel IDs to filter by (None = all channels)

sample_by_channel = not get_all_comments  # Set to True to sample by channel, False to sample by video
videos = None  # List of video IDs to filter by (None = all videos)
num_videos_per_channel = 10  # Number of videos to sample per channel (if sample_by_channel=True)
num_comments_per_video = 100  # Number of comments to sample per video
chunk_size = 10000000  # Number of rows to read per chunk
use_parallel = True  # Enable parallel processing if available
max_workers = min(8, os.cpu_count() or 1)  # Number of parallel workers

print(f"Streaming comments from {comment_f}...", flush=True)

# Open parquet file for streaming
parquet_file = pq.ParquetFile(comment_f)
total_rows = parquet_file.metadata.num_rows
print(f"Total rows in file: {total_rows}", flush=True)

def filter_batch(batch_data):
    """Filter a single batch using PyArrow compute"""
    batch_idx, batch = batch_data
    rows_in_batch = batch.num_rows
    
    # Filter using PyArrow compute (much faster than pandas)
    if channels is not None:
        mask = pc.is_in(batch['channel_id'], value_set=pa.array(channels))
        batch = batch.filter(mask)
    
    if videos is not None:
        mask = pc.is_in(batch['video_id'], value_set=pa.array(videos))
        batch = batch.filter(mask)
    
    # Only convert to pandas if there's data left after filtering
    if batch.num_rows > 0:
        return batch_idx, batch.to_pandas(), rows_in_batch, batch.num_rows
    return batch_idx, None, rows_in_batch, 0

# Stream and collect only filtered data
if use_parallel and max_workers > 1:
    print(f"Streaming and filtering data (parallel with {max_workers} workers)...", flush=True)
else:
    print("Streaming and filtering data (sequential)...", flush=True)

filtered_chunks = {}
rows_processed = 0
rows_kept = 0
lock = threading.Lock()

if use_parallel and max_workers > 1:
    try:
        # Parallel processing with ThreadPoolExecutor (PyArrow releases GIL)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit batches as they're read
            batch_iter = enumerate(parquet_file.iter_batches(batch_size=chunk_size))
            futures = {executor.submit(filter_batch, (idx, batch)): idx for idx, batch in batch_iter}
            
            for future in as_completed(futures):
                batch_idx, df_chunk, rows_in_batch, kept = future.result()
                
                with lock:
                    rows_processed += rows_in_batch
                    if df_chunk is not None:
                        filtered_chunks[batch_idx] = df_chunk
                        rows_kept += kept
                    
                    if rows_processed % 100000000 == 0:
                        now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        print(f"[{now_str}]   Processed {rows_processed:,} rows, kept {rows_kept:,} matching rows...", flush=True)
        
        # Sort chunks by batch index to maintain order
        filtered_chunks = [filtered_chunks[i] for i in sorted(filtered_chunks.keys())]
        
    except Exception as e:
        print(f"Parallel processing failed ({e}), falling back to sequential...", flush=True)
        use_parallel = False

if not use_parallel or max_workers <= 1:
    # Sequential processing
    filtered_chunks = []
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        rows_in_batch = batch.num_rows
        rows_processed += rows_in_batch
        
        # Filter using PyArrow compute (much faster than pandas)
        if channels is not None:
            mask = pc.is_in(batch['channel_id'], value_set=pa.array(channels))
            batch = batch.filter(mask)
        
        if videos is not None:
            mask = pc.is_in(batch['video_id'], value_set=pa.array(videos))
            batch = batch.filter(mask)
        
        # Only convert to pandas if there's data left after filtering
        if batch.num_rows > 0:
            df_chunk = batch.to_pandas()
            filtered_chunks.append(df_chunk)
            rows_kept += len(df_chunk)
        
        if rows_processed % 100000000 == 0:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[{now_str}]   Processed {rows_processed:,} rows, kept {rows_kept:,} matching rows...", flush=True)

print(f"Finished streaming. Processed {rows_processed:,} total rows, kept {rows_kept:,} matching rows", flush=True)

# Concatenate all filtered chunks
print("Concatenating filtered data...", flush=True)
df = pd.concat(filtered_chunks, ignore_index=True)
print(f"Combined into DataFrame with {len(df):,} rows", flush=True)

if not get_all_comments:
    # drop video_ids that have <100 comments
    comment_counts = df['video_id'].value_counts()
    valid_videos = comment_counts[comment_counts >= 100].index
    df = df[df['video_id'].isin(valid_videos)]
    print(f"{len(df):,} comments remaining after dropping videos with <100 comments", flush=True)

# Sample videos per channel if requested
if sample_by_channel and not get_all_comments:
    print("Sampling videos per channel...", flush=True)
    sampled_videos = set()
    for channel_id, group in df.groupby('channel_id'):
        unique_videos = group['video_id'].unique()
        n_sample = min(num_videos_per_channel, len(unique_videos))
        sampled = pd.Series(unique_videos).sample(n=n_sample, random_state=42).tolist()
        sampled_videos.update(sampled)
    
    df = df[df['video_id'].isin(sampled_videos)]
    print(f"Sampled {len(sampled_videos)} videos, {len(df):,} comments remaining", flush=True)

# Sample comments per video
if not get_all_comments:
    print("Sampling comments per video...", flush=True)
    sampled_comments = []
    for video_id, group in df.groupby('video_id'):
        n_sample = min(num_comments_per_video, len(group))
        sampled = group.sample(n=n_sample, random_state=42)
        sampled_comments.append(sampled)

    df_sampled = pd.concat(sampled_comments, ignore_index=True)
    print(f"Sampled {len(df_sampled):,} total comments from {len(sampled_comments)} videos", flush=True)
else:
    df_sampled = df
    print(f"Kept all {len(df_sampled):,} comments from {df_sampled['video_id'].nunique()} videos", flush=True)

# Create output directory if it doesn't exist
if args.output_file:
    output_file = args.output_file
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
else:
    output_dir = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/comment_exports'
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename with date and number of comments
    date_str = datetime.now().strftime('%Y%m%d')
    num_comments = len(df_sampled)
    output_file = os.path.join(output_dir, f'{date_str}_{num_comments}.parquet')

# Save to Parquet
df_sampled.to_parquet(output_file, index=False)
num_channels_saved = df_sampled['channel_id'].nunique()
print(f"Saved {num_comments} comments from {num_channels_saved} unique channels to {output_file}", flush=True)
