"""
Get all comments for video ids from YouNiverse metadata and save to h5 format.
"""

import zstandard
import json
import csv
import sys
import os
import io
from datetime import datetime
import pandas as pd

# Load target_video_ids from feather file
metadata_file = '/projects/MARGULIS/youtube-music-comments/data/YouNiverse/yt_metadata_helper_music.feather'
print(f"Loading metadata from {metadata_file}...", flush=True)
metadata_df = pd.read_feather(metadata_file)

# Get unique display_id values
target_video_ids_set = set(metadata_df['display_id'].unique())
print(f"Loaded {len(target_video_ids_set)} unique video IDs from metadata.", flush=True)

input_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/youtube_comments.ndjson.zst'
output_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/retrieved_youtube_comments_music.h5'

dctx = zstandard.ZstdDecompressor()
csv.field_size_limit(sys.maxsize)

# Collect matching rows
matching_rows = []
header = None

line_count = 0
with open(input_file, 'rb') as ifh:
    reader = dctx.stream_reader(ifh)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    print(f"Started processing at {datetime.now().isoformat()}", flush=True)
    csv_reader = csv.reader(text_stream)
    
    for row in csv_reader:
        line_count += 1
        if line_count == 1:
            # Save header
            header = row
            continue
        
        if line_count % 10000000 == 0:
            print(f"Processed {line_count} lines, found {len(matching_rows)} matches at {datetime.now().isoformat()}", flush=True)
        
        if not row or len(row) < 3:
            continue
        
        vid = row[2]
        if vid in target_video_ids_set:
            matching_rows.append(row)

print(f"Finished processing {line_count} lines. Found {len(matching_rows)} matching comments.", flush=True)

# Convert to DataFrame and save as h5
if matching_rows:
    print(f"Converting to DataFrame and saving to h5...", flush=True)
    df = pd.DataFrame(matching_rows, columns=header)
    df.to_hdf(output_file, key='comments', mode='w', format='table', complevel=9)
    print(f"Saved {len(df)} comments to {output_file}", flush=True)
else:
    print("No matching comments found.", flush=True)
