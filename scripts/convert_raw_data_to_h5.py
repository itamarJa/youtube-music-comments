"""
Get all comments for video ids from YouNiverse metadata and save to parquet format.
"""

import zstandard
import json
import csv
import sys
import os
import io
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load target_video_ids from feather file
metadata_file = '/projects/MARGULIS/youtube-music-comments/data/YouNiverse/yt_metadata_helper_music.feather'
print(f"Loading metadata from {metadata_file}...", flush=True)
metadata_df = pd.read_feather(metadata_file)

# Get unique display_id values and create mapping to channel_id
target_video_ids_set = set(metadata_df['display_id'].unique())
video_id_to_channel = dict(zip(metadata_df['display_id'], metadata_df['channel_id']))
print(f"Loaded {len(target_video_ids_set)} unique video IDs from metadata.", flush=True)

input_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/youtube_comments.ndjson.zst'
output_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/youtube_comments_music.parquet'

dctx = zstandard.ZstdDecompressor()
csv.field_size_limit(sys.maxsize)

# Collect matching rows
matching_rows = []
header = None
chunk_size = 100000  # Write every 100k rows (smaller for memory)
total_written = 0
writer = None
schema = None

def concat_content_text(cell):
    try:
        items = json.loads(cell)
    except Exception:
        items = cell
    return ''.join(item.get('text', '') for item in items)

def write_chunk(rows, header, output_file, writer, schema):
    """Write a chunk of rows to parquet file"""
    if not rows:
        return 0, writer, schema
    
    df_chunk = pd.DataFrame(rows, columns=header)
    df_chunk['content'] = df_chunk['content'].apply(concat_content_text)
    
    # Convert to pyarrow table
    table = pa.Table.from_pandas(df_chunk)
    
    # Initialize writer on first chunk
    if writer is None:
        schema = table.schema
        writer = pq.ParquetWriter(output_file, schema, compression='zstd')
    
    writer.write_table(table)
    return len(df_chunk), writer, schema

line_count = 0
with open(input_file, 'rb') as ifh:
    reader = dctx.stream_reader(ifh)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    print(f"Started processing at {datetime.now().isoformat()}", flush=True)
    csv_reader = csv.reader(text_stream)
    
    for row in csv_reader:
        line_count += 1
        if line_count == 1:
            header = row + ['channel_id']
            continue
        
        if line_count % 10000000 == 0:
            print(f"Processed {line_count} lines, found {total_written + len(matching_rows)} matches at {datetime.now().isoformat()}", flush=True)
        
        if not row or len(row) < 3:
            continue
        
        vid = row[2]
        if vid in target_video_ids_set:
            row_with_channel = row + [video_id_to_channel[vid]]
            matching_rows.append(row_with_channel)
            
            if len(matching_rows) >= chunk_size:
                written, writer, schema = write_chunk(matching_rows, header, output_file, writer, schema)
                total_written += written
                matching_rows = []
                print(f"  Wrote chunk, total saved: {total_written} at {datetime.now().isoformat()}", flush=True)

print(f"Finished processing {line_count} lines.", flush=True)

# Write remaining rows
if matching_rows:
    written, writer, schema = write_chunk(matching_rows, header, output_file, writer, schema)
    total_written += written

# Close writer
if writer is not None:
    writer.close()

if total_written > 0:
    print(f"Saved {total_written} comments to {output_file}", flush=True)
else:
    print("No matching comments found.", flush=True)