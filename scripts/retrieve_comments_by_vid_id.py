"""
Get all comments for a set of video ids from a large compressed ndjson file.
"""

import zstandard
import json, csv, sys
import os
import io
from datetime import datetime


# Load target_video_ids from JSON file
with open(os.path.join(os.path.dirname(__file__), '../configs/target_video_ids.json'), 'r') as f:
	target_video_ids_dict = json.load(f)

# Flatten all video ids to a set for fast lookup, and build a reverse map from video_id to channel_id
video_id_to_channel = {}
for channel_id, video_ids in target_video_ids_dict.items():
	for vid in video_ids:
		video_id_to_channel[vid] = channel_id
target_video_ids_set = set(video_id_to_channel.keys())
print(f"Loaded {len(target_video_ids_set)} target video IDs from {len(target_video_ids_dict)} channels.", flush=True)

input_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/youtube_comments.ndjson.zst'
output_file = '/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/retrieved_youtube_comments_take2.ndjson'

dctx = zstandard.ZstdDecompressor()
csv.field_size_limit(sys.maxsize)

line_count = 0
with open(input_file, 'rb') as ifh, open(output_file, 'w') as ofh:
    reader = dctx.stream_reader(ifh)
    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
    print(f"Started processing at {datetime.now().isoformat()}", flush=True)
    csv_reader = csv.reader(text_stream)
    for row in csv_reader:
        line_count += 1
        if line_count == 1:
            # Always write the first line (header or otherwise)
            ofh.write(json.dumps(row) + '\n')
            continue
        if line_count % 10000000 == 0:
            print(f"Processed {line_count} lines at {datetime.now().isoformat()}", flush=True)
        if not row or len(row) < 1:
            continue
        vid = row[2]
        if vid in target_video_ids_set:
            # Add channel_id to the row as a new field
            row.append(video_id_to_channel[vid])
            ofh.write(json.dumps(row) + '\n')



