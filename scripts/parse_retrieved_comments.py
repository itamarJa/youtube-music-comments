import pandas as pd
import numpy as np
import json, re

inp_f = "/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/retrieved_youtube_comments.ndjson"

# Read all lines
with open(inp_f, 'r') as f:
    lines = f.readlines()

# First line is header
header = json.loads(lines[0])
header.append("channel_id")  # Add the new column name
# Remaining lines are data
data = [json.loads(line) for line in lines[1:]]

# Create DataFrame
df = pd.DataFrame(data, columns=header)

def concat_content_text(cell):
    # Parse the string to a Python object
    try:
        items = json.loads(cell)
    except Exception:
        # If already a list (not a string), just use it
        items = cell
    # Concatenate all 'text' fields
    return ''.join(item.get('text', '') for item in items)

df['content_concat'] = df['content'].apply(concat_content_text)
df['content_clean'] = df['content_concat'].apply(lambda x: re.sub(r'[\n\r\t\\]+', ' ', x))

# Group by video_id and subsample

def subsample(group):
    if len(group) > 150:
        return group.sample(n=150, random_state=42)
    else:
        return group

subsample_df = df.groupby('video_id', group_keys=False).apply(subsample).reset_index(drop=True)

subsample_df.to_csv("/scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/subsampled_youtube_comments_1721com_13vid.csv", index=False)
print('what_now?')