#!/bin/bash
#SBATCH --job-name=get_yt_comments
#SBATCH --account=margulis
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/get_yt_comments_%j.out
#SBATCH --error=logs/get_yt_comments_%j.err

# Load conda environment
source ~/.bashrc
conda activate music-narr-llm

# Run the script
python scripts/get_comments_by_ch_or_vid.py \
    --channels_csv data/final_topic_channels.csv \
    --channel_col channel_id \
    --get_all \
    --output_file /scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/comment_exports/final_topic_comments.parquet
