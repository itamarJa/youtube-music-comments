#!/bin/bash
#SBATCH --job-name=yt_comments_filter
#SBATCH --output=logs/yt_comments_filter_%j.out
#SBATCH --error=logs/yt_comments_filter_%j.err
#SBATCH --time=23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --account=margulis

# Load modules or activate conda environment if needed
source ~/.bashrc
conda activate music-narr-llm

# Run the script
# python /scratch/gpfs/HASSON/ij9216/projects/code/youtube-music-comments/scripts/retrieve_comments_by_vid_id.py
python /scratch/gpfs/HASSON/ij9216/projects/code/youtube-music-comments/scripts/convert_raw_data_to_h5.py

