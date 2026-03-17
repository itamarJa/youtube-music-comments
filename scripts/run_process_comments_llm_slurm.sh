#!/bin/bash
#SBATCH --job-name=llm_comment_labeling
#SBATCH --output=logs/llm_comment_labeling_%j.out
#SBATCH --error=logs/llm_comment_labeling_%j.err
#SBATCH --time=71:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --account=margulis

source ~/.bashrc
conda activate music-narr-llm
module load proxy/default

python /scratch/gpfs/HASSON/ij9216/projects/code/youtube-music-comments/scripts/process_comments_with_llm.py \
  --input-file /scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/comment_exports/final_topic_comments.parquet \
  --inp-col content \
  --prompt-name eval_music_rel \
  --llm-name gpt-4o-mini \
  --batch-size 64 \
  --report-every 10000 \
  --api-batch-size 500 \
  --rpm-limit 1500 \
  --min-batch-seconds 0 \
  --resume \
  --checkpoint-file /scratch/gpfs/MARGULIS/ij9216/projects/data/youtube-comments/comment_exports/final_topic_comments_eval_music_rel_gpt-4o-mini.checkpoint.csv \
  --checkpoint-every-chunks 10
