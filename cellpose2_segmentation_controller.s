#!/bin/bash
#SBATCH --out=cellpose2_run_%j.out
#SBATCH --error=cellpose2_run_%j.err
#SBATCH --time=0-3

module load condaenvs/new/cellpose2

source cellpose2_config.txt

tiff=$(head -n${SLURM_ARRAY_TASK_ID} ${tiff_manifest} | tail -n1)
echo ${tiff}

out_dir=$(head -n${SLURM_ARRAY_TASK_ID} ${out_dir_manifest} | tail -n1)
echo ${out_dir}

file_name=$(basename ${tiff})
basename_no_ext="${file_name%%.*}"

out_dir=${out_dir}/${basename_no_ext}

mkdir -p ${out_dir}

python3 cellpose2_segmentation.py ${tiff} ${out_dir}/ \
    --diameter ${diameter} \
    --channels 0 ${nuc_channel} \
    --flow-threshold ${flow_threshold} \
    --cellprob-threshold ${cellprob_threshold}
