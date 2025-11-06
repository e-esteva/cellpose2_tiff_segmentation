#!/bin/bash
#SBATCH --mem=10GB
#SBATCH --time=0-3
#SBATCH --error=cellpose2_segmentation_%j.err
#SBATCH --out=cellpose2_segmentation_%j.out

cellpose2_segmentation_configFile=$1

echo ${cellpose2_segmentation_configFile}

source ${cellpose2_segmentation_configFile}


# run cellpose2_segmentation module
function cellpose2_segmentation_module {
    local job_id=($(sbatch --export=configfile=${cellpose2_segmentation_configFile} --mail-user=${user} --array=1-${sample_count} --mem=${cellpose2_mem} --partition=a100_short --gres=gpu:${gpus} ${cellpose2_segmentation_module_Path}))
    echo ${job_id[3]}
}



### MODULE 1:
echo cellpose2_segmentation at `date`
mod1_job=$(cellpose2_segmentation_module)
echo Finished performing segmentation at `date`


