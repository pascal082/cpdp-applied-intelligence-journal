WBDA Transfer Learning for Software defect prediction
=======================================================================================================================================================================

# Conda Installation

```sh

#Code used for ICSE 2022
# 
# To set up on the Cluster
module load Miniconda3/4.9.2
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /nesi/nobackup/uoo03229/cond_envs/mwlr_tf/
conda install pip


#conda install -c conda-forge r-base==3.7
conda create --prefix icse_tf  python=3.7 # to install conda base
conda activate icse_tf
```


# Run 

Cd to the default config file and update the Config/Software_Defect/software_defect.yaml file and update the location of data files.
Please see the bash script on how to run the files


The dataset used for the ICSE paper is located in Data/AEEEM

python run.py -type train -config-file /path_to_confi_file/software_defect.yaml 
