#!/bin/bash
#SBATCH --job-name=PRocces
#SBATCH --output=output_%j.log     # Log file with job ID
#SBATCH --error=error_%j.log       # Error log with job ID
#SBATCH --nodes=1                  # Single node
#SBATCH --ntasks=1                 # One main task
#SBATCH --cpus-per-task=15         # CPUs per task (adjust based on needs)
#SBATCH --time 00:40:00            # Adjust as needed
#SBATCH --mem=30G                  # Memory allocation
#SBATCH --gres=gpu:0              # Request 1 GPU (set the number as needed)
#SBATCH --partition=cpu-short     # Specify a GPU-enabled partition


echo "[$SHELL] ## current working directory: "$CWD

# Source the Python virtual environmentsqueue

#source /home/s2334356/data1/.pyenv/.penny_mps/bin/activate


nvidia-smi
module list
module load Python/3.11.3-GCCcore-12.3.0


source /home/s2334356/data1/.pyenv/JAX_env/bin/activate

module list
nvidia-smi

echo "[$SHELL] #### Starting PRoccesng"
echo "[$SHELL] ## This is $SLURM_JOB_USER on $HOSTNAME with job ID $SLURM_JOB_ID"

# Set current working directory and PYTHONPATH
export CWD=$(pwd)
export PLS=$(pip list)
echo "[$SHELL] ## current working directory: "$CWD
echo "[$SHELL] ## Pip list: " $PLS
export PYTHONPATH=$PYTHONPATH:/home/s2334356/data1/MPS_Models


# Run the Python script
echo "[$SHELL] ## Running script"

metrics="/home/s2334356/data1/20Qubitss/export_20_7july/Checkpoint_Metrcs"
#python3 /home/s2334356/data1/JAX_Quimb_Version/Recalc_Wass.py "--metrics_folders=${metrics}" 
#python3 /home/s2334356/data1/JAX_Quimb_Version/Recalc_Wass_old.py "--metrics_folders=${metrics}" 
python3 /home/s2334356/data1/JAX_Quimb_Version/recalc_original.py

echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python. Have a nice day"

