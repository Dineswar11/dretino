! /bin/bash
SBATCH -N 4
SBATCH --ntasks-per-node=128
SBATCH --gres=gpu:A100-SXM4:8
SBATCH --time=00:30:00
SBATCH --error=job.%J.err
SBATCH --output=job.%J.out
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
cd $SLURM_SUBMIT_DIR

source Conda/bin/activate

conda activate pytorch

cd dretino/dretino

python3 ssl_dino.py

