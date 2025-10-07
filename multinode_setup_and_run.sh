export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29600
srun -l python $1
# srun --exact --ntasks=$2 python $1
