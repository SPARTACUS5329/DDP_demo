# Useful Commands 

    salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node 4 --account=<acc_name>_g
    module load pytorch/2.6.0
    python -X importtime file.py