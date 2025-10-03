# A Brief Introduction to PyTorch DDP Model Training
Slide: https://docs.google.com/presentation/d/1y15fzHoBhjBFpmy44Q3VrW0gJMS7ePhCVsRgXNmqC-g/edit?usp=sharing

## Useful Commands 

    Allocate node: salloc --nodes 1 --qos interactive --time 01:00:00 --constraint gpu --gpus-per-node 4 --account=<acc_name>_g
    Enable PyTorch: module load pytorch/2.6.0
    Check for slow import: python -X importtime file.py
    Deep profiling: 
    - pip install -U tensorboard torch-tb-profiler
    - tensorboard --logdir=./log

## Files
- `0_torch_dist.py` Launch 4 processes and performs all_reduce
- `1_torch_dist_gpu.py` Launch 4 processes and performs all_reduce in GPU
- `2_mnist_training.py` A simple MNIST classification pipeline using a single GPU
- `3_mnist_distributed.py` 4 identical processes performing exactly same task - a simple MNIST classification pipeline GPU
- `4_mnist_manual_ddp.py` A manual implementation of DDP on previous pipeline with distributed sampler. Output saved as `./log_ddp`. Inspect using `tensorboard --logdir=./log_ddp`.
- `5_mnist_manual_ddp_profile.py` Tensorboard profiler on previous code, reduced epoch for smaller profile data
- `6_mnist_ddp_pt.py` PyTorch wrapper for DDP. No manual all_reduce needed
- `7_mnist_ddp_pt_timing.py` PyTorch DDP code with arg = GPU number. Prints out the time taken for model training. arg = 4 should give lower runtime.
- `8_mnist_ddp_pt_lr.py` Adjustment of learning rate so that the loss curve matches single GPU.
- `test-profile.py` A smaller DDP ML pipeline with dummy data and Tensorboard profiling for easier profiling viewing. Output saved as `./log`. Inspect using `tensorboard --logdir=./log`.

