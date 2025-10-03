import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Function to be run in each process
def run(rank, world_size):
    # Initialize the process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # Set device for each process
    torch.cuda.set_device(rank)
    # Create a tensor and perform all-reduce operation
    tensor = torch.tensor([rank+1.0]).to(rank)
    print(f"Rank {rank}, initial tensor: {tensor}")
    # All-reduce to compute the mean across all processes
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    print(f"Rank {rank}, mean grad: {tensor.item()}")
    # Clean up
    dist.destroy_process_group()

# Entry point
def main():
    print('spawing process')
    # Spawn 4 processes
    mp.spawn(run, args=(4,), nprocs=4, join=True)

if __name__ == "__main__":
    main()

