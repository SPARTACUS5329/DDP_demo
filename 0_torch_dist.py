import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29501"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    tensor = torch.tensor([rank+1.0])
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    print(f"Rank {rank}, mean grad: {tensor.item()}")


def main():
    mp.spawn(run, args=(4,), nprocs=4, join=True)

if __name__ == "__main__":
    main()

