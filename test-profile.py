import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity

# Toy model
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, 1024)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def train(rank, world_size):
    # 1. Init process group
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # 2. Setup model + optimizer
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # 3. Dummy data
    x = torch.randn(64, 1024).to(rank)
    y = torch.randn(64, 1024).to(rank)

    # 4. Profiler with memory + timing
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,   # 👈 enables memory tracking
        with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler("./log")  # for TensorBoard
    ) as prof:
        for step in range(5):
            optimizer.zero_grad()
            with record_function("forward_pass"):
                out = ddp_model(x)
            with record_function("loss+backward"):
                loss = loss_fn(out, y)
                loss.backward()
            with record_function("optimizer_step"):
                optimizer.step()
            prof.step()  # 👈 marks iteration

    # if rank == 0:
        # Export Chrome trace: chrome://tracing
        # prof.export_chrome_trace("ddp_trace.json")

    dist.destroy_process_group()


def main():
    
    world_size = torch.cuda.device_count()
    print(f"Running on {world_size} GPUs")
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
