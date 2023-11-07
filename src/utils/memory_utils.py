import psutil
import GPUtil

def print_cpu_memory():
    # Print the CPU memory usage
    memory = psutil.virtual_memory()
    print(f"CPU Memory Usage: {memory.used / (1024 ** 3):.2f} GB / {memory.total / (1024 ** 3):.2f} GB")

def print_gpu_memory():
    # Get the first GPU details if available
    gpus = GPUtil.getGPUs()
    if len(gpus) > 0:
        gpu = gpus[0]
        print(f"GPU Memory Usage: {gpu.memoryUsed} MB / {gpu.memoryTotal} MB")
    else:
        print("No NVIDIA GPU found on this system.")