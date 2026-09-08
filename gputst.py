import torch



# 1. PyTorch 版本
print("PyTorch版本:", torch.__version__)

# 2. CUDA版本
print("PyTorch CUDA版本:", torch.version.cuda)

# 3. CUDA是否可用
print("CUDA是否可用:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("\n❌ GPU不可用")
    exit()

# 4. GPU数量
print("GPU数量:", torch.cuda.device_count())

# 5. GPU名称
print("GPU型号:", torch.cuda.get_device_name(0))

