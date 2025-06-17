import torch

# 檢查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA 是否可用 (Is CUDA available?): {cuda_available}")


pair = '(10, 9)'
pair = eval(pair)
print(f"Evaluated pair: {pair}, type: {type(pair)}")

if cuda_available:
    # 顯示可用的 GPU 數量
    device_count = torch.cuda.device_count()
    print(f"找到的 GPU 數量 (Number of GPUs found): {device_count}")
    
    # 顯示目前 GPU 的名稱
    current_device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    print(f"目前使用的 GPU 名稱 (Current GPU name): {current_device_name}")