import torch, platform
print(torch.__version__)               # 应变成 2.x.x+cu118
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # 你的显卡型号