import torch, platform, sys, os
print("PyTorch:", torch.__version__)
print("CUDA run-time visible a PyTorch:", torch.version.cuda)
print("¿GPU disponible?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Nombre de la GPU:", torch.cuda.get_device_name(0))
