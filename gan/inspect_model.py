import torch
import os

path = 'generator.pth'
if not os.path.exists(path):
    # try looking in parent or relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths = [
        os.path.join(script_dir, 'generator.pth'),
        os.path.join(script_dir, '..', 'generator.pth'),
        'd:\\github\\pyprj\\demo\\generator.pth'
    ]
    for p in paths:
        if os.path.exists(p):
            path = p
            break

print(f"Loading {path}...")
try:
    state_dict = torch.load(path, map_location='cpu', weights_only=True)
    print("Keys found in state_dict:")
    for key, tensor in state_dict.items():
        print(f"{key}: {tensor.shape}")
except Exception as e:
    print(f"Error: {e}")
