"""Image-only adapter for the frozen neural comparison model."""
from pathlib import Path
import numpy as np
import torch
from .architecture import SmallCNN

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
_MODEL = SmallCNN()
_MODEL.load_state_dict(torch.load(Path(__file__).with_name("weights.pt"), map_location="cpu", weights_only=True))
_MODEL.eval()

def predict(image):
    with torch.inference_mode():
        x = torch.from_numpy(np.asarray(image).astype(np.float32) / 255.0).reshape(1, 1, 28, 28)
        return int(_MODEL(x).argmax(dim=1).item())
