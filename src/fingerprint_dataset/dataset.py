from pathlib import Path
import numpy as np
from PIL import Image

class FingerprintDataset:
    def __init__(self, root: str):
        self.root = Path(root)
        self.images = sorted(self.root.glob("*.tif"))
    def __len__(self):
        return len(self.images)
    def __getitem__(self,index):
        image=Image.open(self.images[index])
        filename = self.images[index].stem
        subject_id, capture_id = map(int, filename.split("_"))
        return {
                "image": np.array(image),
                "subject_id": subject_id,
                "capture_id": capture_id,
        }