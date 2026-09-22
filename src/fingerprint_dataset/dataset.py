# dataset.py
from pathlib import Path
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class FingerprintSample:
    finger_id: int
    impression_id: int
    path: Path

    def load(self) -> np.ndarray:
        img = Image.open(self.path).convert("L")
        return np.array(img)


class FVCDataset:
    def __init__(self, root_dir: str, extension: str = ".tif"):
        self.root_dir = Path(root_dir)
        self.extension = extension
        self.samples: list[FingerprintSample] = self._index()

    def _index(self) -> list[FingerprintSample]:
        samples = []

        for path in sorted(self.root_dir.glob(f"*{self.extension}")):
            stem = path.stem
            finger_str, impression_str = stem.split("_")

            samples.append(
                FingerprintSample(
                    finger_id=int(finger_str),
                    impression_id=int(impression_str),
                    path=path,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def fingers(self) -> list[int]:
        return sorted(set(s.finger_id for s in self.samples))

    def impressions_of(self, finger_id: int) -> list[FingerprintSample]:
        return [
            s for s in self.samples
            if s.finger_id == finger_id
        ]

    def genuine_pairs(
        self,
    ) -> list[tuple[FingerprintSample, FingerprintSample]]:
        pairs = []

        for finger_id in self.fingers():
            samples = self.impressions_of(finger_id)
            pairs.extend(combinations(samples, 2))

        return pairs

    def impostor_pairs(
        self,
        limit: int | None = None,
    ) -> list[tuple[FingerprintSample, FingerprintSample]]:
        pairs = []

        for f1, f2 in combinations(self.fingers(), 2):
            for s1, s2 in product(
                self.impressions_of(f1),
                self.impressions_of(f2),
            ):
                pairs.append((s1, s2))

                if limit and len(pairs) >= limit:
                    return pairs

        return pairs


if __name__ == "__main__":
    ds = FVCDataset("data/FVC2002_DB2_B")

    print(f"Total images: {len(ds)}")
    print(f"Number of fingers: {len(ds.fingers())}")

    genuine = ds.genuine_pairs()

    impostor = ds.impostor_pairs(
        limit=len(genuine)
    )

    print(f"Genuine pairs: {len(genuine)}")
    print(f"Impostor pairs: {len(impostor)}")

    sample = ds.samples[0]
    img = sample.load()

    print(
        f"Sample: "
        f"finger={sample.finger_id}, "
        f"impression={sample.impression_id}, "
        f"shape={img.shape}"
    )