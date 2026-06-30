import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.io import imread

from dataset.transforms import get_transforms


class BaseMammoPairDataset(Dataset):
    def __init__(self, samples, processor, train, use_aug=False):
        self.samples = samples
        self.processor = processor
        self.train = train
        self.use_aug = use_aug
        self.transforms = get_transforms()

    def __len__(self):
        return len(self.samples)

    def _load_image(self, path):
        img = imread(path).astype(np.uint8)
        if self.train and self.use_aug:
            img = self.transforms(image=img)["image"]
        img = Image.fromarray(img).convert("RGB")
        return img

    def _encode(self, img):
        return self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

    def __getitem__(self, idx):
        s = self.samples[idx]
        cc_img = self._load_image(s["cc_path"])
        mlo_img = self._load_image(s["mlo_path"])
        return {
            "cc": self._encode(cc_img),
            "mlo": self._encode(mlo_img),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }

    def get_labels(self):
        return [int(s["label"]) for s in self.samples]
    

class VinDrMammoDataset(BaseMammoPairDataset):
    def __init__(self, data, data_dir, processor, train, use_aug=False):
        samples = []
        data = data.reset_index(drop=True)
        grouped = data.groupby(["study_id", "laterality"])
        for (study_id, lat), group in grouped:
            cc = group[group["view_position"] == "CC"]
            mlo = group[group["view_position"] == "MLO"]
            if cc.empty or mlo.empty:
                continue
            cc_path = os.path.join(
                data_dir,
                study_id,
                cc.iloc[0]["image_id"] + ".png"
            )
            mlo_path = os.path.join(
                data_dir,
                study_id,
                mlo.iloc[0]["image_id"] + ".png"
            )

            if not os.path.exists(cc_path) or not os.path.exists(mlo_path):
                continue

            label = group["label"].dropna().iloc[0] \
                if "label" in group.columns else None
            if label is None:
                continue
            samples.append({
                "cc_path": cc_path,
                "mlo_path": mlo_path,
                "label": label
            })

        super().__init__(samples, processor, train=train, use_aug=use_aug)

class CMMDDataset(BaseMammoPairDataset):
    def __init__(self, data, data_dir, processor, train, use_aug=False):
        samples = []
        data = data.reset_index(drop=True)
        grouped = data.groupby(["ID1", "laterality"])
        for (pid, lat), group in grouped:
            cc = group[group["view"] == "CC"]
            mlo = group[group["view"] == "MLO"]
            if cc.empty or mlo.empty:
                continue
            cc_path = os.path.join(data_dir, cc.iloc[0]["image_path"])
            mlo_path = os.path.join(data_dir, mlo.iloc[0]["image_path"])

            if not os.path.exists(cc_path) or not os.path.exists(mlo_path):
                continue

            label = cc.iloc[0]["label"]
            samples.append({
                "cc_path": cc_path,
                "mlo_path": mlo_path,
                "label": label
            })

        super().__init__(samples, processor, train=train, use_aug=use_aug)