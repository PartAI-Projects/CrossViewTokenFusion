import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoProcessor
from torchsampler import ImbalancedDatasetSampler

from dataset.dataset import VinDrMammoDataset, CMMDDataset


class MammoDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.data_dir = cfg.data.prep_dir
        self.dataset_name = cfg.data.data_name.lower()

        if self.dataset_name == "vindr":
            self.csv_path = "./data/breast-level_annotations_with_split.csv"

        elif self.dataset_name == "cmmd":
            self.csv_path = "./data/CMMD_with_split.csv"

        else:
            raise ValueError(
                f"Invalid dataset name: {self.dataset_name}"
            )
        
        self.batch_size = cfg.data.batch_size
        self.num_workers = cfg.data.num_workers
        self.processor = AutoProcessor.from_pretrained(
            cfg.model.model_id,
            use_fast=True
        )

    def setup(self, stage=None):
        if self.dataset_name == "vindr":
            train_df, val_df, test_df = self._prepare_vindr()
            dataset_cls = VinDrMammoDataset

        elif self.dataset_name == "cmmd":
            train_df, val_df, test_df = self._prepare_cmmd()
            dataset_cls = CMMDDataset

        else:
            raise ValueError(
                f"Unknown dataset: {self.dataset_name}"
            )

        if stage in ("fit", None):
            self.train_dataset = dataset_cls(
                train_df,
                self.data_dir,
                self.processor,
                train=True,
                use_aug=self.cfg.train.use_augmentation,
            )
            self.val_dataset = dataset_cls(
                val_df,
                self.data_dir,
                self.processor,
                train=False,
            )
            self._print_statistics()
            
        if stage in ("test", None):
            self.test_dataset = dataset_cls(
                test_df,
                self.data_dir,
                self.processor,
                train=False
            )

    def _prepare_vindr(self):
        df = pd.read_csv(self.csv_path)
        task = self.cfg.data.task

        if task == "binary":
            df = df[
                ~df["breast_birads"].isin(
                    ["BI-RADS 1", "BI-RADS 3"]
                )
            ].reset_index(drop=True)
            mapping = {
                "BI-RADS 2": 0,
                "BI-RADS 4": 1,
                "BI-RADS 5": 1
            }
            split_col = "binary_split"

        elif task == "multiclass":
            mapping = {
                "BI-RADS 1": 0,
                "BI-RADS 2": 1,
                "BI-RADS 3": 2,
                "BI-RADS 4": 3,
                "BI-RADS 5": 4
            }
            split_col = "split"

        else:
            raise ValueError(
                f"Unknown VinDr task: {task}"
            )

        df["label"] = df["breast_birads"].map(mapping)
        train_df = df[df[split_col] == "training"]
        val_df = df[df[split_col] == "validation"]
        test_df = df[df[split_col] == "test"]
        return train_df, val_df, test_df

    def _prepare_cmmd(self):
        df = pd.read_csv(self.csv_path)
        mapping = {
            "benign": 0,
            "malignant": 1
        }
        df["label"] = df["classification"].map(mapping)
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "val"]
        test_df = df[df["split"] == "test"]
        return train_df, val_df, test_df

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=ImbalancedDatasetSampler(
                self.train_dataset
            ),
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _print_statistics(self):
        """Print dataset statistics"""
        train_labels = self.train_dataset.get_labels()
        val_labels = self.val_dataset.get_labels()
        train_classes, train_class_count = np.unique(train_labels, return_counts=True)
        val_classes, val_class_count = np.unique(val_labels, return_counts=True)
        print(f'\nSamples (train): {len(self.train_dataset)}')
        print(f'Samples (val):   {len(self.val_dataset)}')
        print('\nClass distribution (train):')
        for cls, count in zip(train_classes, train_class_count):
            print(f'  Class {cls}: {count} ({count/len(train_labels)*100:.2f}%)')
        print('\nClass distribution (val):')
        for cls, count in zip(val_classes, val_class_count):
            print(f'  Class {cls}: {count} ({count/len(val_labels)*100:.2f}%)')