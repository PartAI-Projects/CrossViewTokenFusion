import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from transformers import AutoProcessor, SiglipForImageClassification
from torchmetrics.functional import auroc
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
)

from models.fusion import FusionBlock
from models.prompts.checkpoint import load_stage1_prompts
from models.encoder import VisionEncoderWithFusionBlocks


class FusionTokenLearning(pl.LightningModule):    
    def __init__(self, cfg):
        super().__init__()
        self.cls_lr = cfg.train.cls_lr
        self.vision_head_lr = cfg.train.vision_head_lr
        self.fusion_lr = cfg.train.fusion_lr
        self.weight_decay = cfg.train.weight_decay
        self.warmup_steps = cfg.train.warmup_steps

        if cfg.data.task == "binary":
            self.num_classes = 2
            self.class_names = ["Normal", "Abnormal"]
        else:
            self.num_classes = 5
            self.class_names = ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BI-RADS 4", "BI-RADS 5"]
                
        # Load MedSigLIP
        model_id = cfg.model.model_id
        self.medsiglip = SiglipForImageClassification.from_pretrained(model_id)
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        
        hidden_size = self.medsiglip.vision_model.config.hidden_size
        
        # Load frozen prompts from Stage 1        
        self.deep_prompt, _ = load_stage1_prompts(
            cfg,
            hidden_size=hidden_size,
        )
        
        # Create encoder with frozen prompts and fusion blocks
        self.vision_encoder = VisionEncoderWithFusionBlocks(
            self.medsiglip, 
            self.deep_prompt
        )
        
        # Fusion blocks
        self.fusion_layers = cfg.model.fusion_layers
        self.fusion_blocks = nn.ModuleList([
            FusionBlock(
                cfg,
                embed_dim=hidden_size,
            )
            for _ in self.fusion_layers
        ])

        # Classifier 
        self.classifier = nn.Linear(2 * hidden_size, self.num_classes)
        self._init_classifier()
        
        # Metrics buffers
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
            
    def _init_classifier(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, cc_images, mlo_images):
        cc_hidden = cc_images
        mlo_hidden = mlo_images
        prev_layer = 0

        for fusion_layer, fusion_block in zip(
                self.fusion_layers,
                self.fusion_blocks):

            # first segment
            if prev_layer == 0:
                cc_hidden = self.vision_encoder.encode_up_to_layer(
                    cc_hidden, fusion_layer
                )
                mlo_hidden = self.vision_encoder.encode_up_to_layer(
                    mlo_hidden, fusion_layer
                )

            # subsequent segments
            else:
                cc_hidden = self.vision_encoder.continue_from_layer(
                    cc_hidden,
                    prev_layer,
                    fusion_layer
                )

                mlo_hidden = self.vision_encoder.continue_from_layer(
                    mlo_hidden,
                    prev_layer,
                    fusion_layer
                )

            # fusion
            cc_hidden, mlo_hidden = fusion_block(
                cc_hidden,
                mlo_hidden
            )

            prev_layer = fusion_layer

        # finish encoder
        cc_features = self.vision_encoder.continue_to_end(
            cc_hidden,
            prev_layer
        )
        mlo_features = self.vision_encoder.continue_to_end(
            mlo_hidden,
            prev_layer
        )
        combined_features = torch.cat(
            [cc_features, mlo_features],
            dim=-1
        )
        cls_scores = self.classifier(combined_features)
        return {"cls_scores": cls_scores}
    
    def compute_loss(self, outputs, labels):
        ce_loss = F.cross_entropy(outputs['cls_scores'], labels)
        return ce_loss, {
            "total": ce_loss,
        }
    
    def training_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        
        outputs = self(cc_images, mlo_images)
        loss, loss_dict = self.compute_loss(outputs, labels)
        
        self.log('train_loss', loss_dict['total'], on_step=True, on_epoch=True, prog_bar=True)
        
        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        self.train_preds.append(probs.detach().cpu())
        self.train_labels.append(labels.detach().cpu())
        return loss
    
    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return
        
        preds = torch.cat(self.train_preds)
        labels = torch.cat(self.train_labels)
        
        pred_classes = preds.argmax(dim=1).numpy()
        labels_np = labels.numpy()
        
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(preds, labels, task="multiclass", 
                    num_classes=self.num_classes, average="macro")
        
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_auc', auc, prog_bar=True)
        
        self.train_preds.clear()
        self.train_labels.clear()
    
    def validation_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        
        outputs = self(cc_images, mlo_images)
        loss, _ = self.compute_loss(outputs, labels)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        self.val_preds.append(probs.cpu())
        self.val_labels.append(labels.cpu())
        return loss
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        
        preds = torch.cat(self.val_preds)
        labels = torch.cat(self.val_labels)
        
        pred_classes = preds.argmax(dim=1).numpy()
        labels_np = labels.numpy()
        
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(preds, labels, task="multiclass", 
                    num_classes=self.num_classes, average="macro")
        
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_auc', auc, prog_bar=True)
        
        self.val_preds.clear()
        self.val_labels.clear()
    
    def test_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        outputs = self(cc_images, mlo_images)
        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        self.test_preds.append(probs.cpu())
        self.test_labels.append(labels.cpu())
    
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        
        preds = torch.cat(self.test_preds)
        labels = torch.cat(self.test_labels)
        
        pred_classes = preds.argmax(dim=1).numpy()
        labels_np = labels.numpy()
        
        # ---------------- Metrics ----------------
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(preds, labels, task="multiclass", 
                    num_classes=self.num_classes, average="macro")
        cm = confusion_matrix(labels_np, pred_classes)
        # ---------------- Print Results ----------------
        print(f"\n{'='*70}")
        print("STAGE 2 TEST RESULTS")
        print(f"{'='*70}")
        print(f"F1 Score (macro): {f1:.4f}")
        print(f"AUC (macro): {auc:.4f}")
        print(f"\nConfusion Matrix:")
        print(cm)
        print(f"\nClassification Report:")
        print(classification_report(
            labels_np, pred_classes,
            target_names=self.class_names,
            zero_division=0
        ))
        print(f"{'='*70}\n")
        
        self.test_preds.clear()
        self.test_labels.clear()
    
    def configure_optimizers(self):
        """Optimize only fusion blocks, vision encoder head and classifier"""
        fusion_params = [
                p
                for block in self.fusion_blocks
                for p in block.parameters()
                if p.requires_grad
            ]

        classifier_params = [
                p for p in self.classifier.parameters()
                if p.requires_grad
            ]

        head_params = [
                p for p in self.vision_encoder.head.parameters()
                if p.requires_grad
            ]

        optimizer = AdamW(
            [
                {
                    "params": classifier_params,
                    "lr": self.cls_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": fusion_params,
                    "lr": self.fusion_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": head_params,
                    "lr": self.vision_head_lr,
                    "weight_decay": self.weight_decay,
                },
            ]
        )

        def lr_lambda(current_step):
            warmup = self.warmup_steps
            total = self.trainer.estimated_stepping_batches
            if current_step < warmup:
                return float(current_step) / float(max(1, warmup))
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }
