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

from models.prompts.deep_prompt import DeepPrompt
from models.encoder import VisionEncoderWithDeepPrompts


class PromptLearning(pl.LightningModule):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cls_lr = cfg.train.cls_lr
        self.prompt_lr = cfg.train.prompt_lr
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
        
        # Deep prompts
        self.deep_prompt = DeepPrompt(
            cfg,
            hidden_size=hidden_size,
        )
        
        # Encoder with deep prompts
        self.vision_encoder = VisionEncoderWithDeepPrompts(self.medsiglip, self.deep_prompt)
                
        # Classifier Head 
        self.classifier = nn.Linear(2 * hidden_size, self.num_classes)      
                
        # Buffers for metrics
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []
        
    def encode_vision(self, images):
        return self.vision_encoder(images)
    
    def forward(self, cc_images, mlo_images):        
        # Encode images
        cc_features = self.encode_vision(cc_images)  
        mlo_features = self.encode_vision(mlo_images) 
        
        # Concat Features
        features = torch.cat([cc_features, mlo_features], dim=1)  
        
        # Classification Head
        cls_score = self.classifier(features)
        
        return {
            "cls_scores": cls_score,
        }
        
    def compute_loss(self, outputs, labels):
        cls_score = outputs['cls_scores'] 
        ce_loss = F.cross_entropy(cls_score, labels)
        
        return ce_loss, {                      
            "total": ce_loss,                  
        }
    
    def training_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        
        outputs = self(cc_images, mlo_images)
        loss, loss_dict = self.compute_loss(outputs, labels)
        
        # Store predictions
        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        self.train_preds.append(probs.detach().cpu())
        self.train_labels.append(labels.detach().cpu())
        
        # Logging
        self.log('train_loss', loss_dict['total'], on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        
        outputs = self(cc_images, mlo_images)
        loss, loss_dict = self.compute_loss(outputs, labels)
        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        
        self.val_preds.append(probs.cpu())
        self.val_labels.append(labels.cpu())
        
        self.log('val_loss', loss_dict['total'], on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        cc_images = batch['cc']
        mlo_images = batch['mlo']
        labels = batch['label']
        
        outputs = self(cc_images, mlo_images)
        probs = torch.softmax(outputs['cls_scores'], dim=-1)
        
        self.test_preds.append(probs.cpu())
        self.test_labels.append(labels.cpu())
    
    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return
        
        preds = torch.cat(self.train_preds, dim=0)  
        labels = torch.cat(self.train_labels, dim=0)  

        # Metrics
        pred_classes = torch.argmax(preds, dim=1).numpy()
        labels_np = labels.numpy()
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        
        self.log('train_f1', f1, prog_bar=True)
        self.log('train_auc', auc, prog_bar=True)
        
        # Clear buffers
        self.train_preds.clear()
        self.train_labels.clear()
    
    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
        
        preds = torch.cat(self.val_preds, dim=0)  
        labels = torch.cat(self.val_labels, dim=0)  
        pred_classes = torch.argmax(preds, dim=1).numpy()
        labels_np = labels.numpy()
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        
        self.log('val_f1', f1, prog_bar=True, on_epoch=True)
        self.log('val_auc', auc, prog_bar=True, on_epoch=True)
        
        self.val_preds.clear()
        self.val_labels.clear()
    
    def on_test_epoch_end(self):
        if len(self.test_preds) == 0:
            return
        
        preds = torch.cat(self.test_preds, dim=0)
        labels = torch.cat(self.test_labels, dim=0)
        
        pred_classes = preds.argmax(dim=1).numpy()
        labels_np = labels.numpy()
        
        # ---------------- Metrics ----------------
        f1 = f1_score(labels_np, pred_classes, average='macro', zero_division=0)
        auc = auroc(
            preds,
            labels,
            task="multiclass",
            num_classes=self.num_classes,
            average="macro"
        )
        cm = confusion_matrix(labels_np, pred_classes)        
        # ---------------- Print Results ----------------
        print(f"\n{'='*70}")
        print("STAGE 1 TEST RESULTS")
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
        """Optimize only prompts and classifier"""
        prompt_params = list(self.deep_prompt.parameters())
        classifier_params = list(self.classifier.parameters())
        
        optimizer = AdamW(
            [
                {
                    "params": prompt_params,
                    "lr": self.prompt_lr,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": classifier_params,
                    "lr": self.cls_lr,
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
