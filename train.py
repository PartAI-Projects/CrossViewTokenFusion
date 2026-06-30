import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
)

from dataset.datamodule import MammoDataModule
from models.stage1 import PromptLearning
from models.stage2 import FusionTokenLearning
from models.prompts.checkpoint import SavePrompts


def build_model(cfg, stage):
    if stage == 1:
        return PromptLearning(cfg)

    elif stage == 2:
        return FusionTokenLearning(cfg)

    raise ValueError(f"Unknown stage: {stage}")


def train(cfg, stage):
    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    data_module = MammoDataModule(cfg)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = build_model(cfg, stage)

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------
    checkpoint_f1 = ModelCheckpoint(
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        filename=f"stage{stage}_best_f1"
                 "-{epoch:02d}-{val_f1:.4f}",
    )

    checkpoint_last = ModelCheckpoint(
        save_last=True,
        filename=f"stage{stage}_last",
    )

    callbacks = [
        checkpoint_f1,
        checkpoint_last,
        LearningRateMonitor(logging_interval="step"),
    ]

    # Save prompts only in stage 1
    if stage == 1:
        callbacks.append(SavePrompts(cfg))

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        default_root_dir=cfg.output_dir,
        callbacks=callbacks,
        max_epochs=cfg.train.epochs,
        accelerator="cuda",
        devices=1,
        deterministic=True,
        log_every_n_steps=10,
    )

    print("\n" + "=" * 70)
    if stage == 1:
        print(" STAGE 1: Learning Deep Prompts")
    else:
        print(" STAGE 2: Training Fusion Tokens")
    print("=" * 70 + "\n")

    # --------------------------------------------------
    # Train
    # --------------------------------------------------
    trainer.fit(model, data_module)
    print("\n✓ Training complete!")

    # --------------------------------------------------
    # Test 
    # --------------------------------------------------
    checkpoints = {
        "best_f1": checkpoint_f1.best_model_path,
        "last": checkpoint_last.last_model_path,
    }
    print("\n========== TESTING CHECKPOINTS ==========\n")
    for name, ckpt_path in checkpoints.items():
        if not ckpt_path:
            continue
        print(f"\n▶️ Loading {name} checkpoint:")
        print(ckpt_path)

        if stage == 1:
            best_model = PromptLearning.load_from_checkpoint(
                ckpt_path,
                cfg=cfg,
            )

        else:
            best_model = FusionTokenLearning.load_from_checkpoint(
                ckpt_path,
                cfg=cfg,
            )

        print(f"→ Testing {name} model...")
        trainer.test(
            model=best_model,
            datamodule=data_module,
        )
