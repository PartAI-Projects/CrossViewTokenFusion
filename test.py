import pytorch_lightning as pl

from dataset.datamodule import MammoDataModule
from models.stage1 import PromptLearning
from models.stage2 import FusionTokenLearning


def test(cfg, stage, checkpoint):
    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    data_module = MammoDataModule(cfg)

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    trainer = pl.Trainer(
        accelerator="cuda",
        devices=1,
        deterministic=True,
        enable_checkpointing=False,
    )

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    if stage == 1:
        model = PromptLearning.load_from_checkpoint(
            checkpoint,
            cfg=cfg,
        )

    elif stage == 2:
        model = FusionTokenLearning.load_from_checkpoint(
            checkpoint,
            cfg=cfg,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    # --------------------------------------------------
    # Test
    # --------------------------------------------------
    print(f"\nTesting checkpoint:\n{checkpoint}\n")

    trainer.test(
        model=model,
        datamodule=data_module,
    )
