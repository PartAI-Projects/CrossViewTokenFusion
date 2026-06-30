import argparse
import pytorch_lightning as pl

from configs.config import Config
from preprocess import preprocess
from train import train
from test import test


def run_preprocess(cfg):
    print("\n========== PREPROCESS ==========\n")
    preprocess(cfg)

def run_train(cfg, stage):
    print(f"\n========== STAGE {stage} TRAINING ==========\n")
    train(cfg, stage)

def run_test(cfg, stage, checkpoint):
    print(f"\n========== STAGE {stage} INFERENCE ==========\n")
    if checkpoint is None:
        raise ValueError(
            "Inference requires --checkpoint"
        )
    test(
        cfg=cfg,
        stage=stage,
        checkpoint=checkpoint
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["preprocess", "train", "test"],
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=[1, 2],
        help="Stage number (1 or 2)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for inference"
    )
    args = parser.parse_args()
    cfg = Config()
    pl.seed_everything(42, workers=True)
    
    if args.mode == "preprocess":
        run_preprocess(cfg)

    elif args.mode == "train":
        # Stage 2 requires prompts from Stage 1
        if args.stage == 2 and cfg.prompts_path is None:
            raise ValueError(
                "Stage 2 requires cfg.prompts_path "
                "(learned prompts from Stage 1)."
            )
        run_train(cfg, args.stage)

    elif args.mode == "test":
        run_test(
            cfg,
            args.stage,
            args.checkpoint
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()