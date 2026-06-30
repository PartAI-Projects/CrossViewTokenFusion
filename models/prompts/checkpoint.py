from pathlib import Path
import torch
from pytorch_lightning.callbacks import Callback

from .deep_prompt import DeepPrompt
 

def load_stage1_prompts(cfg, hidden_size, device="cuda"):
    prompts_path = Path(cfg.output_dir) / Path(cfg.prompts_path)
    print("\n" + "=" * 70)
    print(f"Loading Stage 1 prompts from:")
    print(prompts_path)
    print("=" * 70)
    checkpoint = torch.load(prompts_path)
    deep_prompt = DeepPrompt(
        cfg,
        hidden_size,
    )
    deep_prompt.shallow_prompt.data.copy_(
        checkpoint["shallow_prompt"].to(device)
    )
    for i, prompt in enumerate(checkpoint["deep_prompts"]):
        deep_prompt.deep_prompts[i].data.copy_(
            prompt.to(device)
        )
    # freeze prompts
    for p in deep_prompt.parameters():
        p.requires_grad = False
    print(
        f"✓ Loaded prompts "
        f"(depth={cfg.model.prompt_depth})"
    )
    print("✓ Prompts frozen")
    return deep_prompt, checkpoint


class SavePrompts(Callback):
    def __init__(self, cfg):
        super().__init__()
        self.output_dir = Path(cfg.output_dir)
        self.prompts_path = Path(cfg.prompts_path)
        self.best_f1 = -float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        current_f1 = trainer.callback_metrics.get("val_f1")
        if current_f1 is None:
            return
        current_f1 = current_f1.item()
        if current_f1 > self.best_f1:
            self.best_f1 = current_f1
            self._save_prompts(
                trainer=trainer,
                pl_module=pl_module,
            )
            print(
                f"\n✓ New best F1: {current_f1:.4f} "
                f"(epoch {trainer.current_epoch})"
            )

    def _save_prompts(self, trainer, pl_module):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        dp = pl_module.deep_prompt
        checkpoint = {
            # prompts
            "shallow_prompt":
                dp.shallow_prompt.detach().cpu(),

            "deep_prompts":
                [p.detach().cpu()
                 for p in dp.deep_prompts],

            # metadata
            "prompt_depth": dp.prompt_depth,
            "hidden_size": dp.hidden_size,

            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }
        save_path = (
            self.output_dir /
            self.prompts_path
        )
        torch.save(checkpoint, save_path)
        print(f"✓ Prompts saved to: {save_path}")