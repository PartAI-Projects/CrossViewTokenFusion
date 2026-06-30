from typing import Tuple
from dataclasses import dataclass, field

# -------------------------
# Data config
# -------------------------
@dataclass
class DataConfig:
    data_name: str = "vindr"  # vindr | cmmd 
    data_dir: str = "/path/to/dataset/root"  # vindr-mammo or TheChineseMammographyDatabase/CMMD
    prep_dir: str = "./data/vindr_prep"  # Path to save/load preprocessed files
    task: str = "multiclass"  # binary | multiclass (multiclass supported only for VinDr-Mammo)
    batch_size: int = 8
    num_workers: int = 4
    image_size: Tuple[int, int] = (448, 448)

# -------------------------
# Model config
# -------------------------
@dataclass
class ModelConfig:
    model_id: str = "google/medsiglip-448"
    prompt_depth: int = 12
    shallow_prompt_length: int = 1
    fusion_layers: list[int] = field(default_factory=lambda: [23, 26])
    fusion_heads: int = 4
    fusion_dropout: float = 0.1

# -------------------------
# Train config
# -------------------------
@dataclass
class TrainConfig:
    cls_lr: float = 1e-3
    prompt_lr: float = 1e-3
    fusion_lr: float = 1e-3
    vision_head_lr: float = 1e-3
    epochs: int = 15
    weight_decay: float = 0.01
    warmup_steps: int = 100
    use_augmentation: bool = False

# -------------------------
# Main config
# -------------------------
@dataclass
class Config:
    output_dir: str = "./outputs"
    prompts_path: str = "learned_prompts_stage1.pt"

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
