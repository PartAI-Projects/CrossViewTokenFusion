# Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification
[![arXiv](https://img.shields.io/badge/arXiv-2607.06309-b31b1b.svg)](https://arxiv.org/abs/2607.06309)      
This repository contains the official implementation of our proposed two-stage framework for dual-view mammography classification based on MedSigLIP vision model.


<p align="center">
  <img width="940" alt="Overview of the proposed framework" src="https://github.com/user-attachments/assets/f2ab388b-129b-413f-acf7-5f31520a67e3" />
</p>


## Dataset

The following public datasets are used in this work:

- [VinDr-Mammo](https://physionet.org/content/vindr-mammo/1.0.0/)
- [CMMD (Chinese Mammography Database)](https://www.cancerimagingarchive.net/collection/cmmd/)

## Installation

```bash
git clone https://github.com/PartAI-Projects/CrossViewTokenFusion.git
cd CrossViewTokenFusion

# Install PyTorch and torchvision first:
# https://pytorch.org/get-started/locally/

pip install -r requirements.txt
```

## Configuration

Update `configs/config.py` to specify the dataset root directory and adjust other settings (e.g., preprocessing paths, task type, and training hyperparameters) as needed.

## Usage

The pipeline supports three modes: preprocessing, training, and testing.

```bash
# Preprocessing
python run.py --mode preprocess

# Training
# Stage 1: Deep prompt learning 
python run.py --mode train --stage 1

# Stage 2: Cross-view token-based fusion 
python run.py --mode train --stage 2

# Testing / Inference
python run.py --mode test --stage 2 --checkpoint PATH_TO_CKPT
```

## Result
The reported results are obtained on the VinDr-Mammo binary classification task using the official VinDr-Mammo test set. For the binary classification task, BI-RADS 2 is used as the **suspicious benign** class, while BI-RADS 4 and BI-RADS 5 are grouped as the **suspicious malignant** class.
  
| Method          | # Fusion Blocks | F1-Score (%) | AUC-ROC  |
|:------:|:---------------:|:------------:|:-------:|
| [DIVF](https://doi.org/10.48550/arXiv.2309.03506)        | –               | 75.98        | 0.7486  |
| **Proposed Method** | 2               | **77.88**    | **0.8593** |


To reproduce the reported results, after preprocessing, update the following settings in `configs/config.py`:

```python
prompt_depth = 8
shallow_prompt_length = 2
fusion_layers = [12, 23]

prompts_path = "stage1_best_prompts_vindr_bin.pt"
```

Then run inference using the released Stage 2 checkpoint on [Hugging Face](https://huggingface.co/aysangh/medsiglip-fusion-vindr-bin): 

```bash
python run.py --mode test --stage 2 --checkpoint path/to/stage2.ckpt
```

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{pirsoltan2026token,
  title={Token-Based Dual-view Fusion and Adaptation of Large Vision Models for Breast Cancer Classification},
  author={Pirsoltan, Aysan Ghayouri and Babakordi, Shima and Mohammadi, Mohammad Reza},
  journal={arXiv preprint arXiv:2607.06309},
  year={2026}
}
```
