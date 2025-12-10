# HTR-ConvText

<div align="center"> <img src="image/architecture.png" alt="HTR-ConvText Architecture" width="800"/> </div>

<p align="center"> <a href="https://huggingface.co/DAIR-Group/HTR-ConvText"> <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue"> </a> <a href="https://github.com/DAIR-Group/HTR-ConvText/blob/main/LICENSE"> <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-green"> </a> <a href="https://arxiv.org/abs/2512.05021"> <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2512.05021-b31b1b.svg"> </a> <a href="https://github.com/DAIR-Group/HTR-ConvText"> <img alt="GitHub" src="https://img.shields.io/badge/GitHub-Repo-181717.svg"> </a> </p>

## Highlights

HTR-ConvText is a novel hybrid architecture for Handwritten Text Recognition (HTR) that effectively balances local feature extraction with global contextual modeling. Designed to overcome the limitations of standard CTC-based decoding and data-hungry Transformers, HTR-ConvText delivers state-of-the-art performance with the following key features:

- **Hybrid CNN-ViT Architecture**: Seamlessly integrates a ResNet backbone with MobileViT blocks (MVP) and Conditional Positional Encoding, enabling the model to capture fine-grained stroke details while maintaining global spatial awareness.
- **Hierarchical ConvText Encoder**: A U-Net-like encoder structure that interleaves Multi-Head Self-Attention with Depthwise Convolutions. This design efficiently models both long-range dependencies and local structural patterns.
- **Textual Context Module (TCM)**: An innovative training-only auxiliary module that injects bidirectional linguistic priors into the visual encoder. This mitigates the conditional independence weakness of CTC decoding without adding any latency during inference.
- **State-of-the-Art Performance**: Outperforms existing methods on major benchmarks including IAM (English), READ2016 (German), LAM (Italian), and HANDS-VNOnDB (Vietnamese), specifically excelling in low-resource scenarios and complex diacritics.

## Model Overview

HTR-ConvText configurations and specifications:

| Feature             | Specification                                       |
| ------------------- | --------------------------------------------------- |
| Architecture Type   | Hybrid CNN + Vision Transformer (Encoder-Only)      |
| Parameters          | ~65.9M                                              |
| Backbone            | ResNet-18 + MobileViT w/ Positional Encoding (MVP)  |
| Encoder Layers      | 8 ConvText Blocks (Hierarchical)                    |
| Attention Heads     | 8                                                   |
| Embedding Dimension | 512                                                 |
| Image Input Size    | 512×64                                              |
| Inference Strategy  | Standard CTC Decoding (TCM is removed at inference) |

For more details, including ablation studies and theoretical proofs, please refer to our Technical Report.

## Performance

We evaluated HTR-ConvText across four diverse datasets. The model achieves new SOTA results with the lowest Character Error Rate (CER) and Word Error Rate (WER) without requiring massive synthetic pre-training.

| Dataset   | Language    | Ours CER (%) | HTR-VT | OrigamiNet | TrOCR | CRNN  |
|-----------|-------------|--------------|--------|------------|-------|-------|
| IAM       | English     | 4.0          | 4.7    | 4.8        | 7.3   | 7.8   |
| LAM       | Italian     | 2.7          | 2.8    | 3.0        | 3.6   | 3.8   |
| READ2016  | German      | 3.6          | 3.9    | -          | -     | 4.7   |
| VNOnDB    | Vietnamese  | 3.45         | 4.26   | 7.6        | -     | 10.53 |

## Quickstart

### Instalation

1. **Clone the repository**
   ```cmd
   git clone https://github.com/0xk0ry/HTR-ConvText.git
   cd HTR-ConvText
   ```
2. **Create and activate a Python 3.9+ Conda environment**
   ```cmd
   conda create -n htr-convtext python=3.9 -y
   conda activate htr-convtext
   ```
3. **Install PyTorch** using the wheel that matches your CUDA driver (swap the index for CPU-only builds):
   ```cmd
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   ```
4. **Install the remaining project requirements** (everything except PyTorch, which you already picked in step 3).
   ```cmd
   pip install -r requirements.txt
   ```

The code was tested on Python 3.9 and PyTorch 2.9.1.

### Data Preparation

We provide split files (train.ln, val.ln, test.ln) for IAM, READ2016, LAM, and VNOnDB under data/. Organize your data as follows:

```
./data/iam/
├── train.ln
├── val.ln
├── test.ln
└── lines
      ├── a01-000u-00.png
      ├── a01-000u-00.txt
      └── ...
```

### Training

We provide comprehensive scripts in the ./run/ directory. To train on the IAM dataset with the Textual Context Module (TCM) enabled:

```
# Using the provided script
bash run/iam.sh

# OR running directly via Python
python train.py \
    --use-wandb \
    --dataset iam \
    --tcm-enable \
    --exp-name "htr-convtext-iam" \
    --img-size 512 64 \
    --train-bs 32 \
    --val-bs 8 \
    --data-path /path/to/iam/lines/ \
    --train-data-list data/iam/train.ln \
    --val-data-list data/iam/val.ln \
    --test-data-list data/iam/test.ln \
    --nb-cls 80
```

### Inference / Evaluation

To evaluate a pre-trained checkpoint on the test set:

```
python test.py \
    --resume ./checkpoints/best_CER.pth \
    --dataset iam \
    --img-size 512 64 \
    --data-path /path/to/iam/lines/ \
    --test-data-list data/iam/test.ln \
    --nb-cls 80
```

## Citation

If you find our work helpful, please cite our paper:

```
@misc{truc2025htrconvtex,
      title={HTR-ConvText: Leveraging Convolution and Textual Information for Handwritten Text Recognition},
      author={Pham Thach Thanh Truc and Dang Hoai Nam and Huynh Tong Dang Khoa and Vo Nguyen Le Duy},
      year={2025},
      eprint={2512.05021},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.05021},
}
```

## Acknowledgement

This project is inspired by and adapted from [HTR-VT](https://github.com/Intellindust-AI-Lab/HTR-VT). We gratefully acknowledge the authors for their open-source contributions.
