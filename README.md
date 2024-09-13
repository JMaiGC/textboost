# TextBoost: Towards One-Shot Customization of Text-to-Image Models via Fine-tuning Text Encoder

[![arXiv](https://img.shields.io/badge/arXiv-2409.08248-B31B1B.svg)](https://arxiv.org/abs/2409.08248)

Abstract: *Recent breakthroughs in text-to-image models have opened up promising research avenues in personalized image generation, enabling users to create diverse images of a specific subject using natural language prompts. However, existing methods often suffer from performance degradation when given only a single reference image. They tend to overfit the input, producing highly similar outputs regardless of the text prompt. This paper addresses the challenge of one-shot personalization by mitigating overfitting, enabling the creation of controllable images through text prompts. Specifically, we propose a selective fine-tuning strategy that focuses on the text encoder. Furthermore, we introduce three key techniques to enhance personalization performance: (1) augmentation tokens to encourage feature disentanglement and alleviate overfitting, (2) a knowledge-preservation loss to reduce language drift and promote generalizability across diverse prompts, and (3) SNR-weighted sampling for efficient training. Extensive experiments demonstrate that our approach efficiently generates high-quality, diverse images using only a single reference image while significantly reducing memory and storage requirements.*


## Installation

Our code has been tested on `python3.10` with `NVIDIA A6000 GPU`. However, it should work with the other recent Python versions and NVIDIA GPUs.

### Installing Python Packages

We recommend using a Python virtual environment or anaconda for managing dependencies. You can install the required packages using one of the following methods:

#### Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Using `conda`:

```bash
conda env create -f environment.yml
conda activate textboost
```

For the exact package versions we used, please refer to [requirements.txt](requirements.txt) file.



## Training

```bash
CUDA_VISIBLE_DEVICES=3 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nproc-per-node=1 train_textboost.py \
--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 \
--instance_data_dir data/dreambooth/dog  \
--output_dir=output/tb/dog \
--instance_token '<dog> dog' \
--class_token 'dog' \
--validation_prompt 'a <dog> dog in the jungle' \
--validation_steps=50 \
--placeholder_token '<dog>' \
--initializer_token 'dog' \
--learning_rate=5e-5 \
--emb_learning_rate=1e-3 \
--train_batch_size=8 \
--max_train_steps=250 \
--checkpointing_steps=50 \
--num_samples=1 \
--augment=paug \
--lora_rank=4 \
--augment_inversion
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=6 python -m evaluation.dreambooth output/tb-sd1.5-n1 --token-format '<INSTANCE>'
```

## Citation

```bibtex
@article{park2024textboost,
    title   = {TextBoost: Towards One-Shot Customization of Text-to-Image Models},
    author  = {Park, NaHyeon and Kim, Kunhee and Shim, Hyunjung},
    journal = {arXiv preprint},
    year    = {2024},
    eprint  = {arXiv:24xx.yyyyy}
}
```

## License

All materials in this repository are available under the [MIT License](LICENSE).
