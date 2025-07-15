# Multi-GPU Fine-Tuning with Deepspeed

This project demonstrates how to fine-tune a large language model using multiple GPUs.
Includes Deepspeed configuration, logs, and example training script.


# Multi-GPU Fine-Tuning with Deepspeed + Transformers

This demo fine-tunes a BERT model on the IMDB dataset using Hugging Face Trainer, Deepspeed, and FP16 mixed precision.

## How to Run
```bash
deepspeed train.py --deepspeed config/deepspeed_config.json