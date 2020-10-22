# A Pytorch-Lightning Implementation of Transformer Network

This repository includes pytorch-lightning implementations of ["Attention is All You Need"](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf) (Vaswani et al., NIPS 2017) and 
["Weighted Transformer Network for Machine Translation"](https://arxiv.org/pdf/1711.02132.pdf) (Ahmed et al., arXiv 2017)

## Requirements

- python >= 3.5
- torch >= 1.3.0
- pytorch-lighting >= 0.9.0
- torchtext >= 0.4.0
- spacy >= 2.2.2
- dill

## Usage

1. Generate the `m30k_deen_shr.pkl` file followed the repo 
[jadore801120/attention-is-all-you-need](https://github.com/jadore801120/attention-is-all-you-need-pytorch#usage)

2. Run the train scripts.
    ```bash
    python run configs/transformer.py
    ```

3. Show the training log.
    ```bash
    tensorboard --logdir work_dirs/logs/Transformer/0.1.0/
    ```

## Reference
**Paper**
- Vaswani et al., "Attention is All You Need", NIPS 2017
- Ahmed et al., "Weighted Transformer Network for Machine Translation", Arxiv 2017

**Code**
- [jadore801120/attention-is-all-you-need](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [jayparks/transformer](https://github.com/jayparks/transformer.git)
