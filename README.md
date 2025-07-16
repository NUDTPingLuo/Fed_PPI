# Federated Prediction-Powered Inference from Decentralized Data

This repository contains the official implementation of the paper:

**Federated Prediction-Powered Inference from Decentralized Data**  
[Ping Luo], [Deng Xiaoge], [Wen, Ziqing] [Sun Tao], [Li, Dongsheng]  
arXiv: [https://arxiv.org/abs/2409.01730](https://arxiv.org/abs/2409.01730)

This work extends the **Prediction-Powered Inference (PPI)** framework to the federated learning setting, enabling statistically valid inference under data decentralization and distribution shift. We propose new protocols and algorithms that preserve statistical efficiency without centralizing the data.

## Adaptation

This codebase is adapted from [ppi_py](https://github.com/aangelopoulos/ppi_py), the official implementation of  
**"Prediction-powered inference"** by Anastasios N. Angelopoulos et al. published in *Science* [https://www.science.org/doi/abs/10.1126/science.adi6000].

We extend the methodology to federated settings, introducing key algorithmic and theoretical innovations tailored to decentralized data.

## Features

- Federated conformal inference under distribution shift  
- Support for both IID and non-IID client partitions  
- Adaptive prediction-powered protocols with local calibration  
- Confidence intervals that match centralized baselines  
- Full reproducibility with script-based simulation setup

## Reference

If you find this repository useful, please consider citing:

```bibtex
@article{luo2024federated,
  title={Federated Prediction-Powered Inference from Decentralized Data},
  author={Luo, Ping and Deng, Xiaoge and Wen, Ziqing and Sun, Tao and Li, Dongsheng},
  journal={arXiv preprint arXiv:2409.01730},
  year={2024}
}
