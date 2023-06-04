---
license: afl-3.0
---

## Suri: Making Large Model Training Efficient for Single-cell RNA-seq data


## Data

You can try pretraining on the PanglaoDB dataset in the 'data' folder. This dataset is the same as that provided in the scBERT paper.

The datasets available from the CELLxGENE website almost are organized as '.h5ad' file formats. It is also easy to use those single-cell RNA-seq dataset for pretraining. 

## Usage

- Pretrain on single-cell RNA-seq data
```
python --data_path "data_path" pretrain.py
```

## Hardware
- GPU: Nvidia A10
- GPU Memory: 24 GB
- CPU: 30 vCPU	
- Memory: 200 GB

## Time cost

The estimated pretraining time of one epoch using about 1,000,000 cells on an NVIDIA A10 is about 4 hours.


## Disclaimer
This project is used for academic research purposes.


## Citations

You can find more information in these citations if you are interested in the technical details.

```bibtex
@article{yang2022scbert,
  title={scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data},
  author={Yang, Fan and Wang, Wenchuan and Wang, Fang and Fang, Yuan and Tang, Duyu and Huang, Junzhou and Lu, Hui and Yao, Jianhua},
  journal={Nature Machine Intelligence},
  volume={4},
  number={10},
  pages={852--866},
  year={2022},
  publisher={Nature Publishing Group UK London}
}
```

```bibtex
@inproceedings{choromanski2020rethinking,
    title   = {Rethinking Attention with Performers},
    author  = {Krzysztof Choromanski and Valerii Likhosherstov and David Dohan and Xingyou Song and Andreea Gane and Tamas Sarlos and Peter Hawkins and Jared Davis and Afroz Mohiuddin and Lukasz Kaiser and David Belanger and Lucy Colwell and Adrian Weller},
    booktitle   = {International Conference on Learning Representations},
    year        = {2021},
}
```

```bibtex
@article{liu2023sophia,
  title={Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training},
  author={Liu, Hong and Li, Zhiyuan and Hall, David and Liang, Percy and Ma, Tengyu},
  journal={arXiv preprint arXiv:2305.14342},
  year={2023}
}
```
