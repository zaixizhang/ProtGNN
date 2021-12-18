# ProtGNN
Official implementation of AAAI'22 paper "ProtGNN: Towards Self-Explaining Graph Neural Networks" (https://arxiv.org/abs/2112.00911)

## Requirements
```
pytorch                   1.8.0             
torch-geometric           2.0.2
```
## Usage
* Download the required [dataset](https://mailustceducn-my.sharepoint.com/:u:/g/personal/yhy12138_mail_ustc_edu_cn/ET69UPOa9jxAlob03sWzJ50BeXM-lMjoKh52h6aFc8E8Jw?e=lglJcP) to `./datasets`
The hyper-parameters can be set in ./Configures.py

You can run ProtGNN by
```
python -m models.train_gnns
```

## Cite

If you find this repo to be useful, please cite our paper. Thank you.

```
@article{zhang2021protgnn,
  title={ProtGNN: Towards Self-Explaining Graph Neural Networks},
  author={Zhang, Zaixi and Liu, Qi and Wang, Hao and Lu, Chengqiang and Lee, Cheekong},
  journal={arXiv preprint arXiv:2112.00911},
  year={2021}
}
```
