
# Post-Training Sparsity-Aware Quantization

This repository is the official implementation of [Post-Training Sparsity-Aware Quantization](https://arxiv.org/abs/2030.12345). 

## Requirements

Install PyTorch. Specifically, we use version 1.5.1 with CUDA 10.1.
```pytorch
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```
Pick PyTorch 1.5.1 with the appropriate CUDA version from the [official PyTorch website](https://pytorch.org/).  
Then, install the other packages and our custom CUDA package:
```setup
pip install -r requirements.txt
cd cu_gemm_2x48
python ./setup install
```
The ImageNet path, as well as the seeds used to achieve the paper's results, are configured in `Config.py`.  
Throughout this work, we used Ubuntu 18.04, Python 3.6.9, and NVIDIA TITAN V GPU.  

## 8-bit Model Quantization

SPARQ operates on top 8-bit models.
To quantize the models, execute the following command:

```quantize
python ./main.py -a resnet18_imagenet --action QUANTIZE --x_bits 8 --w_bits 8
```
We support the following models: `resnet18_imagenet`, `resnet34_imagenet`, `resnet50_imagenet`, `resnet101_imagenet`, `googlenet_imagenet`, `inception_imagenet`, `densenet_imagenet`.

## SPARQ Evaluation

To evaluate the quantized models,  execute the following:

```eval
python ./main.py -a resnet18_imagenet
                 --action INFERENCE
                 --chkp [PATH^]
                 --x_bits 8 --w_bits 8
                 --eval --round_mode RAW --shift_opt 5 --bit_group 4
```
where `round_mode` is either `RAW` or `ROUND`, `shift_opt` correspond to opt5, opt3, and op2, and `bit_group` correspond to 4, 3, and 2 window bit-width.

[^] `PATH` points to the quantized model and is relative to the `data/results` folder.  
[^^] `shift_opt 5` when paired with `bit_group 3` and `2` correspond to opt6 and opt7, respectively.  
[^^^] In the paper, we used `--batch_size 32` with InceptionV3.

## Pre-trained 2:4 Pruned Models

Models were trained using [this](https://github.com/pytorch/vision/blob/master/references/classification/train.py) PyTorch script and NVIDIA [ASP library](https://github.com/NVIDIA/apex/tree/master/apex/contrib/sparsity).
We used 90 epochs with a learning rate starting from 0.1 and divided by 10 on epochs 30 and 60. Weight decay and momentum are set to 0.0001 and 0.9, respectively.
The pruned models can be downloaded from the links in the table below.

| Model     | Top-1  | Download | 
|-----------|--------|----------|
| ResNet-18 | 69.77% | [Link](https://technionmail-my.sharepoint.com/:u:/g/personal/gilsho_campus_technion_ac_il/EXxRixkzwvtDpQX8MLZw9EcBhjEUgqZUfpv0PJz6pa3ZZg?e=afdmjg) |
| ResNet-50 | 76.16% | [Link](https://technionmail-my.sharepoint.com/:u:/g/personal/gilsho_campus_technion_ac_il/ER1cPJvnxrdJps-PSqJWdx4BjpiSX1_ecMPMmIXl2OBfbA?e=bxddnR) |
| ResNet-101| 77.38% | [Link](https://technionmail-my.sharepoint.com/:u:/g/personal/gilsho_campus_technion_ac_il/EYYvGG7trE1HsD1AzwZFa94BGxqlwBsy0-8V7W5TDICx_A?e=WgGuja) |

To evaluate SPARQ on top of the pruned models, quantize the models as before, and add `--stc` flag to the evaluation command line.
