# GB-CosFace: Rethinking Softmax-based Face Recognition from the Perspective of Open Set Classification

------
This is the official pytorch implementation of the paper "GB-CosFace: Rethinking Softmax-based Face Recognition from the Perspective of Open Set Classification".([link](https://arxiv.org/abs/2111.11186))

## Installation
a.Create a conda virtual environment and activate it.
```shell
conda create -n gb-cosface python=3.7 -y
conda activate gb-cosface
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```
Note: The CUDA version of the installed pytorch needs to match the runtime CUDA version.

c. 

```shell
pip install -r requirements.txt
```

## Datasets

We use MS1MV2 as the training set, and use several popular benchmarks as the validation set, including LFW, CFP-FP, CPLFW, AgeDB-30, and CALFW. Our training and validation data comes from [Insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_). You can also download the data from this [link](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/GB-CosFace/datasets/faces_emore.zip).

We use [IJB-B](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/GB-CosFace/datasets/IJBB.zip) and [IJB-C](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/GB-CosFace/datasets/IJBC.zip) as the testing sets. Please apply for permissions from [NIST](https://www.nist.gov/programs-projects/face-challenges) before your usage.

## Training

### GB-CosFace

a. Edit the file "config.py", edit the "rec" and "val_root" paths to your dataset path.

b. Run the following command.

``` shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1239 train.py --backbone_type iresnet100 --head_type BaseHead --loss_type GBCosFace --batchsize 64 --output [your saving dir] --eval_steps 4000
```

## Testing

We release the GB-CosFace iresnet100 [model](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/GB-CosFace/models/backbone_paper.pth) in the original paper main text and the GB-MagFace [model](http://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/GB-CosFace/models/backbone_paper.pth) in the appendix.

### Testing on IJBB
``` shell
python eval_ijb.py --model-prefix [your backbone path] --image-path [your IJBB root] --result-dir [your save path] --batch-size 512 --backbone_type iresnet100 --target IJBC
```

### Testing on IJBC
``` shell
python eval_ijb.py --model-prefix [your backbone path] --image-path [your IJBC root] --result-dir [your save path] --batch-size 512 --backbone_type iresnet100 --target IJBC
```

# Acknowledgements
This repo is based on [FaceXZoo](https://github.com/JDAI-CV/FaceX-Zoo), [insightface](https://github.com/deepinsight/insightface), and [MagFace](https://github.com/IrvingMeng/MagFace). We thank the authors a lot for their valuable efforts.
