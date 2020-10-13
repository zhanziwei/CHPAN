# CHPAN
Complemental Human Parsing and Attention Guided Network with Multiple Features for Person Re-Identification is implemented in PyTorch

### Prerequisites
* Pytorch 1.1
* cuda 9.0
* python 3.6
* 1GPUs Memory>29G We Recommend Tesla V100


### Datasets
To use our code, firstly you should download ReID dataset (Market1501,DukeMTMC-reID,CUHK03-NP and MSMT17) from [Here](https://pan.baidu.com/s/1G_Ygn68UolKhmiu1eGliLg)(saqs).

Here we use the CUHK03 dataset as an example for description.

"detected" means the bounding boxes are estimated by pedestrian detector

"labeled" means the bounding boxes are labeled by human
```
CUHK03_np
│ 
└───Labeled
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
└───detected
│   └───bounding_box_test
│   │   │   0003_c1_21.jpg
│   │   │   0003_c1_23.jpg
│   │   │   ...
│   └───bounding_box_train
│   │   │   0001_c1_1.png
│   │   │   0001_c1_2.png
│   │   │   ...
│   └───query
│   │   │   0003_c1_22.png
│   │   │   0003_c2_27.png
│   │   │   ...
```

### Train
First ,you must run the perpare.py to get 'pytorch' folder.

You need prepare your dataset-dir in the 'prepare.py' flie
```
download_path = '.../dataset/CUHK'
```
In our train.py,we give you some options,as follows:
```
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='', type=str, help='output model name')
parser.add_argument('--data_dir',default='',type=str, help='training dir path')
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--erasing_p', default=0, type=float, help='Random Erasing probability, in [0,1]')
parser.add_argument('--warm_epoch', default=10, type=int, help='the first K epoch that needs warm up')
parser.add_argument('--epochnum', default=150, type=int, help='please to select the epoch num')
parser.add_argument('--base_lr', default=0.01, type=float, help='the base_learning rate')
parser.add_argument('--tripletmargin', default=0.3, type=float, help='the tripletmargin')
parser.add_argument('--warmup_begin_lr', default=3e-4, type=float, help='warmup learning rate')
parser.add_argument('--factor', default=0.5, type=float, help='the learning rate decracy')
parser.add_argument('--testing', action='store_true', help='import testing features')
parser.add_argument('--featuresize', default=48, type=int, help='the stage4s feature map size')
opt = parser.parse_args()

```

### Usage
```
python3 train.py --gpu_ids .. --name .. --data_dir ../cuhk03-np/labeled/pytorch --batchsize --erasing_p  --warm_epoch  --epochnum  --base_lr  --tripletmargin  --warmup_begin_lr  --factor   --featuresize 

python3 train.py --gpu_ids .. --name .. --data_dir ../cuhk03-np/detected/pytorch --batchsize --erasing_p  --warm_epoch  --epochnum  --base_lr  --tripletmargin --warmup_begin_lr  --factor   --featuresize
```

### Test
```
parser = argparse.ArgumentParser(description='testing')
parser.add_argument('--gpu_ids',default='1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--name',default='', type=str, help='output model name')
parser.add_argument('--test_dir',default='',type=str, help='training dir path')
parser.add_argument('--train_all', action='store_true', help='use all training data' )
parser.add_argument('--batchsize', default=64, type=int, help='batchsize')
parser.add_argument('--featurereid', default=6144, type=int, help='batchsize')
parser.add_argument('--epochnum', default='last', type=str, help='please to select the epoch num')
parser.add_argument('--testing', action='store_true', help='import testing features')
parser.add_argument('--featuresize', default=48, type=int, help='the stage4s feature map size')
opt = parser.parse_args()
```

```
python3 test.py --gpu_ids ... --data_dir ../cuhk03-np/labeled/pytorch --name ... --batchsize 64 --epochnum last --train_all --attentionmodel --testing
python3 test.py --gpu_ids ... --data_dir ../cuhk03-np/detected/pytorch  --name ... --batchsize 64 --epochnum last --train_all --attentionmodel --testing
```

### Evaluation

```
# Using CPU
python3 evaluate.py
# If you want to attain results quickly you can do:
python3 evaluate_gpu.py
# Also, you can use the reranking
python3 evaluate_rerank.py
```
| Datasets | TOP@1 | TOP@5 | TOP@10 |mAP|
| :------: | :------: | :------: | :------: | :------: |
| CUHK_Detected|   |  |  | |
| CUHK_Labeled |   |   |  | |
| DukeMTMC-reID| 89.72   | 94.08  |  95.42 | 78.95|

### Visualization
```
#you should choose the dateset dir
parser.add_argument('--test_dir',default='...\DukeMTMC-reID\pytorch',type=str, help='./test_data')
```
```
python3 demo.py --test_dir ../cuhk03-np/labeled/pytorch 
python3 demo.py --test_dir ../cuhk03-np/detected/pytorch 
```
