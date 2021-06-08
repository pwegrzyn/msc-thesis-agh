# Mode Connectivity and Fast Geometric Ensembling - Further Investigation

This repository contains a PyTorch implementation of the curve-finding and Fast Geometric Ensembling (FGE) procedures from the paper

[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)

by Timur Garipov, Pavel Izmailov, Dmitrii Podoprikhin, Dmitry Vetrov and Andrew Gordon Wilson (NIPS 2018, Spotlight).

Later work on the baseline code has been made by Patryk Wegrzyn as part of Master's Thesis (under BSD 2-Clause License permission).


# Tested environment and dependencies
* [PyTorch](http://pytorch.org/) - tested on Python 3.8, CUDA 10.1, PyTorch 1.7 (later switched to Py3.8, CUDA 11.0.2, cuDNN 8.1.0, PyTorch 1.7.1)

## Important info regarding PyTorch version
The initial code was probably developed under a version of Python that was below 3.7, since the name ```async``` was used as an identifier and since Py3.7
it has been made into a keyword, so trying to run the raw code in Py3.7+ resulted in syntax error.
The ```async``` identifier was used as a kwarg in torch and has been later renamed to ```non_blocking```, so I also made this change here.

Additionally, currently there's a bug in Windows implementation of some of the NumPy dependencies, so using the newest version of NumPy doesn't work.
For now, it's recommended to use: ```pip install numpy==1.19.3```. More info here: https://stackoverflow.com/questions/64729944/runtimeerror-the-current-numpy-installation-fails-to-pass-a-sanity-check-due-to

Also, Torchvision's datasets classes API has changed a bit in the newest versions, so some changes had to be made in the data.py file in order to accomodate the changes
(see https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py for more infor about changed class attributes like ```train_labels, test_labels, train_data, test_data```).

Next thing: the initial version of this code had problems with running code in a main scrip without using the ```if __name__ == "__main__":``` idiom,
and this causes issues with Torch's Multiprocessing Loaders - to fix this all main scripts should be changed to use this idiom.

Lastly, the initial version of the code has some problems with GPU memory management. So for example if you have a 4GB GPU, you will need to use a smaller 
batch_size (probably 32 or less) to avoid getting an OOM error.

## Preparing to run

(Better to run all of this in an admin command line) First, create a venv and activate it
```
python -m venv env
.\env\Scripts\activate
```
Then you can install the requirements.

To deactivate the environment
```
deactivate
```

# Usage

The code in this repository implements both the curve-finding procedure and Fast Geometric Ensembling (FGE), with examples on the CIFAR-10 and CIFAR-100 datasets.

## Curve Finding


### Training the endpoints 

To run the curve-finding procedure, you first need to train the two networks that will serve as the end-points of the curve. You can train the endpoints using the following command

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 [--use_test]
```

Parameters:

* ```DIR``` &mdash; path to training directory where checkpoints will be stored
* ```DATASET``` &mdash; dataset name [CIFAR10/CIFAR100] (default: CIFAR10)
* ```PATH``` &mdash; path to the data directory
* ```TRANSFORM``` &mdash; type of data transformation [VGG/ResNet] (default: VGG)
* ```MODEL``` &mdash; DNN model name:
    - VGG16/VGG16BN/VGG19/VGG19BN 
    - PreResNet110/PreResNet164
    - WideResNet28x10
* ```EPOCHS``` &mdash; number of training epochs (default: 200)
* ```LR_INIT``` &mdash; initial learning rate (default: 0.1)
* ```WD``` &mdash; weight decay (default: 1e-4)

Use the `--use_test` flag if you want to use the test set instead of validation set (formed from the last 5000 training objects) to evaluate performance.

For example, use the following commands to train VGG16, PreResNet or Wide ResNet:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=VGG16 --epochs=200 --lr=0.05 --wd=5e-4 --use_test --transform=VGG
#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH>  --model=[PreResNet110 or PreResNet164] --epochs=150  --lr=0.1 --wd=3e-4 --use_test --transform=ResNet
#WideResNet28x10 
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --data_path=<PATH> --model=WideResNet28x10 --epochs=200 --lr=0.1 --wd=5e-4 --use_test --transform=ResNet
```

### Training the curves

Once you have two checkpoints to use as the endpoints you can train the curve connecting them using the following comand.

```bash
python3 train.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr=<LR_INIT> \
                 --wd=<WD> \
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --init_start=<CKPT1> \ 
                 --init_end=<CKPT2> \
                 [--fix_start] \
                 [--fix_end] \
                 [--use_test]
```

Parameters:

* ```CURVE``` &mdash; desired curve parametrization [Bezier|PolyChain] 
* ```N_BENDS``` &mdash; number of bends in the curve (default: 3)
* ```CKPT1, CKPT2``` &mdash; paths to the checkpoints to use as the endpoints of the curve

Use the flags `--fix_end --fix_start` if you want to fix the positions of the endpoints; otherwise the endpoints will be updated during training. See the section on [training the endpoints](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-endpoints)  for the description of the other parameters.

For example, use the following commands to train VGG16, PreResNet or Wide ResNet:
```bash
#VGG16
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=VGG --data_path=<PATH> --model=VGG16 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=600 --lr=0.015 --wd=5e-4

#PreResNet
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=PreResNet164 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=3e-4

#WideResNet28x10
python3 train.py --dir=<DIR> --dataset=[CIFAR10 or CIFAR100] --use_test --transform=ResNet --data_path=<PATH> --model=WideResNet28x10 --curve=[Bezier|PolyChain] --num_bends=3  --init_start=<CKPT1> --init_end=<CKPT2> --fix_start --fix_end --epochs=200 --lr=0.03 --wd=5e-4
```

### Evaluating the curves

To evaluate the found curves, you can use the following command
```bash
python3 eval_curve.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM>
                 --model=<MODEL> \
                 --wd=<WD> \
                 --curve=<CURVE>[Bezier|PolyChain] \
                 --num_bends=<N_BENDS> \
                 --ckpt=<CKPT> \ 
                 --num_points=<NUM_POINTS> \
                 [--use_test]
```
Parameters
* ```CKPT``` &mdash; path to the checkpoint saved by `train.py`
* ```NUM_POINTS``` &mdash; number of points along the curve to use for evaluation (default: 61)

See the sections on [training the endpoints](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-endpoints) and [training the curves](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-curves) for the description of other parameters.

`eval_curve.py` outputs the statistics on train and test loss and error along the curve. It also saves a `.npz` file containing more detailed statistics at `<DIR>`.

#### CIFAR-100


In the table below we report the minimum and maximum train loss and test error (%) for the networks used as the endpoints and along the curves found by our method on CIFAR-100. 

| DNN (Curve)                |Min Train Loss|Max Train Loss| Min Test Error   | Max Test Error  |
| ---------------------------|:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (Endpoints)          | 0.89         | 0.89         | 27.5             | 27.5            |
| VGG16 (Bezier)             | 0.48         | 0.89         | 27.4             | 30.1            |
| VGG16 (Poly)               | 0.59         | 1.05         | 27.1             | 30.8            |
|                            |              |              |                  |                 |
| PreResNet164 (Endpoints)   | 0.49         | 0.49         | 21.6             | 21.7            |
| PreResNet164 (Bezier)      | 0.26         | 0.49         | 21.3             | 23.4            |
| PreResNet164 (Poly)        | 0.30         | 0.49         | 21.4             | 23.6            |
|                            |              |              |                  |                 |
| WideResNet28x10 (Endpoints)| 0.20         | 0.21         | 18.6             | 18.9            |
| WideResNet28x10 (Bezier)   | 0.11         | 0.21         | 18.3             | 19.2            |
| WideResNet28x10 (Poly)     | 0.13         | 0.21         | 18.4             | 19.0            |

Below we show the train loss and test accuracy along the curves connecting two PreResNet164 networks trained with our method on CIFAR100.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/47621112-45da0d80-dac9-11e8-9e00-12f53fb4844a.png" width=800>
</p>

#### CIFAR-10

In the table below we report the minimum and maximum train loss and test error (%) for the networks used as the endpoints and along the curves found by our method on CIFAR-10. 

| DNN (Curve)               |Min Train Loss|Max Train Loss| Min Test Error   | Max Test Error  |
| ------------------------- |:------------:|:------------:|:----------------:|:---------------:|
| VGG16 (Single)            | 0.24         | 0.24         | 6.79             | 6.94            |
| VGG16 (Bezier)            | 0.14         | 0.24         | 6.79             | 7.75            |
| VGG16 (Poly)              | 0.16         | 0.27         | 6.79             | 8.08            |
|                           |              |              |                  |                 |
| PreResNet164 (Single)     | 0.18         | 0.18         | 4.76             | 4.75            |
| PreResNet164 (Bezier)     | 0.09         | 0.18         | 4.45             | 4.97            |
| PreResNet164 (Poly)       | 0.11         | 0.18         | 4.39             | 5.13            |
|                           |              |              |                  |                 |
| WideResNet28x10 (Single)  | 0.08         | 0.09         | 3.69             | 3.73            |
| WideResNet28x10 (Bezier)  | 0.05         | 0.09         | 3.49             | 3.88            |
| WideResNet28x10 (Poly)    | 0.05         | 0.10         | 3.53             | 4.29            |


## Fast Geometric Ensembling (FGE)

In order to run FGE you need to pre-train the network to initialize the procedure. To do so follow the instructions in the section on [training the endpoints](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-endpoints). Then, you can run FGE with the following command

```bash
python3 fge.py --dir=<DIR> \
                 --dataset=<DATASET> \
                 --data_path=<PATH> \
                 --transform=<TRANSFORM> \
                 --model=<MODEL> \
                 --epochs=<EPOCHS> \
                 --lr_init=<LR_INIT> \
                 --wd=<WD> \
                 --ckpt=<CKPT> \
                 --lr_1=<LR1> \
                 --lr_2=<LR2> \
                 --cycle=<CYCLE> \
                 [--use_test]
```
Parameters:

* ```CKPT``` path to the checkpoint saved by `train.py`
* ```LR1, LR2``` the minimum and maximum learning rates in the cycle
* ```CYCLE``` cycle length in epochs (default:4)

See the section on [training the endpoints](https://github.com/izmailovpavel/curves-dnn-loss-surfaces/blob/master/README.md#training-the-endpoints)  for the description of the other parameters.

In the Figure below we show the learning rate (top), test error (middle) and distance from the initial value `<CKPT>` as a function of iteration for FGE with PreResNet164 on CIFAR100. Circles indicate when we save models for ensembling.

<p align="center">
  <img src="https://user-images.githubusercontent.com/14368801/47262174-5f6acc00-d4af-11e8-954f-dfef255ad3ae.png" width=500>
</p>



## CIFAR-100

To reproduce the results from the paper run:
```bash
#VGG16
python3 train.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR100 --use_test --transform=VGG --model=VGG16 --epochs=200 --wd=5e-4 --lr=0.05 --save_freq=40
python3 fge.py --dir=<DIR> --ckpt=<DIR>/checkpoint-160.pt --data_path=<PATH> --dataset=CIFAR100 --use_test --transform=VGG --model=VGG16 --epochs=40 --wd=5e-4 --lr_1=1e-2 --lr_2=1e-2 --cycle=2

#PreResNet
python3 train.py --dir=<DIR>  --data_path=<PATH> --dataset=CIFAR100 --use_test --transform=ResNet --model=PreResNet164 --epochs=200 --wd=3e-4 --lr=0.1 --save_freq=40
python3 fge.py --dir=<DIR> --ckpt=<DIR>/checkpoint-160.pt --data_path=<PATH> --dataset=CIFAR100 --use_test --transform=ResNet --model=PreResNet164 --epochs=40 --wd=3e-4 --lr_1=0.05 --lr_2=0.01 --cycle=2

#WideResNet28x10
python3 train.py --dir=<DIR> --data_path=<PATH> --dataset=CIFAR100 --use_test --transform=ResNet --model=WideResNet28x10 --epochs=40 --wd=5e-4 --lr=0.1 --save_freq=40
python3 fge.py --dir=<DIR> --ckpt=<DIR>/checkpoint-160.pt--data_path=<PATH> --dataset=CIFAR100 --use_test --transform=ResNet --model=WideResNet28x10 --epochs=40 --wd=5e-4 --lr_1=0.05 --lr_2=0.01 --cycle=2
```

Test accuracy (%) of FGE and ensembling of independently trained networks (*Ind*) on CIFAR-100 for different training budgets. For each model the _Budget_ is defined as the number of epochs required to train the model with the conventional SGD procedure. 

| DNN (Method, Budget)      |  1 Budget    |  2 Budgets   |   3 Budgets      |
| ------------------------- |:------------:|:------------:|:----------------:|
| VGG16 (Ind, 200)          | 72.5 ± 0.1   | 74.8         | 75.6             |  
| VGG16 (FGE, 200)          | 74.6 ± 0.1   | 76.1         | 76.6             |  
| PreResNet164 (Ind, 200)   | 78.4 ± 0.1   | 80.5         | 81.6             |  
| PreResNet164 (FGE, 200)   | 80.3 ± 0.2   | 81.3         | 81.7             |  
| WideResNet28x10 (Ind, 200)| 80.8 ± 0.3   | 82.4         | 83.0             |  
| WideResNet28x10 (FGE, 200)| 82.3 ± 0.2   | 82.9         | 83.2             |  

<!---
## CIFAR-10

Test accuracy (%) of FGE and ensembling of independently trained networks (*Ind*) on CIFAR-10 for different training budgets.

!!Update!!

| DNN (Method, Budget)      |  1 Budget    |  2 Budgets   |   3 Budgets      |
| ------------------------- |:------------:|:------------:|:----------------:|
| VGG16 (Ind, 200)          | 93.17 ± 0.15 | 93.58        | 94.05            |  
| VGG16 (FGE, 200)          | 93.64 ± 0.03 | 94.01        | 94.33            |  
| PreResNet164 (Ind, 150)   | 00.00 ± 0.00 | 00.00        | 00.00            |   
| PreResNet164 (FGE, 150)   | 00.00 ± 0.00 | 00.00        | 00.00            |  
| WideResNet28x10 (Ind, 200)| 96.25 ± 0.04 | 96.65        | 96.80            |  
| WideResNet28x10 (FGE, 200)| 96.42 ± 0.07 | 96.49        | 96.53            |  
 
-->
 
# References
 
 Provided model implementations were adapted from
 * VGG: [github.com/pytorch/vision/](https://github.com/pytorch/vision/)
 * PreResNet: [github.com/bearpaw/pytorch-classification](https://github.com/bearpaw/pytorch-classification)
 * WideResNet: [github.com/meliketoy/wide-resnet.pytorch](https://github.com/meliketoy/wide-resnet.pytorch)
