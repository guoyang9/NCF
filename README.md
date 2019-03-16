# NCF
A pytorch and tensorflow GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17

This code only covers the Movielens 1M Dataset https://grouplens.org/datasets/movielens/. 

Based on the paper's result, the improvement of with pre-train over without pre-train is not too significant, so this version is just the NCF with the intergration of GMF and MLP.


## The requirements are as follows:
1.python 3.5

2.pytorch 0.4.0

3.tensorboardX (mainly useful when you want to visulize the loss, https://github.com/lanpa/tensorboard-pytorch)

4.tensorflow 1.7


## Example to run:
```
python main.py --batch_size=128 --lr=0.00001 --embed_size=16
```
