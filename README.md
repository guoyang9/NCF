# NCF
A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17

Note that we use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data).


## The requirements are as follows:
* python==3.6

* pandas==0.24.2

* numpy==1.16.2

* pytorch==1.0.1

* gensim==3.7.1

* tensorboardX==1.6 (mainly useful when you want to visulize the loss, see https://github.com/lanpa/tensorboard-pytorch)

## Example to run:
```
python main.py --batch_size=256 --lr=0.001 --embed_size=16
```
