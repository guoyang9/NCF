# NCF
A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17

Note that we use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data).

There is one thing weird about this implementation. As I totally mimiced all the settings in He's repo, but the final performance is not the smae as reported in the original papers. That the Hit Ratio is approaching 1.0 but the NDCG is pretty low! And the performance is not too sensitive with the increasing of number of factors. I'll correct this in futher commits, just leave it here.

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
