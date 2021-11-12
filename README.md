# NCF
### A pytorch GPU implementation of He et al. "Neural Collaborative Filtering" at WWW'17
- Forked repository from guoyang9/NCF.
- Updated coding style and experiment settings by zhykoties.

Note that I use the two sub datasets provided by Xiangnan's [repo](https://github.com/hexiangnan/neural_collaborative_filtering/tree/master/Data).

I randomly utilized a factor number **32**, MLP layers **3**, epochs is **20**, and posted the results in the original paper and this implementation here. I employed the **exactly same settings** with Xiangnan, including batch_size, learning rate, and all the initialization methods in Xiangnan's keras repo. From the results I observed, this repo can replicate the performance of the original NCF.
Xiangnan's keras [repo](https://github.com/hexiangnan/neural_collaborative_filtering):

Models | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ | --------------- | ----------------- | --------------- | -----------------
MLP    | 0.692 | 0.425 | 0.868 | 0.542
GMF    | - | - | - | -
NeuMF (without pre-training) | 0.701 | 0.425 | 0.870 | 0.549
NeuMF (with pre-training)	 | 0.726 | 0.445 | 0.879 | 0.555


This pytorch code:

Models | MovieLens HR@10 | MovieLens NDCG@10 | Pinterest HR@10 | Pinterest NDCG@10
------ | --------------- | ----------------- | --------------- | -----------------
MLP    | 0.691 | 0.416 | 0.866 | 0.537
GMF    | 0.708 | 0.429 | 0.867 | 0.546
NeuMF (without pre-training) | 0.701 | 0.424 | 0.867 | 0.544
NeuMF (with pre-training)	 | 0.720 | 0.439 | 0.879 | 0.555


## The requirements are as follows:
	* python==3.6
	* pandas==0.24.2
	* numpy==1.16.2
	* pytorch==1.0.1
	* gensim==3.7.1

## Example to run:
```
python main.py --dataset=ml-1m --model=ncf
```

## Dataset info
From the official repo: https://github.com/hexiangnan/neural_collaborative_filtering

We provide two processed datasets: MovieLens 1 Million (ml-1m) and Pinterest (pinterest-20). 

train.rating: 
- Train file.
- Each Line is a training instance: userID\t itemID\t rating\t timestamp (if have)

test.rating:
- Test file (positive instances). 
- Each Line is a testing instance: userID\t itemID\t rating\t timestamp (if have)

test.negative
- Test file (negative instances).
- Each line corresponds to the line of test.rating, containing 99 negative samples.  
- Each line is in the format: (userID,itemID)\t negativeItemID1\t negativeItemID2 ...