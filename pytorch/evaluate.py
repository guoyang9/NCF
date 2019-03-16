from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch

import numpy as np

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item),
							dtype=torch.float32)
		return torch.reciprocal(torch.log2(index+2))
	return 0

def metrics(model, test_dataloader, top_k):
	HR, NDCG = [], []

	for batch_data in test_dataloader:
		user = batch_data['user'].long().cuda()
		item = batch_data['item'].long().cuda()
		# label = batch_data['label'].numpy()

		prediction = model(user, item)
		_, indices = torch.topk(prediction, top_k)
		recommend = torch.take(item, indices)

		HR.append(hit(item[0], recommend))
		NDCG.append(ndcg(item[0], recommend))

	return np.mean(HR), np.mean(NDCG)

		

