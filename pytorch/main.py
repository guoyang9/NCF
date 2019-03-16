from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import NCF_input
from NCF_input import NCFDataset
from NCF import NCF
import evaluate


DATA_PATH = './Data'

parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=0.001, type=float, 
					help="learning rate.")
parser.add_argument("--dropout", default=0.0, type=float, 
					help="dropout rate.")
parser.add_argument("--batch_size", default=128, type=int, 
					help="batch size when training.")
parser.add_argument("--gpu", default="0", type=str, 
					help="gpu card ID.")
parser.add_argument("--epochs", default=20, type=str, 
					help="training epoches.")
parser.add_argument("--top_k", default=10, type=int, 
					help="compute metrics@top_k.")
parser.add_argument("--clip_norm", default=5.0, type=float,
					help="clip norm for preventing gradient exploding.")
parser.add_argument("--embed_size", default=16, type=int,
					help="embedding size for users and items.")
parser.add_argument("--num_neg", default=4, type=int,
					help="sample negative items for training.")
parser.add_argument("--test_num_neg", default=99, type=int,
					help="sample part of negative items for testing.")

FLAGS = parser.parse_args()

opt_gpu = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu


################################## PREPARE DATASET ##############################

if not os.path.exists(DATA_PATH):
	os.makedirs(DATA_PATH)

	((train_features, train_labels), 
		(test_features, test_labels), 
		(user_size, item_size), 
		(user_set, item_set), 
		(user_bought, user_negative)) = NCF_input.load_data()

	train_features, train_labels = NCF_input.add_negative(train_features, 
			train_labels, user_negative, FLAGS.num_neg, is_training=True)
	test_features, test_labels = NCF_input.add_negative(test_features, 
			test_labels, user_negative, FLAGS.test_num_neg, is_training=False)

	data = dict([('train_features', train_features), ('train_labels', train_labels),
				('test_features', test_features), ('test_labels', test_labels),
				('user_size', user_size), ('item_size', item_size),
				('user_bought', user_bought), ('user_negative', user_negative)])
	np.save(os.path.join(DATA_PATH, 'data.npy'), data)

data = np.load(os.path.join(DATA_PATH, 'data.npy')).item()
print("Loading data finished!")

user_size = data['user_size']
item_size = data['item_size']

#Construct the train and test datasets
train_dataset = NCFDataset(data['train_features'], data['train_labels'])
test_dataset = NCFDataset(data['test_features'], data['test_labels'])

train_dataloader = DataLoader(train_dataset,
		batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset,
		batch_size=FLAGS.test_num_neg+1, shuffle=False, num_workers=4)

############################### CREATE MODEL #####################################

if os.path.exists('./model.pt'):
	model = torch.load('./model.pt')
else:
	model = NCF(user_size, item_size, FLAGS.embed_size, FLAGS.dropout)
	model.cuda()
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)

writer = SummaryWriter() #For visualization

############################### TRAINING #########################################

count = 0
for epoch in range(FLAGS.epochs):
	model.train() # Enable dropout (if have).
	start_time = time.time()

	for idx, batch_data in enumerate(train_dataloader):
		#Assign the user and item on GPU later.
		user = batch_data['user'].long().cuda()
		item = batch_data['item'].long().cuda()
		label = batch_data['label'].float().cuda()

		model.zero_grad()
		prediction = model(user, item)
		loss = loss_function(prediction, label)
		loss.backward()
		# nn.utils.clip_grad_norm(model.parameters(), FLAGS.clip_norm)
		optimizer.step()

		writer.add_scalar('data/loss', loss.data.item(), count)
		count += 1

	model.eval() #Disable dropout (if have).
	HR, NDCG = evaluate.metrics(model, test_dataloader, FLAGS.top_k)

	elapsed_time = time.time() - start_time
	print("Epoch: %d" %epoch + " Epoch time: " + time.strftime(
					"%H: %M: %S", time.gmtime(elapsed_time)))
	print("Hit ratio is %.3f\tNdcg is %.3f" %(np.mean(HR), np.mean(NDCG)))

torch.save(model, 'm.pt')
