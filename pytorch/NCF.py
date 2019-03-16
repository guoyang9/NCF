from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F 
from torch.autograd import Variable

class NCF(nn.Module):
	def __init__(self, user_size, item_size, embed_size, dropout):
		super(NCF, self).__init__()
		self.user_size = user_size
		self.item_size = item_size
		self.embed_size = embed_size
		self.dropout = dropout

		# Custom weights initialization.
		def init_weights(m):
			# if isinstance(m, nn.Conv2d):
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				nn.init.constant_(m.bias, 0)

		self.embed_user_GMF = nn.Linear(self.user_size, self.embed_size)
		self.embed_user_MLP = nn.Linear(self.user_size, self.embed_size)
		self.embed_item_GMF = nn.Linear(self.item_size, self.embed_size)
		self.embed_item_MLP = nn.Linear(self.item_size, self.embed_size)

		self.embed_user_GMF.apply(init_weights)
		self.embed_user_MLP.apply(init_weights)
		self.embed_item_GMF.apply(init_weights)
		self.embed_item_MLP.apply(init_weights)

		self.MLP_layers = nn.Sequential(
			nn.Linear(embed_size*2, embed_size*2),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(embed_size*2, embed_size),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			nn.Linear(embed_size, embed_size//2),
			nn.ReLU(),
			nn.Dropout(p=self.dropout)
			)
		self.MLP_layers.apply(init_weights)

		self.predict_layer = nn.Linear(embed_size*3//2, 1)
		self.predict_layer.apply(init_weights)

	def convert_one_hot(self, feature, size):
		""" Convert user and item ids into one-hot format. """
		batch_size = feature.shape[0]
		feature = feature.view(batch_size, 1)
		f_onehot = torch.cuda.FloatTensor(batch_size, size)
		f_onehot.zero_()
		f_onehot.scatter_(-1, feature, 1)

		return f_onehot

	def forward(self, user, item):
		user = self.convert_one_hot(user, self.user_size)
		item = self.convert_one_hot(item, self.item_size)

		embed_user_GMF = F.relu(self.embed_user_GMF(user))
		embed_user_MLP = F.relu(self.embed_user_MLP(user))
		embed_item_GMF = F.relu(self.embed_item_GMF(item))
		embed_item_MLP = F.relu(self.embed_item_MLP(item))


		# GMF part begins. 
		output_GMF = embed_user_GMF * embed_item_GMF

		# MLP part begins.
		interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)

		inter_layer = self.MLP_layers(interaction)

		# Concatenation part begins.
		concat = torch.cat((output_GMF, inter_layer), -1)
		prediction = F.sigmoid(self.predict_layer(concat))

		return prediction.view(-1)


