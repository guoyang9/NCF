from __future__ import absolute_import
from __future__ import division

import numpy as np 
import pandas as pd 

from torch.utils.data import Dataset


DATA_DIR = '/home/yang/Datasets/ml-1m/ratings.dat' # raw data
DATA_PATH = './Data' # for saving the processed data
COLUMN_NAMES = ['user', 'item']


def re_index(s):
	""" for reindexing the item set. """
	i = 0
	s_map = {}
	for key in s:
		s_map[key] = i
		i += 1

	return s_map

def load_data():
	full_data = pd.read_csv(
		DATA_DIR, sep='::', header=None, names=COLUMN_NAMES, 
		usecols=[0,1], dtype={0: np.int32, 1: np.int32}, engine='python')
	
	#Forcing the user index begining from 0.
	full_data.user = full_data.user - 1
	user_set = set(full_data.user.unique())
	item_set = set(full_data.item.unique())

	user_size = len(user_set)
	item_size = len(item_set)

	item_map = re_index(item_set)
	item_list = []
	for i in range(len(full_data)):
		item_list.append(item_map[full_data['item'][i]])
	full_data['item'] = item_list

	item_set = set(full_data.item.unique())

	#Group each user's interactions(purchased items) into dictionary.
	user_bought = {}
	for i in range(len(full_data)):
		u = full_data['user'][i]
		t = full_data['item'][i]
		if u not in user_bought:
			user_bought[u] = []
		user_bought[u].append(t)

	#Group each user's negative interactions.
	user_negative = {}
	for key in user_bought:
		user_negative[key] = list(item_set - set(user_bought[key]))

	#Splitting the full set into train and test sets.
	user_length = full_data.groupby('user').size().tolist() #User transaction count.
	split_train_test = [] 

	for i in range(len(user_set)):
		for _ in range(user_length[i] - 1):
			split_train_test.append('train')
		split_train_test.append('test') #Last one for test.

	full_data['split'] = split_train_test
	train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)
	test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
	del train_data['split']
	del test_data['split']

	labels = np.ones(len(train_data), dtype=np.int32)

	train_features = train_data
	train_labels = labels.tolist()
	test_features = test_data
	test_labels = test_data['item'].tolist() #take the groundtruth item as test labels.

	return ((train_features, train_labels), 
			(test_features, test_labels), 
			(user_size, item_size), 
			(user_set, item_set), 
			(user_bought, user_negative))

def add_negative(features, labels, user_negative, numbers, is_training):
	""" Adding negative samples to training and testing data. """
	feature_user, feature_item, labels_add, features_dict = [], [], [], {}

	for i in range(len(features)):
		user = features['user'][i]
		item = features['item'][i]
		label = labels[i]

		feature_user.append(user)
		feature_item.append(item)
		labels_add.append(label)

		#uniformly sample negative ones from candidate negative items
		neg_samples = np.random.choice(user_negative[user], size=numbers, 
								replace=False).tolist()

		if is_training:
			for k in neg_samples:
				feature_user.append(user)
				feature_item.append(k)
				labels_add.append(0)
		else:
			for k in neg_samples:
				feature_user.append(user)
				feature_item.append(k)
				labels_add.append(k)

	features_dict['user'] = feature_user
	features_dict['item'] = feature_item

	return features_dict, labels_add


class NCFDataset(Dataset):
	def __init__(self, features, labels):
		"""
		After load_data processing, read train or test data. Num_neg is different for
		train and test. User_neg is the items that users have no explicit interaction.
		"""
		self.features = features
		self.labels = labels

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, idx):
		user = self.features['user'][idx]
		item = self.features['item'][idx]
		label = self.labels[idx]

		sample = {'user': user, 'item': item, 'label': label}

		return sample
