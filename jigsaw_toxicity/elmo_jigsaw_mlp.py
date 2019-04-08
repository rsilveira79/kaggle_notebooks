## General Imports

from contextlib import contextmanager
import warnings
import pandas as pd
import numpy as np
from string import punctuation
import os, re, sys, json, requests, time

### Torch, Sklearn imports
from sklearn.model_selection import train_test_split
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
print(torch.__version__)


### NLP Libs
import allennlp
from allennlp.modules.elmo import Elmo, batch_to_ids
import spacy
import gensim
from nltk import download
from nltk.corpus import stopwords
download('stopwords')

### Visualization Libs
import matplotlib.pyplot as plt
import seaborn as sns 


### Pre-processing text

def transformText(text, do_stop=False, do_stem=False):
	stops = set(stopwords.words("english"))
	# Convert text to lower
	text = text.lower()
	
	# Cleaning input
	text = text.replace("'s","")
	text = text.replace("’s","")
	text = text.replace("?","")
	text = text.replace("-","")
	
	# Removing non ASCII chars    
	text = re.sub(r'[^\x00-\x7f]',r' ',text)
	# Strip multiple whitespaces
	text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
	# Removing all the stopwords
	if (do_stop==True):
		filtered_words = [word for word in text.split() if word not in stops]
	else:
		filtered_words = [word for word in text.split()]
	# Preprocessed text after stop words removal
	text = " ".join(filtered_words)
	# Remove the punctuation
	text = gensim.parsing.preprocessing.strip_punctuation2(text)
	# Strip multiple whitespaces
	text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
	if (do_stem==True):
		# Stemming
		text = gensim.parsing.preprocessing.stem_text(text)
	return text

### ELMo Embeddings
def get_elmo(sent, elmo):
	sent = [sent.split()]
	character_ids = batch_to_ids(sent)
	embeddings = elmo(character_ids)
	rep = embeddings['elmo_representations'][0]
	if torch.cuda.is_available():
		rep = rep.cuda()
	rep = rep.squeeze(dim=0)
	avg = rep.mean(dim=0)
	return avg

### Intents Dataloader
class Intents(Dataset):
	def __init__(self, dataframe, elmo):
		self.len = len(dataframe)
		self.data = dataframe
		self.elmo = elmo
		
	def __getitem__(self, index):
		phrase = self.data.clean_comment_text[index]
		X = get_elmo(phrase, self.elmo)
		y = self.data.label_class[index]
		return X, y
	
	def __len__(self):
		return self.len

### MLP Model
class MLPUncertainty(nn.Module):
	def __init__(self, inputdim, 
						nclasses, 
						nhidden, 
						dropout = 0,
						cudaEfficient=True,
						decay = 1e-6):
		super(MLPUncertainty, self).__init__()
		"""
		PARAMETERS:
		-dropout:    dropout for MLP
		"""
		
		self.inputdim = inputdim
		self.hidden_dim = nhidden
		self.dropout = dropout
		self.decay = decay
		self.nclasses = nclasses
		
		if cudaEfficient:
			self.model = nn.Sequential(
				nn.Linear(self.inputdim, nhidden),
				nn.Dropout(p=self.dropout),
				nn.ReLU(),
				nn.Linear(nhidden, self.nclasses),
				).cuda()
		else:
			self.model = nn.Sequential(
				nn.Linear(self.inputdim, nhidden),
				nn.Dropout(p=self.dropout),
				nn.ReLU(),
				nn.Linear(nhidden, self.nclasses),
				)
	def forward(self, x):
		log_probs = self.model(x)
		return log_probs

### Main Function
def main():
	warnings.filterwarnings('ignore')
	
	print("[GPU]: {}".format(torch.cuda.is_available()))
	train = pd.read_csv('dataset/train.csv')
	test = pd.read_csv('dataset/test.csv')

	# ## Apply dataset text cleaning and suffling the train dataset
	train['clean_comment_text']=train['comment_text'].apply(lambda x: transformText(x))
	train = train.sample(frac=1).reset_index(drop=True)
	train['label_class']=(train['target'].values > 0.5).astype(int)

	# Elmo Embeddings
	options_file = '../../vectors/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
	weight_file =  '../../vectors/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
	elmo = Elmo(options_file, weight_file, 1, dropout=0)
	# ## Train/Test Split
	data_split = int(0.8*len(train))
	train_dataset = train[:data_split]
	valid_dataset = train[data_split:-1].reset_index(drop=True)
	training_set = Intents(train_dataset, elmo)
	validing_set = Intents(valid_dataset, elmo)
	params = {'batch_size': 64, 'shuffle': True, 'num_workers': 0}
	training_loader = DataLoader(training_set, **params)
	testing_loader = DataLoader(validing_set, **params)

	## Model Instatiation
	INP_DIM = elmo.get_output_dim()
	NUM_LABELS = len(set(train.label_class))
	NHIDDEN = 1024
	DROPOUT = 0
	model = MLPUncertainty(inputdim = INP_DIM ,
			nhidden = NHIDDEN,
			nclasses = NUM_LABELS,
			dropout = DROPOUT, 
			cudaEfficient = torch.cuda.is_available())
	if torch.cuda.is_available():
		model = model.cuda()

	## Hyperparams
	loss_function = nn.CrossEntropyLoss()
	learning_rate = 0.0005
	optimizer = optim.Adam(params =  model.parameters(), lr=learning_rate)
	max_epochs = 15

	## Start training
	for epoch in range(max_epochs):
		print("EPOCH -- {}".format(epoch))
		for i, (sent, label) in enumerate(training_loader):
			optimizer.zero_grad()
			sent = Variable(sent)
			label = Variable(label)
			if torch.cuda.is_available():
				sent = sent.cuda()
				label = label.cuda()
			output = model.forward(sent)
			loss = loss_function(output, label)
			loss.backward()
			optimizer.step()
			if i%5 == 0:
				correct = 0
				total = 0
				for sent, label in testing_loader:
					sent = Variable(sent)
					label = Variable(label)
					if torch.cuda.is_available():
						sent = sent.cuda()
						label = label.cuda()
					output = model.forward(sent)
					_, predicted = torch.max(output.data, 1)
					total += label.size(0)
					correct += (predicted.cpu() == label.cpu()).sum()
				accuracy = 100.00 * correct.numpy() / total
				# Print Loss
				print('Iteration: {}. Loss: {}. Accuracy: {}%'.format(i, loss.data, accuracy))
				
	## Save Model
	model_file = 'model_elmo.pth'
	torch.save(model.state_dict(), model_file)

if __name__ == '__main__':
	main()
