#-*- coding: utf-8 -*-
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
from sklearn.metrics import average_precision_score
import pdb
import spacy
from random import seed
import itertools
from numpy import linalg as LA

tf.reset_default_graph()
random_seed = 320
np.random.seed(random_seed)
tf.set_random_seed(random_seed)

class Answer_Generator():
	def __init__(self,  batch_size, input_embedding_size, dim_image, dim_hidden_QI, dim_hidden_QIA, max_words_q, drop_out_rate, emb_matrix, decay, initial_bound):
		self.batch_size = batch_size
		self.input_embedding_size = input_embedding_size
		self.dim_image = dim_image
		self.dim_hidden_QI = dim_hidden_QI
		self.dim_hidden_QIA = dim_hidden_QIA
		self.max_words_q = max_words_q
		self.drop_out_rate = drop_out_rate
		self.decay = decay

		# Before-LSTM-embedding
		self.embed_ques_W = tf.Variable(emb_matrix, name='embed_ques_W')
		# self.embed_ques_W.assign(emb_matrix)
		self.embed_pos = tf.Variable(tf.random_uniform([7, 1], 0.0, 2.0, seed = random_seed), name='embed_pos')
		self.att_weight = tf.Variable(tf.random_uniform([1, 1], 0.0, 1.0, seed = random_seed, name = 'att_weight'))

		# question-embedding W1
		self.embed_Q_W = tf.Variable(tf.random_uniform([self.input_embedding_size, self.dim_hidden_QI], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_Q_W')
		self.embed_Q_b = tf.Variable(tf.random_uniform([self.dim_hidden_QI], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_Q_b')
		# 300 * 4096

		self.filters_1 = tf.Variable(tf.random_uniform([1, self.input_embedding_size,self.input_embedding_size], -1.0*initial_bound, initial_bound, seed = random_seed), name='filters_1')
		self.filters_2 = tf.Variable(tf.random_uniform([2, self.input_embedding_size,self.input_embedding_size], -1.0*initial_bound, initial_bound, seed = random_seed), name='filters_2')
		self.filters_3 = tf.Variable(tf.random_uniform([3, self.input_embedding_size,self.input_embedding_size], -1.0*initial_bound, initial_bound, seed = random_seed), name='filters_3')

		# Answer-embedding W3
		self.embed_A_W = tf.Variable(tf.random_uniform([self.input_embedding_size, self.dim_hidden_QIA], -1.0*initial_bound, initial_bound, seed = random_seed),name='embed_A_W')
		self.embed_A_b = tf.Variable(tf.random_uniform([self.dim_hidden_QIA], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_A_b')
		# 300 * 4096

		# image-embedding W2
		self.embed_image_W = tf.Variable(tf.random_uniform([dim_image, self.dim_hidden_QI], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_image_W')
		self.embed_image_b = tf.Variable(tf.random_uniform([dim_hidden_QI], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_image_b')
		# 2048 * 4096

		self.embed_scor_W = tf.Variable(tf.random_uniform([dim_hidden_QIA, num_output], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_scor_W')
		self.embed_scor_b = tf.Variable(tf.random_uniform([num_output], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_scor_b')

		# QI-embedding W3
		self.embed_QI_W = tf.Variable(tf.random_uniform([dim_hidden_QI, dim_hidden_QIA], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_QI_W')
		self.embed_QI_b = tf.Variable(tf.random_uniform([dim_hidden_QIA], -1.0*initial_bound, initial_bound, seed = random_seed), name='embed_QI_b')
		# 4096 * 4096

	def build_model(self, is_training):

		image = tf.placeholder(tf.float32, [None, 49, self.dim_image])

		question = tf.placeholder(tf.int32, [None, self.max_words_q])
		answer = tf.placeholder(tf.int32, [None, self.max_words_q])
		question_length = tf.placeholder(tf.int32, [None])
		answer_length = tf.placeholder(tf.int32, [None])
		label = tf.placeholder(tf.float32, [None,2])

		ques_pos = tf.placeholder(tf.int32, [None, self.max_words_q])
		ans_pos = tf.placeholder(tf.int32, [None, self.max_words_q])

		q_length = tf.reshape(question_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
		a_length = tf.reshape(answer_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
		q_length = tf.tile(q_length, [1, self.input_embedding_size])  # Create multiple columns.
		a_length = tf.tile(a_length, [1, self.input_embedding_size])  # Create multiple columns.
		q_length = tf.cast(q_length, tf.float32)
		a_length = tf.cast(a_length, tf.float32)

		stride_1 = 1
		stride_2 = 1
		stride_3 = 1

		inputs_ques = tf.nn.embedding_lookup(self.embed_ques_W, question)
		inputs_ans = tf.nn.embedding_lookup(self.embed_ques_W, answer)

		inputs_ques_tag = tf.nn.embedding_lookup(self.embed_pos, ques_pos)
		inputs_ans_tag = tf.nn.embedding_lookup(self.embed_pos, ans_pos)

		inputs_ques = inputs_ques * inputs_ques_tag
		inputs_ans = inputs_ans * inputs_ans_tag

		inputs_ques_1 = tf.nn.conv1d(inputs_ques, self.filters_1, stride_1, padding = "SAME")
		inputs_ques_2 = tf.nn.conv1d(inputs_ques, self.filters_2, stride_2, padding = "SAME")
		inputs_ques_3 = tf.nn.conv1d(inputs_ques, self.filters_3, stride_3, padding = "SAME")
		inputs_ques = tf.maximum(tf.maximum(inputs_ques_1,inputs_ques_2),inputs_ques_3)

		inputs_ans_1 = tf.nn.conv1d(inputs_ans, self.filters_1, stride_1, padding = "SAME")
		inputs_ans_2 = tf.nn.conv1d(inputs_ans, self.filters_2, stride_2, padding = "SAME")
		inputs_ans_3 = tf.nn.conv1d(inputs_ans, self.filters_3, stride_3, padding = "SAME")
		inputs_ans = tf.maximum(tf.maximum(inputs_ans_1,inputs_ans_2),inputs_ans_3)

		# input_ques: 500 * 26 * 300
		# input_ans: 500 * 26 * 300

		ques_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_ques, [-1, 300]), self.embed_Q_W, self.embed_Q_b), [-1, 26, 4096])
		ques_local = tf.tanh(ques_local_emb)
		# 50 * 26 * 4096

		ans_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_ans, [-1, 300]), self.embed_A_W, self.embed_A_b), [-1, 26, 4096])
		ans_local = tf.tanh(ans_local_emb)
		# 500 * 26 * 4096

		img_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(image, [-1, 2048]), self.embed_image_W, self.embed_image_b), [-1, 49, 4096])
		img_local = tf.nn.relu(img_local_emb)
		# img_local = img_local_emb * tf.nn.relu(img_local_emb)
		# 500 * 196 * 4096

		ques_aff = tf.matmul(ques_local, tf.transpose(img_local, [0, 2, 1]))
		ans_aff = tf.matmul(ans_local, tf.transpose(img_local, [0, 2, 1]))
		# 500 * 26 * 196

		ques_aff_softmax = tf.nn.softmax(ques_aff, dim = -1)
		ans_aff_softmax = tf.nn.softmax(ans_aff, dim = -1)

		ques_pool = tf.reduce_max(ques_aff_softmax, 1)
		ans_pool = tf.reduce_max(ans_aff_softmax, 1)

		ques_ans_pool = ans_pool + self.att_weight * ques_pool
		att_pool = ques_ans_pool/tf.reshape(tf.reduce_sum(ques_ans_pool, 1), [-1, 1])
		att = tf.reshape(att_pool, [-1, 1, 49])

		image_emb = tf.reduce_sum(tf.matmul(att, img_local), 1)
		# 500 * 2048

		state_que = tf.div(tf.reduce_sum(inputs_ques, 1), q_length)
		state_ans = tf.div(tf.reduce_sum(inputs_ans, 1), a_length)
		# batch_size * 300

		loss = 0.0

		# multimodal (fusing question & image)
		Q_drop = tf.nn.dropout(state_que, 1-self.drop_out_rate)
		Q_linear = tf.nn.xw_plus_b(Q_drop, self.embed_Q_W, self.embed_Q_b)
		Q_emb = tf.tanh(Q_linear)


		A_drop = tf.nn.dropout(state_ans, 1-self.drop_out_rate)
		A_linear = tf.nn.xw_plus_b(A_drop, self.embed_A_W, self.embed_A_b)
		A_emb = tf.tanh(A_linear)

		# A_emb_BN = tf.contrib.layers.batch_norm(A_emb, decay=self.decay, is_training = is_training, scope = 'A_emb_BN')

		QI = tf.multiply(Q_emb, image_emb)

		# QI_BN = tf.contrib.layers.batch_norm(QI, decay=self.decay, is_training = is_training, scope = 'QI_BN')

		QI_drop = tf.nn.dropout(QI, 1-self.drop_out_rate)
		QI_linear = tf.nn.xw_plus_b(QI_drop, self.embed_QI_W, self.embed_QI_b)
		QI_emb = tf.tanh(QI_linear)

		QIA = tf.multiply(QI_emb, A_emb)

		QIA_BN = tf.contrib.layers.batch_norm(QIA, decay=self.decay, is_training = is_training, scope = 'QIA_BN')

		scores_emb = tf.nn.xw_plus_b(QIA_BN, self.embed_scor_W, self.embed_scor_b)   #zhe
		cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=scores_emb, labels=label)

		sample_num = tf.shape(scores_emb)[0] / 4;
		pos_score_emb = scores_emb[:sample_num,0]
		neg_score_emb = scores_emb[sample_num:,0]
		# print scores_emb.shape
		# print neg_score_emb.shape
		diff_score = []
		for i in xrange(3):
			sc = tf.gather(neg_score_emb, tf.range(i,sample_num * 3,3))
			tmp = tf.nn.relu(margin + sc - pos_score_emb)
            #            tmp = margin + sc - pos_score_emb
			# tmp = (margin + sc - pos_score_emb) * tf.nn.relu(margin + sc - pos_score_emb)
			diff_score.append(tmp)

		max_diff_score = tf.maximum(tf.maximum(diff_score[0], diff_score[1]), diff_score[2])

		loss = tf.reduce_mean(cross_entropy) + 0.2 * tf.reduce_mean(max_diff_score) # or 0.5
		# Calculate loss
		
		return loss, image, question, answer, question_length, answer_length, label, ques_pos, ans_pos

	def build_generator(self, is_training):

		image = tf.placeholder(tf.float32, [None, 49, self.dim_image])

		question = tf.placeholder(tf.int32, [None, self.max_words_q])
		answer = tf.placeholder(tf.int32, [None, self.max_words_q])
		question_length = tf.placeholder(tf.int32, [None])
		answer_length = tf.placeholder(tf.int32, [None])
		label = tf.placeholder(tf.float32, [None,2])

		ques_pos = tf.placeholder(tf.int32, [None, self.max_words_q])
		ans_pos = tf.placeholder(tf.int32, [None, self.max_words_q])

		q_length = tf.reshape(question_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
		a_length = tf.reshape(answer_length, [-1, 1])    # Convert to a len(yp) x 1 matrix.
		q_length = tf.tile(q_length, [1, self.input_embedding_size])  # Create multiple columns.
		a_length = tf.tile(a_length, [1, self.input_embedding_size])  # Create multiple columns.
		q_length = tf.cast(q_length, tf.float32)
		a_length = tf.cast(a_length, tf.float32)

		stride_1 = 1
		stride_2 = 1
		stride_3 = 1

		inputs_ques = tf.nn.embedding_lookup(self.embed_ques_W, question)
		inputs_ans = tf.nn.embedding_lookup(self.embed_ques_W, answer)

		inputs_ques_tag = tf.nn.embedding_lookup(self.embed_pos, ques_pos)
		inputs_ans_tag = tf.nn.embedding_lookup(self.embed_pos, ans_pos)

		inputs_ques = inputs_ques * inputs_ques_tag
		inputs_ans = inputs_ans * inputs_ans_tag

		inputs_ques_1 = tf.nn.conv1d(inputs_ques, self.filters_1, stride_1, padding = "SAME")
		inputs_ques_2 = tf.nn.conv1d(inputs_ques, self.filters_2, stride_2, padding = "SAME")
		inputs_ques_3 = tf.nn.conv1d(inputs_ques, self.filters_3, stride_3, padding = "SAME")
		inputs_ques = tf.maximum(tf.maximum(inputs_ques_1,inputs_ques_2),inputs_ques_3)

		inputs_ans_1 = tf.nn.conv1d(inputs_ans, self.filters_1, stride_1, padding = "SAME")
		inputs_ans_2 = tf.nn.conv1d(inputs_ans, self.filters_2, stride_2, padding = "SAME")
		inputs_ans_3 = tf.nn.conv1d(inputs_ans, self.filters_3, stride_3, padding = "SAME")
		inputs_ans = tf.maximum(tf.maximum(inputs_ans_1,inputs_ans_2),inputs_ans_3)

		# input_ques: 500 * 26 * 300
		# input_ans: 500 * 26 * 300

		ques_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_ques, [-1, 300]), self.embed_Q_W, self.embed_Q_b), [-1, 26, 4096])
		ques_local = tf.tanh(ques_local_emb)
		# 50 * 26 * 4096

		ans_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(inputs_ans, [-1, 300]), self.embed_A_W, self.embed_A_b), [-1, 26, 4096])
		ans_local = tf.tanh(ans_local_emb)
		# 500 * 26 * 4096

		img_local_emb = tf.reshape(tf.nn.xw_plus_b(tf.reshape(image, [-1, 2048]), self.embed_image_W, self.embed_image_b), [-1, 49, 4096])
		img_local = tf.nn.relu(img_local_emb)
		# img_local = img_local_emb * tf.nn.sigmod(img_local_emb)
		# 500 * 196 * 4096

		ques_aff = tf.matmul(ques_local, tf.transpose(img_local, [0, 2, 1]))
		ans_aff = tf.matmul(ans_local, tf.transpose(img_local, [0, 2, 1]))
		# 500 * 26 * 196

		ques_aff_softmax = tf.nn.softmax(ques_aff, dim = -1)
		ans_aff_softmax = tf.nn.softmax(ans_aff, dim = -1)

		ques_pool = tf.reduce_max(ques_aff_softmax, 1)
		ans_pool = tf.reduce_max(ans_aff_softmax, 1)

		ques_ans_pool = ans_pool + self.att_weight * ques_pool
		att_pool = ques_ans_pool/tf.reshape(tf.reduce_sum(ques_ans_pool, 1), [-1, 1])
		att = tf.reshape(att_pool, [-1, 1, 49])

		image_emb = tf.reduce_sum(tf.matmul(att, img_local), 1)
		# 500 * 1024

		state_que = tf.div(tf.reduce_sum(inputs_ques, 1), q_length)
		state_ans = tf.div(tf.reduce_sum(inputs_ans, 1), a_length)
		# batch_size * 300


		loss = 0.0

		# multimodal (fusing question & image)
		Q_drop = tf.nn.dropout(state_que, 1-self.drop_out_rate)
		Q_linear = tf.nn.xw_plus_b(Q_drop, self.embed_Q_W, self.embed_Q_b)
		Q_emb = tf.tanh(Q_linear)

		# image_emb_BN = tf.contrib.layers.batch_norm(image_emb, decay=self.decay, is_training = is_training, scope = 'image_emb_BN')

		A_drop = tf.nn.dropout(state_ans, 1-self.drop_out_rate)
		A_linear = tf.nn.xw_plus_b(A_drop, self.embed_A_W, self.embed_A_b)
		A_emb = tf.tanh(A_linear)

		# A_emb_BN = tf.contrib.layers.batch_norm(A_emb, decay=self.decay, is_training = is_training, scope = 'A_emb_BN')

		QI = tf.multiply(Q_emb, image_emb)

		# QI_BN = tf.contrib.layers.batch_norm(QI, decay=self.decay, is_training = is_training, scope = 'QI_BN')

		QI_drop = tf.nn.dropout(QI, 1-self.drop_out_rate)
		QI_linear = tf.nn.xw_plus_b(QI_drop, self.embed_QI_W, self.embed_QI_b)
		QI_emb = tf.tanh(QI_linear)

		QIA = tf.multiply(QI_emb, A_emb)

		QIA_BN = tf.contrib.layers.batch_norm(QIA, decay=self.decay, is_training = is_training, scope = 'QIA_BN', reuse = True)

		scores_emb = tf.nn.xw_plus_b(QIA_BN, self.embed_scor_W, self.embed_scor_b)   #zhe
		
		generated_ANS = tf.transpose(scores_emb)

		return scores_emb, image, question, answer, question_length, answer_length, ques_pos, ans_pos

#####################################################
#                 Global Parameters         #
#####################################################
print('Loading parameters ...')
# Data input setting
input_img_h5_train = 'spatial_data_img_residule_train_14by14to7by7_norm.h5'
input_img_h5_test = 'spatial_data_img_residule_test_14by14to7by7_norm.h5'
input_ques_h5 = './data_prepro_0417_v1.h5'

# Train Parameters setting
learning_rate_global = 0.0001          # learning rate for rmsprop
learning_rate_nlp = 0.0002
learning_rate_decay_start = -1      # at what iteration to start decaying learning rate? (-1 = dont)
batch_size = 18 #576            # batch_size for each iterations
input_embedding_size = 300      # The encoding size of each token in the vocabulary
dim_image = 2048
dim_hidden_QI = 4096
dim_hidden_QIA = 4096 #1024         # size of the common embedding vector
num_output = 2#1000         # number of output answers
img_norm = 1                # normalize the image feature. 1 = normalize, 0 = not normalize
decay_factor = 0.99997592083
margin = 0.2
# Check point
checkpoint_path = 'model_save/'

# misc
gpu_id = 0
max_itr = 150000
n_epochs = 300
max_words_q = 26
num_answer = 1000

#####################################################

def right_align(seq, lengths):
	v = np.zeros(np.shape(seq))
	N = np.shape(seq)[1]
	for i in range(np.shape(seq)[0]):
		v[i][N-lengths[i]:N]=seq[i][0:lengths[i]]
	return v

def get_data():

	train_data = {}
	# load json file
	# load image feature
	print('loading image feature...')
	with h5py.File(input_img_h5_train,'r') as hf:
		# -----0~82459------  at most 47000
		tem = hf.get('images_train')
		img_feature = np.array(tem).reshape(-1, 49, 2048)
		# batch * 7 * 7 * 2048
	# load h5 file
	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		# total number of training data is 215375
		# question is (26, )
		tem = hf.get('ques_train')
		train_data['question'] = np.array(tem)-1
		# max length is 23
		tem = hf.get('ques_length_train')
		train_data['length_q'] = np.array(tem)
		# total 82460 img
		tem = hf.get('img_pos_train')
		# convert into 0~82459
		train_data['img_list'] = np.array(tem)-1
		# answer
		tem = hf.get('ans_train')
		train_data['answer'] = np.array(tem)-1

		tem = hf.get('ans_length_train')
		train_data['length_a'] = np.array(tem)

		tem = hf.get('target_train')
		train_data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

		train_data['emb_matrix'] = np.array(hf.get('emb_matrix'))

		train_data['ques_pos'] = np.array(hf.get('pos_train_ques')) - 1
		train_data['ans_pos'] = np.array(hf.get('pos_train_ans')) - 1


	print('question & answer aligning')
	train_data['question'] = right_align(train_data['question'], train_data['length_q'])
	train_data['answer'] = right_align(train_data['answer'], train_data['length_a'])
	train_data['ques_pos'] = right_align(train_data['ques_pos'], train_data['length_q'])
	train_data['ans_pos'] = right_align(train_data['ans_pos'], train_data['length_a'])


	print('Normalizing image feature')
	if img_norm:
		# tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
		tem = LA.norm(img_feature, axis = 2)
		for i in range(tem.shape[0]):
			for j in range(tem.shape[1]):
				img_feature[i,j,:] = img_feature[i,j,:]/tem[i, j]
		# img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(2048,1))))

	return img_feature, train_data

def get_nonzero_num(np_arr):
	return (np_arr != 0).sum(1)

def get_data_test():

	dataset = {}
	test_data = {}
	# load json file
	# load image feature
	print('loading image feature...')
	with h5py.File(input_img_h5_test,'r') as hf:
		tem = hf.get('images_test')
		img_feature = np.array(tem).reshape(-1, 49, 2048)
	# load h5 file
	print('loading h5 file...')
	with h5py.File(input_ques_h5,'r') as hf:
		# total number of training data is 215375
		# question is (26, )
		tem = hf.get('ques_test')
		test_data['question'] = np.array(tem)-1
		# max length is 23
		tem = hf.get('ques_length_test')
		test_data['length_q'] = np.array(tem)
		# total 82460 img
		tem = hf.get('img_pos_test')
		# convert into 0~82459
		test_data['img_list'] = np.array(tem)-1
		# quiestion id
		tem = hf.get('question_id_test')
		test_data['ques_id'] = np.array(tem)
		# answer
		tem = hf.get('ans_test')
		test_data['answer'] = np.array(tem)-1

		tem = hf.get('ans_length_test')
		test_data['length_a'] = np.array(tem)

		tem = hf.get('target_test')
		test_data['target'] = np.transpose(np.vstack((np.array(tem), 1-np.array(tem))))

		test_data['emb_matrix'] = np.array(hf.get('emb_matrix'))

		test_data['ques_pos'] = np.array(hf.get('pos_test_ques'))-1
		test_data['ans_pos'] = np.array(hf.get('pos_test_ans'))-1

	print('question aligning')
	test_data['question'] = right_align(test_data['question'], test_data['length_q'])
	test_data['answer'] = right_align(test_data['answer'], test_data['length_a'])
	test_data['ques_pos'] = right_align(test_data['ques_pos'], test_data['length_q'])
	test_data['ans_pos'] = right_align(test_data['ans_pos'], test_data['length_a'])

	print('Normalizing image feature')
	if img_norm:
		# tem = np.sqrt(np.sum(np.multiply(img_feature, img_feature), axis=1))
		tem = LA.norm(img_feature, axis = 2)
		for i in range(tem.shape[0]):
			for j in range(tem.shape[1]):
				img_feature[i,j,:] = img_feature[i,j,:]/tem[i, j]
		# img_feature = np.divide(img_feature, np.transpose(np.tile(tem,(2048,1))))

	return img_feature, test_data

def train():
	print('loading dataset...')
	img_feature_train, train_data = get_data()
	img_feature_test, test_data = get_data_test()
	num_train = train_data['question'].shape[0]

	print('constructing  model...')

	tf.reset_default_graph()
	tf.set_random_seed(random_seed)

	model = Answer_Generator(
		batch_size = batch_size,
		input_embedding_size = input_embedding_size,
		dim_image = dim_image,
		dim_hidden_QI = dim_hidden_QI,
		dim_hidden_QIA = dim_hidden_QIA,
		max_words_q = max_words_q,
		drop_out_rate = 0,
		emb_matrix = train_data['emb_matrix'],
		decay = 0.995,
		initial_bound = 0.10)

	tf_loss, tf_image, tf_question, tf_answer, tf_question_length, tf_answer_length, tf_label, tf_ques_pos_train, tf_ans_pos_train = model.build_model(True)

	tvars = tf.trainable_variables()

	nlp_vars = [tvars[0]]
	global_vars = tvars[1:]

	global_step = tf.Variable(0, tf.int32)

	lr_global = tf.train.exponential_decay(learning_rate_global, global_step, 1, decay_factor)
	opt_global = tf.train.AdamOptimizer(learning_rate = lr_global)
	lr_nlp = tf.train.exponential_decay(learning_rate_nlp, global_step, 1, decay_factor)
	opt_nlp = tf.train.AdamOptimizer(learning_rate = lr_nlp)

	# gradient clipping

	gvs = tf.gradients(tf_loss, nlp_vars + global_vars)
	gvs_nlp = [gvs[0]]
	gvs_global = gvs[1:]
	train_op_nlp = opt_nlp.apply_gradients(zip(gvs_nlp, nlp_vars), global_step=global_step)
	train_op_global = opt_global.apply_gradients(zip(gvs_global, global_vars),global_step=global_step)
	train_op = tf.group(train_op_nlp, train_op_global)

	sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
	saver = tf.train.Saver(max_to_keep=100)

	tf.global_variables_initializer().run()

	print('start training...')
	for itr in range(max_itr):

		pos_id = 4*np.array(list(range(num_train/4)))
		np.random.shuffle(pos_id)

		flag = False
		for idx in xrange(0, num_train/4-1, batch_size/3):

			tStart = time.time()

			if idx + batch_size/3 >= num_train/4-1:
				flag = True

			curr_pos = pos_id[idx:min(idx + batch_size/3, num_train/4)]
			np.random.shuffle(curr_pos)
			curr_neg = [np.random.choice([id+1, id+2, id+3], 3, replace = False) for id in curr_pos]
			curr_neg = list(itertools.chain(*curr_neg))
			# np.random.shuffle(curr_neg)
			# print len(curr_pos), len(curr_neg)
			index = list(curr_pos) + list(curr_neg)
			# np.random.shuffle(index)

			current_question = train_data['question'][index,:]
			current_length_q = train_data['length_q'][index]
			current_answer = train_data['answer'][index]
			current_length_a = train_data['length_a'][index]
			current_img_list = train_data['img_list'][index]
			current_target = train_data['target'][index]
			# print current_target.shape
			# for i in xrange(len(current_target)):
			# 	if current_target[i][0] == 1:
			# 		current_target[i][0] = 1
			# 	else:
			# 		current_target[i][1] = 1

			current_target = np.array(current_target);
			current_ques_pos = train_data['ques_pos'][index,:]
			current_ans_pos = train_data['ans_pos'][index,:]
			current_img = img_feature_train[current_img_list,:]
 
			# do the training process!!!
			_, loss = sess.run(
						[train_op, tf_loss],
						feed_dict={
							tf_image: current_img,
							tf_question: current_question,
							tf_answer: current_answer,
							tf_label: current_target,
							tf_question_length: current_length_q,
							tf_answer_length: current_length_a,
							tf_ques_pos_train: current_ques_pos,
							tf_ans_pos_train: current_ans_pos
							})


			tStop = time.time()

			print_itr = itr * (num_train/4)/(batch_size/3) + idx

			if np.mod(idx, 10000) == 0:
				print ("Iteration: ", itr, "____", idx, " Loss: ", loss, " Learning Rate: ", lr_global.eval())
				print ("Time Cost:", round(tStop - tStart,2), "s")
			if flag or (itr == 0 and idx == 0):
				print ("Iteration: ", itr, "____", idx, " is done. Saving the model ...")

				num_test = test_data['question'].shape[0]

				tf_proba_test, tf_image_test, tf_question_test, tf_answer_test, tf_question_test_length, \
				tf_answer_test_length, tf_ques_pos_test, tf_ans_pos_test = model.build_generator(False)

				result = {}
				for current_batch_start_idx in xrange(0, num_test-1, batch_size):
					tStart = time.time()
					# set data into current
					if current_batch_start_idx + batch_size < num_test:
						current_batch_file_idx = range(current_batch_start_idx, current_batch_start_idx + batch_size)
					else:
						current_batch_file_idx = range(current_batch_start_idx, num_test)

					current_question = test_data['question'][current_batch_file_idx,:]
					current_length_q = test_data['length_q'][current_batch_file_idx]
					current_img_list = test_data['img_list'][current_batch_file_idx]
					current_answer = test_data['answer'][current_batch_file_idx,:]
					current_length_a = test_data['length_a'][current_batch_file_idx]
					current_ques_id  = test_data['ques_id'][current_batch_file_idx]
					current_target = test_data['target'][current_batch_file_idx]
					current_ques_pos = test_data['ques_pos'][current_batch_file_idx,:]
					current_ans_pos = test_data['ans_pos'][current_batch_file_idx,:]
					current_img = img_feature_test[current_img_list,:] # (batch_size, dim_image)

					pred_proba = sess.run(
							tf_proba_test,
							feed_dict={
								tf_image_test: current_img,
								tf_question_test: current_question,
								tf_answer_test: current_answer,
								tf_question_test_length: current_length_q,
								tf_answer_test_length: current_length_a,
								tf_ques_pos_test: current_ques_pos,
								tf_ans_pos_test: current_ans_pos
								})					
					# initialize json list

					target, prob = getMaximumLikelihood(current_target, pred_proba, len(current_img))

					for i in list(range(0, len(current_img))):
						if str(current_ques_id[i]) not in result:
							result[str(current_ques_id[i])] = [target[i], prob[i]]
						else:
							if result[str(current_ques_id[i])][1] < prob[i]:
								result[str(current_ques_id[i])] = [target[i], prob[i]]
								

					tStop = time.time()

				print ("Testing done.")
				tStop_total = time.time()
				
				acc = 0

				# print list(xresult.iteritems())[:100]
				for k,v in result.iteritems():
					acc += v[0]
				print('Accuracy of test: ' + str(acc*1.0/len(result)))
				f2 = open("test_acc_v7w.txt", "a")
				f2.write(str(itr) + "____" + str(idx) + '\t' + str(acc*1.0/len(result)) + "\n")
				f2.close()


	print ("Finally, saving the model ...")
	tStop_total = time.time()
	print ("Total Time Cost:", round(tStop_total - tStart_total,2), "s")

def getMaximumLikelihood(raw_target, raw_prob, batch_size):
	target = np.zeros((batch_size,))
	prob = np.zeros((batch_size,))
	for i in list(range(0, batch_size)):
                prob[i] = softmax(raw_prob[i,0], raw_prob[i,1])
                target[i] = raw_target[i,0]
		#prob[i] = raw_prob[i,0];
		#target[i] = raw_target[i,0]

	return target, prob

def softmax(a, b):
	return np.exp(a)/(np.exp(a) + np.exp(b))

if __name__ == '__main__':
	with tf.device('/gpu:'+str(0)):
		train()
 
