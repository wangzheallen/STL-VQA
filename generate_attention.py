#-*- coding: utf-8 -*-
# 64.72% #
import tensorflow as tf
import pandas as pd
import numpy as np
import os, h5py, sys, argparse
import ipdb
import time
import math
import cv2
import codecs, json
# from tensorflow.python.ops import rnn_cell
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

triple = json.load(open('/home/liangjic/BMVC/VQA_20170301/vqa_raw_train.json'))
print triple[210716]
for xid in xrange(len(triple)):
	if xid % 4 == 0 and triple[xid]['ans'] == 0:
		print xid
		break
	if xid % 4 > 0 and triple[xid]['ans'] == 1:
		print xid
		break
