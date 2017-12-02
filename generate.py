import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.image import imread
import cv2
import random

tag_dict = {}
tag_dict['CD'] = 1
tag_dict['JJ'] = 2
tag_dict['JJR'] = 2
tag_dict['JJS'] = 2
tag_dict['NN'] = 3
tag_dict['NNS'] = 3
tag_dict['NNP'] = 3
tag_dict['NNPS'] = 3
tag_dict['VB'] = 4
tag_dict['VBD'] = 4
tag_dict['VBG'] = 4
tag_dict['VBN'] = 4
tag_dict['VBP'] = 4
tag_dict['VBZ'] = 4
tag_dict['VBZ'] = 4
tag_dict['WP'] = 5
tag_dict['WP$'] = 5
tag_dict['WRB'] = 6
w = [1.41417003,0.65907812,1.98009944,1.50878048,0.68756366,1.72526932,1.37473035]

def sent_pos(sent):
	# sent is a list of words
	return [tag_dict[x[1]] if x[1] in tag_dict else 7 for x in pos_tag(sent)]

def nltk_tokenize(sent):
	sent = str(sent).lower()
	sent = sent.replace("-", " ")
	sent = sent.replace("/", " ")
	sent = sent.replace("`", " ")
	token = word_tokenize(sent)

	for i in range(len(token)):
		if token[i].isalpha():
			token[i] = spell(token[i]).lower()
	return token
prefix = '/home/liangjic/BMVC/rebuttal/'
# prefix = '/home/zwang15/data/'
# file_list = os.listdir(prefix + 'att')
file_list = sorted(os.listdir(prefix + 'att'))
# print file_list 

xtmp = 0
# f, ax = plt.subplots(2,3,sharex=False,sharey=False)
# f, ax = plt.subplots(0,0,sharex=False,sharey=False)
for file in file_list:
	# if not file[-6:].startswith('29776') and not file[-6].startswith('21071'):
	# 	continue
	# if file.find('sit') == -1 and file.find('Low') == -1:
	# 	continue
	# att = cv2.GaussianBlur(att,(5,5),10)



	tmp = file.split('*')
	qid = int(tmp[0])
		
	img = tmp[1]
	ques = tmp[2]
	ans = tmp[3]
	
	path = prefix + 'neg_result/'+ file + 'pdf'
	if qid <= 297764 or qid >= 297768:
		continue

	# path = prefix + '/pos_result/'+ file + '.pdf'
	# if qid % 4 or img !='v7w_498371':
	# 	continue
	print file
	if os.path.exists(path):
		continue

	att = np.load(prefix + 'att/' + file)
	att = np.array(map(lambda x: ((x-x.min())/(x.max()-x.min())), att)).reshape(7,7)
	ori = Image.open(prefix + 'images/' + img + '.jpg')
	ori = ori.resize((224,224))
	x = np.zeros((224,224))
	y = np.zeros((224,224))
	for idx in range(7):
		for idy in range(7):
			if att[idx][idy] > 0.7:
				att[idx][idy] = 1
	for idx in range(7):
		for idy in range(7):
			for i in range(224):
				for j in range(224):
					xx = idx * 32 + 16
					yy = idy * 32 + 16
					d = ((i - xx) ** 2 + (j - yy) ** 2) ** 0.5
					x[i][j] += max(0,att[idx][idy] - d * 0.01)
					# x[i][j] = max(x[i][j], att[idx][idy] - d * 0.01)
					# if x[i][j] < 0.4:
					# 	x[i][j] = 0	
	# f, ax = plt.subplots(1,1,sharex=False,sharey=False)
	# ax[0,0].get_xaxis().set_visible(False)
	# ax[0,0].get_yaxis().set_visible(False)
	plt.suptitle(file)
	plt.imshow( ori )
	plt.imshow( x, cmap=plt.cm.jet, alpha=0.9, interpolation='gaussian' )
	plt.savefig(path,foramt = 'pdf', dpi = 1000)
	# x = cv2.GaussianBlur(x,(5,5),0)
	# plt.title(ques + ' ' + ans )
	# plt.subplots(xtmp,1)


	# ax[xtmp,0].get_xaxis().set_visible(False)
	# ax[xtmp,0].get_yaxis().set_visible(False)
	# ax[xtmp,0].imshow( ori )
	# ax[xtmp,0].imshow( x, cmap=plt.cm.jet, alpha=0.9, interpolation='gaussian' )
	

	# fig,ax = plt.subplots(xtmp,2)

	# ques = nltk_tokenize(str(ques).lower())[:-1]
	# ans = nltk_tokenize(str(ans).lower())[:-1]
	# ques_pos = sent_pos(ques)
	# ans_pos = sent_pos(ans)
	# quesx = np.arange(len(ques + ans))
	# quesy = [w[x - 1] for x in (ques_pos + ans_pos)]
	# print ques + ans
	# ax[xtmp,1].plot(quesx, quesy,'ro-')
	# ax[xtmp,1].set_xticks(quesx)
	# ax[xtmp,1].set_xticklabels(ques + ans,rotation=90)

	# ax[xtmp,2].get_xaxis().set_visible(False)
	# ax[xtmp,2].get_yaxis().set_visible(False)
	# ax[xtmp,2].imshow( ori )
	# xtmp += 1
# f.autofmt_xdate()
# plt.tight_layout()
# plt.hold()
# plt.savefig(prefix + '1.pdf',foramt = 'pdf', dpi = 1000)
# plt.show()

