import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from PIL import Image
from matplotlib.image import imread
# import cv2
import random
from nltk.tokenize import word_tokenize
from nltk import pos_tag

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
			token[i] = token[i].lower()
	return token
#prefix = '/media/data/home/liangjic/BMVC/rebuttal/'
prefix = './'
# prefix = '/home/zwang15/data/'
# file_list = os.listdir(prefix + 'att')
file_list = sorted(os.listdir(prefix + 'att'))
# print file_list 

xtmp = 0
# f, ax = plt.subplots(2,3,sharex=False,sharey=False)
# f, ax = plt.subplots(0,0,sharex=False,sharey=False)
def Generate_Heat(att):
	x = np.zeros((224,224))
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

	return x 
def Get_graph(ax, aword, gid):
	xori = 1
	for file in file_list:
		qid,img,ques,ans = file.split('*')
		qid = int(qid)
		if qid != gid:
			continue
		#graph_path = '/media/data/home/liangjic/BMVC/rebuttal/graph/'+str(gid)
		graph_path = './graph/'+str(gid)
		if os.path.exists(graph_path):
			x = np.load(graph_path)
		else:
			att = np.load(prefix + 'att/' + file)
			att = np.array(map(lambda x: ((x-x.min())/(x.max()-x.min())), att)).reshape(7,7)			
			x = Generate_Heat(att)
			x.dump(graph_path)

		xori = Image.open(prefix + 'images/' + img + '.jpg')
		ori = xori.resize((224,224))
		ax.imshow(ori)
		ax.imshow(x, cmap=plt.cm.jet, alpha=0.9, interpolation='gaussian' )
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		ques = nltk_tokenize(str(ques).lower())[:-1]
		ans = nltk_tokenize(str(ans).lower())[:-1]
		ques_pos = sent_pos(ques)
		ans_pos = sent_pos(ans)
		quesx = np.arange(len(ques + ans))
		quesy = [w[x - 1] for x in (ques_pos + ans_pos)]
		print (ques + ans)

		aword.plot(quesx, quesy,'ro-')
		aword.set_xticks(quesx)
		aword.set_xticklabels(ques + ans,rotation=30, fontsize = 10)
		#aword.set_size_inches((3,3))
	return xori
		
	
def Pic(ax, graph_id, i):
	Get_graph(ax[1], ax[2], graph_id[1])
	ori = Get_graph(ax[3], ax[4], graph_id[0])
	
	ax[0].get_xaxis().set_visible(False)
	ax[0].get_yaxis().set_visible(False)
	if i==2 or i==4:
		ax[0].plot(np.random.random(size=30))
	ax[0].imshow( ori )

#graph_id =[[179456, 179457],[335448, 335449],[128800, 128801],[94908, 94909]]
good_id = [[100,101], [10052, 10053], [297766,297764], [32968, 32969], [202652, 202653]]#, [159696, 159697]] # [297740,297741],
bad_id  = [[179456, 179457], [335448, 335449],[128800, 128801],[94908, 94909],[383988, 383989]]#, [324, 325]]
# f, ax = plt.subplots(4,5,sharex=False,sharey=False)	
# f.set_size_inches(18.5, 10.5)
# for tmp_ax, gid in zip(ax, graph_id):
# 	Pic(tmp_ax, gid)
graph_id = good_id

# plt.tight_layout()
# plt.savefig(prefix + '2.eps',foramt = 'eps', dpi = 100, bbox_inches='tight')


from random import shuffle
file_list = os.listdir(prefix + 'att')
shuffle(file_list)

f, ax = plt.subplots(4,5, sharex=False,sharey=False)	
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=-0.1, hspace=0.5)
f.set_size_inches(15.5, 10.5)

# for xid, file in enumerate(file_list[:2]):
# 	qid = int(file.split('*')[0])
# 	graph_id[xid] = [qid / 4 * 4, qid / 4 * 4 + 1]
	
print graph_id
i = 0
for tmp_ax, gid in zip(ax, graph_id):
	Pic(tmp_ax, gid, i)
	i=i+1

# plt.tight_layout()
plt.savefig(prefix + '2_4.eps',foramt = 'eps', dpi = 200, bbox_inches='tight')
















#for num, gid in enumerate(graph_id):
# 	f, ax = plt.subplots(1,5,sharex=False,sharey=False)
# 	f.set_size_inches(18.5, 10.5)
# 	Pic(ax, gid)
# 	plt.tight_layout()
# 	plt.savefig(prefix + str(num) + '.eps',foramt = 'eps', dpi = 200)
