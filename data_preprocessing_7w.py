
import json
import os
import argparse
import pdb
from nltk.tokenize import word_tokenize
import numpy as np

def main(params):
	train = []
	test = []
	imdir='v7w_%s.jpg'
	print 'Loading annotations and questions...'
	data = json.load(open('dataset_v7w_%s.json' %(params['data_set']), 'r'))["images"]

	train_txt = open("train_ques_ans.txt", "w")
	test_txt = open("test_ques_ans.txt", "w")

	for image in data:
		# print image.keys()
		for QA in image['qa_pairs']:
			correct_ans = QA['answer']
			question_id = QA['qa_id']
			image_path = imdir%(QA["image_id"])
			question = QA['question']
			# add correct answer
			if image['split'] == 'test':
				test_txt.write(str(question) + "\n")
				test_txt.write(str(correct_ans) + "\n")
				test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': correct_ans, 'ans': 1})
			else:
				train_txt.write(str(question) + "\n")
				train_txt.write(str(correct_ans) + "\n")
				train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': correct_ans, 'ans': 1})
			mc_ans = QA['multiple_choices']
			assert len(mc_ans) == 3
			# add wrong answers
			for wrong_ans in mc_ans:
				if image['split'] == 'test':
					test_txt.write(str(wrong_ans) + "\n")
					test.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': wrong_ans, 'ans': 0})
				else:
					train_txt.write(str(wrong_ans) + "\n")
					train.append({'ques_id': question_id, 'img_path': image_path, 'question': question, 'MC_ans': wrong_ans, 'ans': 0})

	train_txt.close()
	test_txt.close()
	json.dump(train, open('vqa_raw_train.json', 'w'))
	json.dump(test, open('vqa_raw_test.json', 'w'))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_set', default = 'telling',help = 'which data set, telling or pointing')
	args = parser.parse_args()
	params = vars(args) # convert to ordinary dict
	main(params)









