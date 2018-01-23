# Structured Triplet Learning with Pos-tag Guided Attention for Visual Question Answering
This is the code for "Structured Triplet Learning with Pos-tag Guided Attention for Visual Question Answering, WACV 2018 (Zhe Wang, Xiaoyi Liu, Liangjian Chen, Limin Wang, Yu Qiao, Xiaohui Xie, Charless Fowlkes)", The good practice in the VQA system such as pos-tag attention, structured triplet learning and triplet attention is very general and can be inserted into almost any visual and language task.

If you find the code useful, please cite the paper: 
##### Structured Triplet Learning with Pos-tag Guided Attention for Visual Question Answering WACV 2018 (Zhe Wang, Xiaoyi Liu, Liangjian Chen, Limin Wang, Yu Qiao, Xiaohui Xie, Charless Fowlkes)
If you have feedback for the code, please contact:
##### buptwangzhe2012 at gmail dot com

# Performance

Below is the step by step effectiveness verification of our method, note to speed up the verification, we use the 7by7 feature instead of 14by14 feature

| Method | V7W | VQA validation |
| ------ | ------ | ------ |
| Our Baseline | 65.6 | 58.3 |
| +POS tag guided attention (POS-Att) | 66.3 | 58.7 |
| +Convolutional N-Gram (Conv N-Gram) |  66.2 | 59.3 |
| +POS-Att +Conv N-Gram | 66.6 | 59.5 |
| +POS-Att +Conv N-Gram +Triplet attention-Q | 66.8 | 60.1 |
| +POS-Att +Conv N-Gram +Triplet attention-A | 67.0 | 60.1 |
| +POS-Att +Conv N-Gram +Triplet attention-Q+A | 67.3 | 60.2 |
| +POS-Att +Conv N-Gram +Triplet attention-Q+A + structured Learning Triplets | 67.5 | 60.3 |

Our full model performance

| Method | V7W Telling | VQA Test Standard | VQA Test Dev | VQA Test Dev Y/N | VQA Test Dev Num | VQA Test Dev Other |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Ours | 68.2 | 69.6 | 69.7 | 81.9 | 44.3 | 64.7 |


# Pre-requisite

tensorflow, torch, pandas, h5py, ipdb, cv2, pdb, spacy, sklearn, matplotlib, PIL, nltk

# Quick Demo
Download the V7W telling feature shared on https://drive.google.com/open?id=1Hofquxw22j8soyjE0vuZqxcNuvJd-e9V
And run "CUDA_VISIBLE_DEVICES=0 python v7w.py"

# Data pre-processing

Download Visual7W: http://web.stanford.edu/~yukez/visual7w/
And glove: http://nlp.stanford.edu/data/wordvecs/glove.6B.zip  from https://github.com/stanfordnlp/GloVe
Download: https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7

python data_preprocessing_7w.py --data_set telling

python prepro_7w.py

th prepro_img_residule.lua

# Visualization
**Architecture**: 

![Architecture](https://github.com/wangzheallen/STL_VQA/blob/master/architecture.png )


**Good Practice**: 

python comparisons_wacv.py

![goodpractice](https://github.com/wangzheallen/STL_VQA/blob/master/goodpractice.png )
**Good Samples**: 

python draw_heat_new.py

![good samples](https://github.com/wangzheallen/STL_VQA/blob/master/goodsample.png )
**Bad Samples**: 
![bad samples](https://github.com/wangzheallen/STL_VQA/blob/master/badsample.png )


# License

MIT 



