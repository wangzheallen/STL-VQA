# Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering
This is the code for "Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering", The good practice in the VQA system such as pos-tag attention, structed triplet learning and triplet attention is very general and can be inserted into almost any visual and language task.

If you find the code useful, please cite the paper: 
##### Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering


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
| +POS-Att +Conv N-Gram +Triplet attention-Q+A + structed Learning Triplets | 67.5 | 60.3 |

Our full model performance

| Method | V7W Telling | VQA Test Standard | VQA Test Dev | VQA Test Dev Y/N | VQA Test Dev Num | VQA Test Dev Other |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| Ours | 68.2 | 69.6 | 69.7 | 81.9 | 44.3 | 64.7 |




# Quick Demo
Download the V7W telling feature shared on https://drive.google.com/open?id=1Hofquxw22j8soyjE0vuZqxcNuvJd-e9V
And run "CUDA_VISIBLE_DEVICES=0 python v7w.py"

# Data pre-processing

Download Visual7W: http://web.stanford.edu/~yukez/visual7w/
And glove: http://nlp.stanford.edu/data/wordvecs/glove.6B.zip  from https://github.com/stanfordnlp/GloVe

python data_preprocessing_7w.py
python prepro_7w.py

# Visualization
**Good Practice**: 

python comparisons_wacv.py

![goodpractice](https://github.com/wangzheallen/Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering/blob/master/goodpractice.png )
**Good Samples**: 

python draw_heat_new.py

![good samples](https://github.com/wangzheallen/Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering/blob/master/goodsample.png )
**Bad Samples**: 
![bad samples](https://github.com/wangzheallen/Structed-Triplets-Learning-with-Pos-tag-Guided-Attention-for-Visual-Question-Answering/blob/master/badsample.png )


# License

MIT 



