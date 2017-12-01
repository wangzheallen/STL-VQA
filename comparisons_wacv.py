"""
Bar chart demo with pairs of bars grouped for easy comparison.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# n_groups = 3



# plt.bar(range(len(data)), data)
# plt.show()
#plt.xlabel('Data Imbalance')
#plt.ylabel('Accuracy')

# plt.title('Comparison Between Different Data Imbalance Rate')
# #plt.legend()
#

# plt.show()
#plt.savefig(u't.png')
# fig, axes = plt.subplots(2,2)
# axes.yaxis.set_major_locator(MaxNLocator(prune='upper'))
# plt.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(10,5))

data = [67.1, 67.3, 66.9]
index = np.arange(len(data))
ax1.bar(index, data, width = 0.3,color = 'b')
ax1.set_xticks(index)
ax1.set_xticklabels(('1:1', '1:2', '1:3'))
ax1.set_ylim(66.5, 67.5)
ax1.tick_params(axis='y', colors='black')
ax1.set_ylabel('accuracy',color = 'black')
# plt.xlabel('imbalance')
ax1.set_title('(a) Imbalance', y = -0.3)

# plt.subplots_adjust(right = 0.1)
data = [66.7,67.3,67.1,66.8,66.6]
index = np.arange(len(data))
ax2.bar(index, data, width = 0.3,color = 'b')
ax2.set_xticks(index)
ax2.set_xticklabels(('9','18','36','72','108'))
ax2.set_ylim(66.5, 67.5)
ax2.tick_params(axis='y', colors='black')
ax2.set_title('(b) Batch Size', y = -0.3)
ax2.set_ylabel('accuracy',color = 'black')

data = [65.9,66.5,67.3,67.3]
index = np.arange(len(data))
ax3.bar(index, data, width = 0.3,color = 'b')
ax3.set_xticks(index)
ax3.set_xticklabels(('1-gram', '2-gram', '3-gram', '4-gram'))
ax3.set_ylim(65, 68)
ax3.tick_params(axis='y', colors='black')
ax3.set_ylabel('accuracy',color= 'black')
ax3.set_title('(c) Convolutional N-gram', y = -0.3)


data1 = [66.3, 52.8, 50.7, 67.3]
data2 = [50 + (i - 2.5) / 12.5 * 18.0 for i in [14,3,4,9]]
print data2
index = np.arange(len(data1))
tax = ax4.twinx()

ax4.bar(index - 0.15, data1, width = 0.3,color = 'b')
ax4.bar(index + 0.15, data2, width = 0.3,color = 'black')
ax4.set_xticks(index)
ax4.set_xticklabels(('No BN','Data', 'QI', 'QIA'))
ax4.set_ylim(50, 68)
ax4.set_ylabel('accuracy',color = 'black')
ax4.set_title('(d) Batch Normalization', y = -0.3)
ax4.tick_params(axis='y', colors='black')
tax.set_ylabel('epoch',color = 'black')
tax.set_ylim(2,16)
tax.tick_params(axis='y', colors='black')

data = [67.3,67.5,67.4,67.2,67.0]
index = np.arange(len(data))
ax5.bar(index, data, width = 0.3,color = 'b')
ax5.set_xticks(index)
ax5.set_xticklabels(('0.1','0.2','0.3','0.4','0.5'))
ax5.set_ylim(66.7, 67.8)
ax5.tick_params(axis='y', colors='black')
ax5.set_title('(e) lambda2', y = -0.3)
ax5.set_ylabel('accuracy',color = 'black')


data = [67.3,67.5,67.4]
index = np.arange(len(data))
ax6.bar(index, data, width = 0.3,color = 'b')
ax6.set_xticks(index)
ax6.set_xticklabels(('0.1','0.2','0.3'))
ax6.set_ylim(67.2, 67.6)
ax6.tick_params(axis='y', colors='black')
ax6.set_title('(f) margin', y = -0.3)
ax6.set_ylabel('accuracy',color = 'black')



plt.tight_layout()
# plt.show()
plt.savefig('comparisons_wacv.eps',format = 'eps',dpi = 1000)
