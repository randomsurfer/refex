import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import brewer2mpl

try:
    data_dir = sys.argv[1]
except IndexError:
    print 'Usage :: %s <data_dir>' % sys.argv[0]
    sys.exit(1)

all = {}
for file_name in os.listdir(data_dir):
    month_index = int(file_name.split('-')[0])
    data = np.loadtxt(open(data_dir+'/'+file_name,"rb"), delimiter=",")
    all[month_index] = data

one_person = np.zeros((12, 8))

for key in sorted(all.keys()):
    one_person[key-1] = all[key][20]

# tclass = (1. * tclass.T / tclass.T.sum()).T
#print one_person
one_person = one_person.T
# print one_person
# print sum(one_person[:,0])

# widths = np.array([1] * 12)
# gapd_widths = [i - .01 for i in widths]

for j in xrange(0, one_person.shape[1]):
    if sum(one_person[:, j]) != 0.0:
        one_person[:, j] = one_person[:, j] / sum(one_person[:, j])

# one_person = (1. * one_person.T / one_person.T.sum()).T
# print sum(one_person[:,0])

# copy = np.copy(one_person.T)
# copy = (1. * copy.T / copy.T.sum()).T
# copy = copy.T
# one_person = (1. * one_person / one_person.sum())
# one_person = (1. * one_person.T / one_person.T.sum()).T
# print one_person.shape
# print one_person[0]
colors = brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
ind = np.arange(12)
bottom = np.cumsum(one_person, axis=0)

# print one_person.shape, bottom.shape
plt.ylim([0.0,1.0])
plt.xticks([])
plt.yticks([])
plt.title("107")
plt.bar(ind, one_person[0], color=colors[0], width=1, edgecolor='none')

for j in xrange(1, one_person.shape[0]):
    plt.bar(ind, one_person[j], color=colors[j], bottom=bottom[j-1], width=[1]*12, edgecolor='none')
plt.show()
