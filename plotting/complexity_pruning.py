from individuals.utils import read_individuals
import pickle
from library.ensemble import ensemble
import os
import numpy as np
from tqdm import tqdm
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
from matplotlib_dashboard import MatplotlibDashboard as MD

plt.figure(figsize=(15,7))
dashboard = MD([
    ['left' ,'right'],
], wspace=0.1)

paths = # a set of individuals
weights = [1]*len(paths)
STEP=1
MAX_LEN = 40
TEACHERS_COUNT = 5


words = [pickle.load(open(os.path.join(path,WORDS), 'rb')) for path in paths]
words = [list(map(' '.join, word)) for word in words]
trees, gold = read_individuals('test', paths)


x = np.arange(0, MAX_LEN+2*STEP, STEP)
def analyse(**args):
    candids = []
    mins = []
    maxs = []
    for length in tqdm(x):
        filtered_trees = [[] for tree in trees[:TEACHERS_COUNT]]
        for j, word in enumerate(words[0]):
            if len(word.split())>=length and len(word.split())<length+STEP:
                for i, tree in enumerate(trees[:TEACHERS_COUNT]):
                    filtered_trees[i].append(tree[j])
        if not len(filtered_trees[0]):
            candids.append(np.nan)
            mins.append( np.nan )
            maxs.append( np.nan )   
            continue
        
        candid_count = ensemble(filtered_trees, beta=1, weights=weights[:TEACHERS_COUNT], parallel=True, return_candid_count=True, progress_bar=False, **args)
        mins.append( min(candid_count) )
        maxs.append( max(candid_count) )
        step = sum(candid_count)/len(candid_count)
        candids.append(step)

    candids, mins, maxs = np.array(candids), np.array(mins), np.array(maxs)
    return mins, candids, maxs

mins, candids, maxs = analyse()
mins2, candids2, maxs2 = analyse(guarantee=False)

x = x+STEP/2
candids3 = (2**x)

x = np.insert(x, 0, 0)
mins = np.insert(mins, 0, 0)
candids = np.insert(candids, 0, 0)
maxs = np.insert(maxs, 0, 0)
mins2 = np.insert(mins2, 0, 0)
candids2 = np.insert(candids2, 0, 0)
maxs2 = np.insert(maxs2, 0, 0)
candids3 = np.insert(candids3, 0, 0)

labels = ['Without pruning', 'Theorem 5 only', 'Theorems 5 and 6']
colors= ['#f9b33a', '#008c66', '#003f5c']

l3, = dashboard['left'].plot(x[~np.isnan(candids)], candids[~np.isnan(candids)], label=labels[2], color=colors[2], linewidth=3)
l2, = dashboard['left'].plot(x[~np.isnan(candids2)], candids2[~np.isnan(candids2)], linestyle='-.', color=colors[1], label=labels[1], linewidth=3)
l1, = dashboard['left'].plot(candids3, linestyle='--', label=labels[0], linewidth=3, color=colors[0])
dashboard['left'].fill_between(x[~np.isnan(candids)], mins[~np.isnan(candids)], maxs[~np.isnan(candids)], alpha=.1, color=colors[2])
dashboard['left'].fill_between(x[~np.isnan(candids2)], mins2[~np.isnan(candids2)], maxs2[~np.isnan(candids2)], alpha=.1, color=colors[1])
dashboard['left'].set_ylabel('# candidate constituents', fontsize=25)
dashboard['left'].set_xlabel('n: length of the sentence\n(fixing # individuals to 5)', fontsize=25)
dashboard['left'].set_ylim((0, max(candids2[~np.isnan(candids2)])))
dashboard['left'].set_xlim((-STEP, MAX_LEN))
dashboard['left'].tick_params(labelsize=25)
dashboard['left'].tick_params(labelsize=25)
# dashboard['left'].legend(loc='upper right', bbox_to_anchor=(.5,1), fontsize=25)


x = np.arange(1, len(trees)+1, 1)
def analyse(**args):
    candids = []
    for i in tqdm(x):
        candid_count = ensemble(trees[:i], beta=1, weights=weights[:i], parallel=True, return_candid_count=True, progress_bar=False, **args)
        candids.append(np.mean(candid_count))
    return candids

candids = analyse()
candids2 = analyse(guarantee=False)

dashboard['right'].plot(x, candids, label=labels[2], linewidth=3, color=colors[2])
dashboard['right'].plot(x, candids2, linestyle='-.', label=labels[1], linewidth=3, color=colors[1])
# dashboard['right'].set_ylabel('Average # candidate constituents', fontsize=25)
dashboard['right'].set_xlabel('K: # individuals\n(for all sentence lengths)', fontsize=25)
dashboard['right'].set_ylim((0, max(candids2)))
dashboard['right'].set_xlim((1, len(trees)))
dashboard['right'].tick_params(labelsize=25)
dashboard['right'].tick_params(labelsize=25)
dashboard['right'].xaxis.set_ticks(x)
dashboard['right'].yaxis.tick_right()
plt.setp(dashboard['right'].get_yticklabels()[0], visible=False)
# dashboard['right'].legend(loc='upper left', fontsize=25)
dashboard['right'].legend([l1, l2, l3], labels, loc='lower right', fontsize=25, bbox_to_anchor=(1, .3))

props = dict(boxstyle='square', facecolor='white', pad=.29)
dashboard['left'].text(.022, .917, '(a)', transform=dashboard['left'].transAxes, fontsize=30, bbox=props)
dashboard['right'].text(.022, .917, '(b)', transform=dashboard['right'].transAxes, fontsize=30, bbox=props)

plt.savefig('complexity_pruning.pdf', bbox_inches='tight')