from TNLCFRS.parser.helper.metric import discoF1
import pickle
from library.ensemble import ensemble
from teachers_guide import teachers
import os
import numpy as np
from tqdm import tqdm
import json

f1_means = json.load(open('analysis_incremental_data/f1_means.json', 'r'))
f1_stds = json.load(open('analysis_incremental_data/f1_stds.json', 'r'))
f1c_means = json.load(open('analysis_incremental_data/f1c_means.json', 'r'))
f1c_stds = json.load(open('analysis_incremental_data/f1c_stds.json', 'r'))
f1d_means = json.load(open('analysis_incremental_data/f1d_means.json', 'r'))
f1d_stds = json.load(open('analysis_incremental_data/f1d_stds.json', 'r'))
f1_means, f1_stds = np.array(f1_means), np.array(f1_stds)
f1c_means, f1c_stds = np.array(f1c_means), np.array(f1c_stds)
f1d_means, f1d_stds = np.array(f1d_means), np.array(f1d_stds)
f1_means *= 100
f1_stds *= 100
f1c_means *= 100
f1c_stds *= 100
f1d_means *= 100
f1d_stds *= 100

import matplotlib.pyplot as plt
from matplotlib_dashboard import MatplotlibDashboard as MD
import matplotlib.ticker as ticker


def plot(ax, means, stds, color, label, marker):
    labels = [str(i) for i in range(1, len(means)+1)]
    ax.plot(labels, means, linestyle=':', color=color)
    ax.errorbar(labels, means, fmt='none', yerr=stds, capsize=7, elinewidth=2.5, capthick=1.5, color=color)
    style = ax.scatter(labels, means, marker=marker, s=120, label=label, color=color)
    return ax, style

top_range = 40.5, 56.4
but_range = 1,11
fig = plt.figure(figsize=(15,7))
dashboard = MD([
    ['left' ,'right'],
    ['leftb' ,'rightb'],
], wspace=0.1,
height_ratios=[top_range[1]-top_range[0], but_range[1]-but_range[0]])
leftb = fig.add_subplot(dashboard['leftb'])
left = fig.add_subplot(dashboard['left'], sharex=leftb)
rightb = fig.add_subplot(dashboard['rightb'], sharey=leftb)
right = fig.add_subplot(dashboard['right'], sharex=rightb, sharey=leftb)
plt.setp(rightb.get_yticklabels(), visible=False)
plt.setp(right.get_yticklabels(), visible=False)
plt.setp(left.get_xticklabels(), visible=False)
plt.setp(right.get_xticklabels(), visible=False)

props = dict(boxstyle='square', facecolor='white', pad=.29)
leftb.text(.875, .11, '(a)', transform=leftb.transAxes, fontsize=30, bbox=props)
rightb.text(.022, .11, '(b)', transform=rightb.transAxes, fontsize=30, bbox=props)


left, line2 = plot(left, f1c_means, f1c_stds, color='#93af39', label='CF1', marker='^')
left, line1 = plot(left, f1_means, f1_stds, color='#007e73', label='F1', marker='o')
left.set_ylim(*top_range)
leftb, line3 = plot(leftb, f1d_means, f1d_stds, color='#003f5c', label='DF1', marker='s')
leftb.set_ylim(*but_range)
leftb.spines['top'].set_visible(False)
left.spines['bottom'].set_visible(False)
left.tick_params(labeltop=False)
left.tick_params(labelsize=25)
left.xaxis.set_ticks_position('none') 
leftb.xaxis.tick_bottom()
leftb.tick_params(labelsize=25)
left.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
leftb.yaxis.set_major_locator(ticker.MultipleLocator(base=5))
d = .015
kwargs = dict(transform=left.transAxes, color='k', clip_on=False)
left.plot((1-d, 1+d), (-d, +d), **kwargs)
left.plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=leftb.transAxes)
b_slash_range = (1-1.5*d, 1+d*1.9)
leftb.plot((1-d, 1+d), b_slash_range, **kwargs)
leftb.plot((-d, +d), b_slash_range, **kwargs)
plt.tight_layout()
grid_color = '#d3d3d3'
left.grid(axis='x', color=grid_color)
leftb.grid(axis='x', color=grid_color)
left.legend([line2, line1, line3], [r'$F_1^{cont}$', r'$F_1^{overall}$', r'$F_1^{disco}$'], loc='lower right', fontsize=25, bbox_to_anchor=(1, -.15))
leftb.set_xlabel('# individuals in the ensemble', fontsize=25)

fold = 'test'
WORDS = f'{fold}.words.pkl'
PREDICTION = f'{fold}.prediction.pkl'
GOLD = f'{fold}.gold.pkl'
source='lassy_0_inc_10_20_testsorted'
paths = list(teachers[source].keys())
weights = list(teachers[source].values())


words = [pickle.load(open(os.path.join(path,WORDS), 'rb')) for path in paths]
words = [list(map(' '.join, word)) for word in words]
reorders = [[word.index(gword) for gword in words[0]] for word in words]
trees = [pickle.load(open(os.path.join(path,PREDICTION), 'rb')) for path in paths]
trees = [[tree[i] for i in reorder] for tree, reorder in zip(trees, reorders)]
for tree in trees:
    for t in tree:
        n = max([c[-1] for c in t[0]+t[1]])
        if (0, n) not in t[0]:
            t[0].append((0, n))
gold = pickle.load(open(os.path.join(paths[0],GOLD), 'rb'))

f1s = json.load(open('best_to_worst_incremental_data/f1s.json', 'r'))
f1cs = json.load(open('best_to_worst_incremental_data/f1cs.json', 'r'))
f1ds = json.load(open('best_to_worst_incremental_data/f1ds.json', 'r'))

def evaluate_teacher(tree):
    f1_metric = discoF1()
    f1_metric(tree, gold)
    f1_d, prec_d, recall_d = f1_metric.corpus_uf1_disco
    f1_c, prec_c, recall_c = f1_metric.corpus_uf1
    f1 = f1_metric.all_uf1 if type(f1_metric.all_uf1) is float else f1_metric.all_uf1[0]
    return f1*100, f1_c*100, f1_d*100

tf1s, tf1cs, tf1ds = zip(*[evaluate_teacher(tree) for tree in trees])
f1s = np.array(f1s)
f1cs = np.array(f1cs)
f1ds = np.array(f1ds)
f1s *= 100
f1cs *= 100
f1ds *= 100


def plot(ax, means, base, color, label, marker):
    labels = [str(i) for i in range(1, len(means)+1)]
    ax.plot(labels, means, linestyle=':', color=color)
    style = ax.scatter(labels, means, marker=marker, s=120, label=label, color=color)
    base_style = ax.scatter(labels, base, marker=marker, s=120, label=label, color=color, facecolors='none')
    return ax, style, base_style

right, _, line1 = plot(dashboard['right'], f1s, tf1s, color='#007e73', label='F1', marker='o')
right, _, line2 = plot(right, f1cs, tf1cs, color='#93af39', label='CF1', marker='^')
right.set_ylim(*top_range)
rightb, _, line3 = plot(dashboard['rightb'], f1ds, tf1ds, color='#003f5c', label='DF1', marker='s')
rightb.set_ylim(*but_range)
rightb.yaxis.set_visible(False)
right.yaxis.set_visible(False)
right.spines['bottom'].set_visible(False)
rightb.spines['top'].set_visible(False)
right.tick_params(labeltop=False)
right.tick_params(labelsize=25)
right.xaxis.set_ticks_position('none') 
rightb.xaxis.tick_bottom()
rightb.tick_params(labelsize=25)
kwargs = dict(transform=right.transAxes, color='k', clip_on=False)
right.plot((1-d, 1+d), (-d, +d), **kwargs)
right.plot((-d, +d), (-d, +d), **kwargs)
kwargs.update(transform=rightb.transAxes)
rightb.plot((1-d, 1+d), b_slash_range, **kwargs)
rightb.plot((-d, +d), b_slash_range, **kwargs)
plt.tight_layout()
grid_color = '#d3d3d3'
right.grid(axis='x', color=grid_color)
rightb.grid(axis='x', color=grid_color)
# right.yaxis.tick_right()
# rightb.yaxis.tick_right()
right.legend([line2, plt.scatter([],[],alpha=0), line1, plt.scatter([],[],alpha=0), line3, plt.scatter([],[],alpha=0)], [',', '', ',', '', " Individual", ' performance'], loc='lower left', fontsize=25, bbox_to_anchor=(.005, -.17), ncol=3, handletextpad=-.6, columnspacing=-.6, labelspacing=0.)
# right.legend([line1, line3, line2], [',', '       performance', ",  Individual      "], loc='lower left', fontsize=25, bbox_to_anchor=(0, -.19), ncol=2, handletextpad=-.7, columnspacing=-8.8, labelspacing=0.)
rightb.set_xlabel('# individuals in the ensemble', fontsize=25)

plt.savefig('individual_count_analysis.pdf', bbox_inches='tight')