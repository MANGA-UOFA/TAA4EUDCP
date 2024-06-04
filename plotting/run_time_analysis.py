import pickle
from library.ensemble import ensemble
import os
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
from matplotlib_dashboard import MatplotlibDashboard as MD
from collections import defaultdict
import numpy as np

paths = #individuals
weights = #individual weights

words = [pickle.load(open(os.path.join(path,WORDS), 'rb')) for path in paths]
words = [list(map(' '.join, word)) for word in words]
trees = read_individuals('test', paths)

def process(len_time):
    data = defaultdict(lambda: [])
    for l, time in len_time:
        data[l].append(time)
    out = [(l, min(data[l]), np.mean(data[l]), max(data[l])) for l in range(1, 6) if l in data]
    for len_b in range(6, max(data.keys())+1, 5):
        d = sum([data[i] for i in range(len_b, len_b+5)], [])
        if not len(d):
            continue
        out.append((len_b+1.5, min(d), np.mean(d), max(d)))
    d = sum([data[i] for i in range(38, 43)], [])
    if len(d):
        out.append((40, min(d), np.mean(d), max(d)))
    x, mins, means, maxs = zip(*out)
    x = list(x)
    mins = list(mins)
    means = list(means)
    maxs = list(maxs)
    return x, mins, means, maxs

avgs, times = ensemble(trees, weights=weights, parallel=False, guarantee=False, DP=True, MAX_LEN=17, return_times=True, MitM=False)
lens = [len(w.split()) for w in words[0]]
avgs, lens, times = zip(*[(a, w, t) for a, w, t in zip(avgs, lens, times) if a is not None])
DP4F = sorted(zip(lens, times), key=lambda x: x[0])
DP4F_x, DP4F_mins, DP4F_means, DP4F_maxs = process(DP4F)

avgs, times = ensemble(trees, weights=weights, parallel=False, MitM=False, MAX_LEN=25, return_times=True)
lens = [len(w.split()) for w in words[0]]
avgs, lens, times = zip(*[(a, w, t) for a, w, t in zip(avgs, lens, times) if a is not None])
woMitM = sorted(zip(lens, times), key=lambda x: x[0])
woMitM_x, woMitM_mins, woMitM_means, woMitM_maxs = process(woMitM)

avgs, times = ensemble(trees, weights=weights, parallel=False, MitM=False, MAX_CANDID=20, return_times=True)
lens = [len(w.split()) for w in words[0]]
avgs, lens, times = zip(*[(a, w, t) for a, w, t in zip(avgs, lens, times) if a is not None])
woMitMm = sorted(zip(lens, times), key=lambda x: x[0])
woMitMm_x, woMitMm_mins, woMitMm_means, woMitMm_maxs = process(woMitMm)
up_bar = max(woMitM_means)
woMitM_count = len(woMitM_means)
for x, m in zip(woMitMm_x, woMitMm_mins):
    if x > max(woMitM_x):
        woMitM_x.append(x)
        woMitM_mins.append(m)
        woMitM_maxs.append(up_bar)

avgs, times = ensemble(trees, weights=weights, parallel=False, return_times=True)
lens = [len(w.split()) for w in words[0]]
avgs, lens, times = zip(*[(a, w, t) for a, w, t in zip(avgs, lens, times) if a is not None])
MitM = sorted(zip(lens, times), key=lambda x: x[0])
MitM_x, MitM_mins, MitM_means, MitM_maxs = process(MitM)

labels = ['DP solution', 'w/o meet-in-the-middle', 'w/ meet-in-the-middle']
colors= ['#ff6060', '#ac64b5', '#006fa2']

plt.figure(figsize=(15,7))
dashboard = MD([
    ['left'],
], wspace=0.1)


l1, = dashboard['left'].plot(DP4F_x, DP4F_means, linestyle='--', color=colors[0], label=labels[0], linewidth=3)
l2, = dashboard['left'].plot(woMitM_x[:woMitM_count], woMitM_means, linestyle='-.', color=colors[1], label=labels[1], linewidth=3)
l3, = dashboard['left'].plot(MitM_x, MitM_means, label=labels[2], color=colors[2], linewidth=3)
dashboard['left'].fill_between(MitM_x, MitM_mins, MitM_maxs, alpha=.1, color=colors[2])
dashboard['left'].fill_between(woMitM_x, woMitM_mins, woMitM_maxs, alpha=.1, color=colors[1])
dashboard['left'].fill_between(DP4F_x, DP4F_mins, DP4F_maxs, alpha=.1, color=colors[0])
dashboard['left'].set_ylabel('Inference time (s)', fontsize=25)
dashboard['left'].set_xlabel('n: length of the sentence\n(fixing # individuals to 5)', fontsize=25)
dashboard['left'].set_yscale('log')
dashboard['left'].set_ylim((10**-5, up_bar))
dashboard['left'].set_xlim((2, 40))
dashboard['left'].tick_params(labelsize=25)
dashboard['left'].tick_params(labelsize=25)
dashboard['left'].legend(fontsize=25)#, loc='lower right', bbox_to_anchor=(.5,1)

plt.savefig('run_time.pdf', bbox_inches='tight')