import pickle
import os

def read_individuals(fold, paths):
    WORDS = f'{fold}.words.pkl'
    PREDICTION = f'{fold}.prediction.pkl'
    GOLD = f'{fold}.gold.pkl'

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
            for i in range(n):
                if (i, i+1) not in t[0]:
                    t[0].append((i, i+1))

    gold = pickle.load(open(os.path.join(paths[0],GOLD), 'rb'))
    return trees, gold