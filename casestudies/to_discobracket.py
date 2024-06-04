from library.DP4F import words_count_in_constituent
from teachers_guide import teachers
import pickle
from tqdm import tqdm

def isin(constituent1, constituent2):
    if len(constituent1) <= 2:
        if len(constituent2) <= 2:
            return (constituent1[0] >= constituent2[0]) and (constituent1[1] <= constituent2[1])
        else:
            return any(isin(constituent1, constituent2[i:i+2]) for i in range(0, len(constituent2), 2))
    else:
        return all(isin(constituent1[i:i+2], constituent2) for i in range(0, len(constituent1), 2))

class Node:
    def __init__(self, label, span):
        self.label = label
        if words_count_in_constituent(span)==1:
            self.label = str(span[0])
        self.span = span
        self.children = []

    def attach_word(self, word):
        self.label += '='+word

    def add(self, span, label):
        assert isin(span, self.span)
        for child in self.children:
            if isin(span, child.span):
                return child.add(span, label)
                break
        else:
            node = Node(label, span)
            self.children.append(node)
            return node

    def __str__(self):
        if words_count_in_constituent(self.span) == 1:
            return self.label
        return f'({self.label}'+(' ' if len(self.children) else '')+' '.join(map(str, self.children))+')'

def build_tree(spans, words, include_labels=False):
    spans = sorted(spans, key=lambda x: words_count_in_constituent(x[:-1] if include_labels else x), reverse=True)
    if include_labels:
        spans, labels = [s[:-1] for s in spans], [s[-1] for s in spans]
    else:
        labels = [' ']+['.' if words_count_in_constituent(span)==1 else 'O' for span in spans[1:]]
    root = Node(labels[0], spans[0])
    for span, label in zip(spans[1:], labels[1:]):
        node = root.add(span, label)
        if words_count_in_constituent(span)==1:
            node.attach_word(words[span[0]])
    return root

# How to use:
# str(build_tree(constituents, words)
# constituents is a flat list of all continuous and discontinuous constituents
# words is a list of words

