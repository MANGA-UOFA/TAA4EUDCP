from collections import defaultdict
from more_itertools import distinct_permutations as idp
import numpy as np
from library.span_graph import SpanGraph
from library.utils import total_count, objective_function


def left_step(span_graph, left_proportion=1/2):
    n = int(left_proportion*span_graph.size)
    scores = defaultdict(lambda: [(-1, None) if i else (0, '0'*n) for i in range(n+1)])
    for i in range(1, n+1):
        for key in idp('0'*(n-i)+'1'*i):
            key = ''.join(key)
            selection = (np.array(list(key)) == '1').astype(int)
            whole_key = ''.join(map(str, selection))
            the1 = np.nonzero(selection)[0][0]
            the1score = span_graph.scores[the1]

            # w/ the1
            connected2the1 = (span_graph.matrix[the1, :n] * selection).astype(int)
            connected2the1[the1] = 0
            connected2the1_key = ''.join(map(str, connected2the1))
            for j in range(connected2the1.sum()+1):
                score, bestsubset = scores[connected2the1_key][j]
                if score >= 0:
                    scores[whole_key][j+1] = (score+the1score, bestsubset[:the1]+'1'+bestsubset[the1+1:])

            # w/o the1
            selection_wo_the1 = selection.copy()
            selection_wo_the1[the1] = 0
            wo_the1_key = ''.join(map(str, selection_wo_the1))
            for j, (score, bestsubset) in enumerate(scores[wo_the1_key]):
                if score > scores[whole_key][j][0]:
                    scores[whole_key][j] = (score, bestsubset)

    return scores, n


def right_step(span_graph, left_scores, left_size):
    right_size = span_graph.size - left_size
    scores = [(-1, None) if i else (0, '0'*span_graph.size) for i in range(span_graph.size+1)]
    for i in range(right_size+1):
        for key in idp('0'*(right_size-i) + '1'*i):
            key = ''.join(key)
            selection = np.array([0]*left_size+list(key))=='1'
            if span_graph.is_complete(selection):
                right_score = span_graph.total_hitcounts(selection)
                left_fullyconnected_subset = np.all(span_graph.matrix[selection], axis=0)[:left_size].astype(int)
                left_fullyconnected_key = ''.join(map(str, left_fullyconnected_subset))
                for j, (score, bestsubset) in enumerate(left_scores[left_fullyconnected_key]):
                    if score >= 0 and score+right_score > scores[j+i][0]:
                        scores[j+i] = (score+right_score, bestsubset+key)
    return scores


def meet_in_the_middle(span_graph, left_proportion=1/2):
    return right_step(span_graph, *left_step(span_graph, left_proportion=left_proportion))


def search(n, hit_counts, guaranteed, guaranteed_total_hitcount, beta=1, binary=False, MitM=True):
    span_graph = SpanGraph(hit_counts)
    spans = span_graph.spans
    answers = list(filter(lambda x: x[0]>=0, meet_in_the_middle(span_graph, left_proportion=1/2 if MitM else 0)))
    if binary:
        answer = max(answers, key=lambda x: x[0])[1]
    else:
        answer = max(answers, key=lambda x: objective_function(x[0], x[1].count('1'), guaranteed_total_hitcount, len(guaranteed)+(beta**2)*(total_count(n))))[1]
    span_selection = [span for i,span in enumerate(spans) if answer[i]=='1']+(guaranteed)
    return span_selection