from collections import defaultdict
from itertools import accumulate
from library.span_graph import SpanGraph
from library.utils import total_count
from library.meet_in_the_middle import search as NP_search
from library.DP4F import search as DP_search


def hitcount_lower_bound(n, guaranteed_count, guaranteed_total_hitcount, accumulative_hit_counts, beta):
    thresholds = [(accumulative_hit_counts[j]+guaranteed_total_hitcount) / (guaranteed_count+j+(beta**2)*total_count(n))
                  for j in range(min(total_count(n)-guaranteed_count, len(accumulative_hit_counts)))]
    threshold = min(thresholds) if len(thresholds) else 0
    return threshold


def is_guaranteed(k, hit):
    return hit==k


def prune(n, k, hit_counts, beta, lowerbound=True, guarantee=True, binary=False):
    if guarantee:
        guaranteed = {span: count for span, count in hit_counts.items() if is_guaranteed(k, count)}
        guaranteed, guaranteed_total_hitcount = list(guaranteed.keys()), sum(guaranteed.values())
        hit_counts = {span: count for span, count in hit_counts.items() if not is_guaranteed(k, count)}
    else:
        guaranteed, guaranteed_total_hitcount = [], 0
    if lowerbound and not binary:
        accumulative_hit_counts = [0]+list(accumulate(sorted(hit_counts.values())))
        hit_counts = {span: count for span, count in hit_counts.items() if count>hitcount_lower_bound(n, len(guaranteed), guaranteed_total_hitcount, accumulative_hit_counts, beta)}
    return hit_counts, guaranteed, guaranteed_total_hitcount


def count_hits(trees, weights):
    hit_counts = defaultdict(lambda: 0)
    for (continues, disco), weight in zip(trees, weights):
        for span in continues+disco:
            hit_counts[span] += weight
    return hit_counts


def break_to_continues_disco(constituents):
    continues = [span for span in constituents if len(span)==2]
    disco = [span for span in constituents if len(span)==4]
    return [continues, disco]


def make_binary_by_righ_branching(span_selection, n):
    if len(span_selection[0])+len(span_selection[1]) == total_count(n):
        return span_selection
    continuous, discontinuous = span_selection
    continuous = [x for x in continuous if x[-1]-x[0]>1]
    for end in range(n, 0, -1):
        for begin in range(end-2, -1, -1):
            candid = (begin, end)
            if candid not in continuous and \
                all(SpanGraph.continuous_spans_are_compatible(candid, c) for c in continuous) and \
                all(SpanGraph.continuous_discontinuous_spans_are_compatible(candid, d) for d in discontinuous):
                continuous.append(candid)
                if len(continuous)+len(discontinuous) == n-1:
                    return [continuous+[(i,i+1) for i in range(n)], discontinuous]


def find_average_tree(trees, weights, beta=1, MAX_CANDID=float('inf'), MAX_LEN=float('inf'), DP=False,
        lowerbound=True, guarantee=True, binary=False, return_candid_count=False, log=False, MitM=True):
    assert not (DP and guarantee)
    assert not (MitM and DP)
    n = max([c[-1] for c in trees[0][0]+trees[0][1]])
    k = sum(weights)
    hit_counts = count_hits(trees, weights)
    hit_counts, guaranteed, guaranteed_total_hitcount = prune(n, k, hit_counts, beta, lowerbound=lowerbound, guarantee=guarantee, binary=binary)
    if return_candid_count:
        return len(hit_counts)
    if not len(hit_counts):
        return break_to_continues_disco(guaranteed)
    if n > MAX_LEN:
        if log:
            print(f'Missed [{len(n)} Words]')
        return None
    if len(hit_counts) > MAX_CANDID:
        if log:
            print(f'Missed [{len(hit_counts)} Candids]')
        return None
    search = DP_search if DP else NP_search
    span_selection = search(n, hit_counts, guaranteed, guaranteed_total_hitcount, beta=beta, binary=binary, MitM=MitM)
    span_selection = break_to_continues_disco(span_selection)
    if binary:
        span_selection = make_binary_by_righ_branching(span_selection, n)
    return span_selection