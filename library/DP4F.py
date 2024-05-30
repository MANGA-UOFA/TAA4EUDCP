from itertools import chain
from library.utils import trivial_count, total_count, objective_function
from copy import copy

def words_count_in_constituent(constituent):
    return sum([constituent[i+1]-constituent[i] for i in range(0, len(constituent), 2)])
    
def constituent_range(constituent):
    return chain(*[range(*r) for r in zip(constituent[::2], constituent[1::2])])

def cut_constituent(constituent, break_point):
    left = [p for p in constituent if p<break_point]
    if len(left)%2:
        left.append(break_point)
    right = [p for p in constituent if p>break_point]
    if len(right)%2:
        right.insert(0, break_point)
    return left, right

def unify_constituent(constituent):
    unified = []
    for point in constituent:
        if len(unified) and point==unified[-1]:
            unified.pop()
        else:
            unified.append(point)
    return unified

def search(n, hit_counts, guaranteed, guaranteed_total_hitcount, beta=1, binary=False, MitM=False):
    assert not MitM

    def distribution_search(DP_table, max_allowed, left, right):
        scenarios = []
        for left_constituent_count in range(max_allowed+1):
            right_constituent_count = max_allowed-left_constituent_count
            left_score = DP_table[left_constituent_count][tuple(left)][0]
            right_score = DP_table[right_constituent_count][tuple(right)][0]
            score = left_score + right_score
            scenarios.append((left_constituent_count, score))
        return max(scenarios, key=lambda x: x[1])
    
    def left_right_search(DP_table, constituent, max_allowed):
        best_score, best_left, best_right, best_left_balance = float('-inf'), None, None, 0
        for a in constituent_range(constituent):
            part1, rest1 = cut_constituent(constituent, a)
            for b in constituent_range(rest1):
                if b==rest1[0] and len(part1):
                    continue
                part2, rest2 = cut_constituent(rest1, b)
                for c in constituent_range(rest2):
                    if c==rest2[0]:
                        continue
                    part3, part4 = cut_constituent(rest2, c)
                    left = (part1+part3)
                    right = (part2+part4)
                    if len(left)>4 or len(right)>4:
                        continue
                    left_balance, score = distribution_search(DP_table, max_allowed, left, right)
                    if score > best_score:
                        best_score, best_left, best_right, best_left_balance = score, left, right, left_balance
        return best_score, best_left, best_right, best_left_balance

    def inclusive_exclusive_search(DP_table, constituent, max_allowed, inclusive_tuple):
        constituent_hit_count = hit_counts.get(constituent, 0)
        if words_count_in_constituent(constituent)==1:
            if max_allowed==0:
                return (0, None, None, 0, False), None
            if max_allowed==1:
                return (constituent_hit_count, None, None, 0, True), (0, None, None, 0)
            return (float('-inf'), None, None, 0, False), None
        inclusive_score, inclusive_left, inclusive_right, inclusive_left_balance = inclusive_tuple
        inclusive_score += constituent_hit_count
        exclusive_score, exclusive_left, exclusive_right, exclusive_left_balance = left_right_search(DP_table, constituent, max_allowed)
        inclusive = inclusive_score > exclusive_score
        score, left, right, left_balance = (inclusive_score, inclusive_left, inclusive_right, inclusive_left_balance) \
                        if inclusive else (exclusive_score, exclusive_left, exclusive_right, exclusive_left_balance)
        left, right = None if left is None else tuple(left), None if right is None else tuple(right)
        return (score, left, right, left_balance, inclusive), (exclusive_score, exclusive_left, exclusive_right, exclusive_left_balance)

    DP_table = {constituent_count: {} for constituent_count in range(total_count(n)+1)}
    for length in range(1, n+1):
        for left_length in range(length):
            right_length = length - left_length
            left_range = range(n+1-length) if left_length else [0]
            for left_b in left_range:
                left_e = left_b + left_length
                right_range = range(left_e+1, n+1-right_length) if left_length else range(n+1-right_length)
                for right_b in right_range:
                    right_e = right_b + right_length
                    constituent = (left_b, left_e, right_b, right_e) if left_length else (right_b, right_e)
                    exclusive_tuple = (float('-inf'), None, None, 0)
                    for constituent_count in range(total_count(n)+1):
                        best, exclusive_tuple = inclusive_exclusive_search(DP_table, constituent, constituent_count, exclusive_tuple)
                        DP_table[constituent_count][constituent] = best

    def retrive_children_constituents(DP_table, constituent, constituent_count):
        if constituent is None:
            return []
        if constituent not in DP_table[constituent_count]:
            assert words_count_in_constituent(constituent)==constituent_count
            return [(i, i+1) for i in constituent_range(constituent)]
        score, left, right, left_balance, inclusive = DP_table[constituent_count][constituent]
        left_constituents = retrive_children_constituents(DP_table, left, left_balance)
        right_constituents = retrive_children_constituents(DP_table, right, constituent_count-left_balance-inclusive)
        this_constituent = [constituent] if inclusive else []
        return this_constituent+left_constituents+right_constituents

    best_constituent_count = max([(i, DP_table[i][(0, n)]) for i in range(len(DP_table))],
                                 key=lambda x: x[1][0] if binary else objective_function(x[1][0], x[0], 0, (beta**2)*(total_count(n)))
                                )[0]
    span_selection = retrive_children_constituents(DP_table, (0, n), best_constituent_count)
    return span_selection