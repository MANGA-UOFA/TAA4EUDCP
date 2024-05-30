from tqdm import tqdm
import concurrent.futures
from library.average_tree import find_average_tree
import time

def worker(i, ts, weights, beta, MAX_CANDID, MAX_LEN, lowerbound, guarantee, return_candid_count, DP, binary, MitM):
    avg = find_average_tree(ts,
        beta=beta,
        weights=weights,
        MAX_CANDID=MAX_CANDID,
        MAX_LEN=MAX_LEN,
        lowerbound=lowerbound,
        guarantee=guarantee,
        return_candid_count=return_candid_count,
        DP=DP,
        binary=binary,
        MitM=MitM,
    )
    return i, avg

def ensemble(references, beta=1, weights=None, MAX_CANDID=float('inf'), MAX_LEN=float('inf'),
             lowerbound=True, guarantee=True, return_candid_count=False, MitM=True,
             parallel=True, return_times=False, progress_bar=True, DP=False, binary=False):
    if weights is None:
        weights = [1]*len(references)
    assert len(weights)==len(references)
    assert not (parallel and return_times)
    bar = tqdm if progress_bar else lambda x, **args: x
    if parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(worker, i, [tree[i] for tree in references], weights, beta, MAX_CANDID, MAX_LEN, lowerbound, guarantee, return_candid_count, DP, binary, MitM) for i in range(len(references[0]))]
            avgs = [f.result() for f in bar(concurrent.futures.as_completed(futures), total=len(futures))]
            avgs = sorted(avgs, key=lambda x: x[0])
            avgs = list(map(lambda x: x[1], avgs))
    else:
        avgs = []
        times = []
        for i in bar(range(len(references[0]))):
            start = time.time()
            a = worker(i, [tree[i] for tree in references], weights, beta, MAX_CANDID, MAX_LEN, lowerbound, guarantee, return_candid_count, DP, binary, MitM)[1]
            end = time.time()
            times.append(end-start)
            avgs.append(a)
    if return_times:
        return avgs, times
    return avgs