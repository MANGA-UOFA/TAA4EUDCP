# TAA4EUDCP

An official implementation for the paper "Tree-Averaging Algorithms for Ensemble-Based Unsupervised Discontinuous Constituency Parsing."

## Usage

The `ensemble` function takes multiple parameters to control its behavior, providing flexibility in its application. Below is a guide on how to use this function effectively.

### Function Signature

```python
def ensemble(references, beta=1, weights=None, MAX_CANDID=float('inf'), MAX_LEN=float('inf'),
             lowerbound=True, guarantee=True, return_candid_count=False, MitM=True,
             parallel=True, return_times=False, progress_bar=True, DP=False, binary=False):
```

### Parameters

- **references**: A list of references (trees) to be combined.
- **beta**: To indicate f_beta as the ensemble objective, default is 1.
- **binary**: A boolean indicating whether to make the results binary. Default is `False`.
- **weights**: A list of weights corresponding to each reference. If not provided, all weights are set to 1.
- **MAX_CANDID**: The maximum number of candidates to consider. Default is infinity.
- **MAX_LEN**: The maximum length of the sequence to consider. Default is infinity.
- **lowerbound**: A boolean indicating whether to apply a lower bound during the pruning step. Default is `True`.
- **guarantee**: A boolean indicating whether to exclude guaranteed spans from the search. Default is `True`.
- **parallel**: A boolean indicating whether to execute in parallel. Default is `True`.
- **return_times**: A boolean indicating whether to return the execution times for each sentence. Default is `False`.
- **return_candid_count**: A boolean indicating whether to return the count of candidates. Default is `False`.
- **DP**: A boolean indicating whether to use dynamic programming. Otherwise, follows the general search. Default is `False`.
- **MitM**: A boolean indicating whether to use the meet-in-the-middle approach. Default is `True`.
- **progress_bar**: A boolean indicating whether to display a progress bar. Default is `True`.

### Returns

The function returns the average tree representations. If `return_times` is set to `True`, it also returns the execution times for each sentence.

### Example Usage

```python
from evaluate import f1
from individuals.utils import read_individuals
from library.ensemble import ensemble

# Reading individual references and their weights and the gold standard for evaluation
individuals, weights = zip(*[
    ('individuals/lassy/TN_LCFRS2024-01-27-00_54_23', 5.95),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_58_43', 5.82),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_55_50', 5.94),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_55_01', 5.52),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_56_08', 1.38),
])
individuals, gold = read_individuals('test', individuals)

# Generating the average tree
avgs = ensemble(individuals, weights=weights)

# Evaluating the results
print(f1(avgs, gold))
```

<a href="https://shayeghb.github.io/"><img src="https://shayeghb.github.io/img/favicon.png" style="background-color:red;"/></a>
