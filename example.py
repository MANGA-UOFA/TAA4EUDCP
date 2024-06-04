from evaluate import f1
from individuals.utils import read_individuals
from library.ensemble import ensemble

individuals, weights = zip(*[
    ('individuals/lassy/TN_LCFRS2024-01-27-00_54_23', 5.95),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_58_43', 5.82),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_55_50', 5.94),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_55_01', 5.52),
    ('individuals/lassy/TN_LCFRS2024-01-27-00_56_08', 1.38),
])
individuals, gold = read_individuals('test', individuals)
avgs = ensemble(individuals, weights=weights)
print(f1(avgs, gold))