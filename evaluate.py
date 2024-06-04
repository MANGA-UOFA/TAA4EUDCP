import sys
sys.path.insert(0, 'TN-LCFRS/parser')
from helper.metric import discoF1


def f1(prediction, gold):
    f1computer = discoF1()
    f1computer(prediction, gold)
    f1_d, prec_d, recall_d = f1computer.corpus_uf1_disco
    f1_c, prec_c, recall_c = f1computer.corpus_uf1
    f1 = f1computer.all_uf1 if type(f1computer.all_uf1) is float else f1computer.all_uf1[0]
    metrics = {
        'overal': f1,
        'continuous': f1_c,
        'discontinuous': f1_d
    }
    return metrics