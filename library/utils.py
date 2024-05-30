def trivial_count(length):
    return length+1

def total_count(length):
    return 2*length-1

def f_beta(total_hitcount, pred_size, ref_size, beta=1):
    return (total_hitcount) / ((pred_size) + ref_size*(beta**2))

def objective_function(Q_total_hitcount, Q_size, alpha1, alpha2):
    return (Q_total_hitcount + alpha1) / (Q_size + alpha2)