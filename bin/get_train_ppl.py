from math import exp
from sys import argv

fname = argv[1]
with open(fname, 'r') as f:
    for line in f.readlines():
        if line.startswith("total_loss_ema"):
            last_line = line

lm_loss_wgt = last_line.split(', ')[6].strip()
lm_loss_wgt = float(lm_loss_wgt.split()[-1])

print(round(exp(lm_loss_wgt), 2))
