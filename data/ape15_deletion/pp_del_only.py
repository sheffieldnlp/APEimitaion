import numpy as np
import sys

ALIGN = sys.argv[1]
SRC = sys.argv[2]
TGT = sys.argv[3]
PE = sys.argv[4]
FEATS = sys.argv[5]

align = open(ALIGN).readlines()
src = open(SRC).readlines()
tgt = open(TGT).readlines()
pe = open(PE).readlines()
feats = open(FEATS).readlines()

f_align = []
f_src = []
f_tgt = []
f_pe = []
f_feats = []

i = 0
j = 0
for i in xrange(len(align)):
    if "I" in align[i] or "S" in align[i]:
        # skip sentence in feats file
        while feats[j].strip() != '':
            j += 1
        j += 1
        continue
    f_align.append(align[i].strip())
    f_src.append(src[i].strip())
    f_tgt.append(tgt[i].strip())
    f_pe.append(pe[i].strip())
    while feats[j].strip() != '':
        f_feats.append(feats[j].strip())
        j += 1
    f_feats.append(feats[j].strip())
    j += 1

np.savetxt(sys.argv[6], np.array(f_align), fmt='%s')
np.savetxt(sys.argv[7], np.array(f_src), fmt='%s')
np.savetxt(sys.argv[8], np.array(f_tgt), fmt='%s')
np.savetxt(sys.argv[9], np.array(f_pe), fmt='%s')
np.savetxt(sys.argv[10], np.array(f_feats), fmt='%s')
