import wqe
import numpy as np
import sys
import os
from imitation.state import State
from copy import deepcopy
from sklearn.metrics import hamming_loss
from sklearn.metrics import f1_score
from eval_wqe import score_wmt_plain


TRAIN_DIR = '../data/wqe16/train'
DEV_DIR = '../data/wqe16/dev'
TEST_DIR = '../data/wqe16/test'


def load_data(_dir, mode, test=False):
    with open(os.path.join(_dir, mode + '.mt')) as f:
        mt = [line.split() for line in f.readlines()]
    if test:
        tags = [['UNK'] * len(sent) for sent in mt]
    else:
        with open(os.path.join(_dir, mode + '.tags')) as f:
            tags = [line.split() for line in f.readlines()]
    sent_lens = [len(sent) for sent in mt]
    #print sent_lens
    feat_tensor = get_features(_dir, mode, sent_lens)
    instances = [wqe.WQEInstance(x, y, feats) for x, y, feats in zip(mt, tags, feat_tensor)]
    return instances


def get_features(_dir, mode, sent_lens):
    feat_tensor = []
    cols = [0, 1, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    with open(os.path.join(_dir, mode + '.feats.baseline')) as f:
        i = 0
        feat_matrix = []
        for line in f:
            feat_vector = line.split()
            feat_matrix.append(np.array(feat_vector, dtype=object))
            if len(feat_matrix) == sent_lens[i]:
                feat_matrix = np.array(feat_matrix, dtype=object)
                to_float = feat_matrix[:, cols].astype(float)
                feat_matrix[:, cols] = to_float
                feat_tensor.append(deepcopy(feat_matrix))
                feat_matrix = []
                i += 1
            
    return feat_tensor


def get_dev_metrics(results, test_instances):
    f_bad = []
    f_ok = []
    f_scores = []
    hamm_scores = []
    for p, t in zip(results, test_instances):
        tags = t.output.tags
        pred = p.split()
        print pred
        print tags
        print ''
        hamm = hamming_loss(pred, tags) * len(p)
        f1_bad = f1_score(pred, tags, labels=['OK', 'BAD'], pos_label='BAD')
        f1_ok = f1_score(pred, tags, labels=['OK', 'BAD'], pos_label='OK')
        hamm_scores.append(hamm)
        f_scores.append(f1_bad * f1_ok)
        f_bad.append(f1_bad)
        f_ok.append(f1_ok)
    return np.mean(f_scores), np.mean(f_bad), np.mean(f_ok), np.mean(hamm_scores)
    

train_instances = load_data(TRAIN_DIR, 'train')#[:100]
dev_instances = load_data(DEV_DIR, 'dev')
test_instances = load_data(TEST_DIR, 'test', test=True)

import random
import numpy
random.seed(13)           
numpy.random.seed(13)

model = wqe.WQE()

# set the params
params = wqe.WQE.params()
# Setting this to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
params.iterations = int(sys.argv[1])
params.learningParam = float(sys.argv[2])
params.samplesPerAction = 1

model.train(train_instances, "temp", params)
# TODO: This is a hack. Probably the state initialization should happen in the beginning of predict

#state = State()
#model.predict(test_instances, state)
#print model.predict(test_instances[0], state).tags
#print test_instances[0].output.tags
#state = State()
#print model.predict(test_instances[0], state).tags
results_dev = []
preds_dev = []
for instance in dev_instances:
    state = State()
    try:
        pred = model.predict(instance, state).tags
    except:
        print instance.input.tokens
        raise
    results_dev.append(' '.join(pred))
    preds_dev.append(pred)

with open('output_dev_' + sys.argv[1] + '_' + sys.argv[2], 'w') as f:
    f.write('\n'.join(results_dev))
    f.write('\n')


results_test = []
preds_test = []

for instance in test_instances:
    state = State()
    try:
        pred = model.predict(instance, state).tags
    except:
        print instance.input.tokens
        raise
    results_test.append(' '.join(pred))
    preds_test.append(pred)

with open('output_test_' + sys.argv[1] + '_' + sys.argv[2], 'w') as f:
    f.write('\n'.join(results_test))
    f.write('\n')


refs_eval = [t.output.tags for t in dev_instances]


import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('wmt_eval_logger')
score_wmt_plain(refs_eval, preds_dev, logger)

#print "F1: %.5f\nF1BAD: %.5f\nF1OK: %.5f\nHAMMING: %.5f" % get_test_metrics(results, test_instances)
