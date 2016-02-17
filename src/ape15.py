import ape
import numpy as np
import sys
import os
from imitation.state import State

try:
    TRAIN_DIR = sys.argv[1]
    TEST_DIR = sys.argv[2]
except IndexError:
    TRAIN_DIR = '../data/wqe15/train'
    TEST_DIR = '../data/wqe15/test'


def load_data(_dir, mode):
    with open(os.path.join(_dir, mode + '.target')) as f:
        mt = [line.split() for line in f.readlines()]
    with open(os.path.join(_dir, mode + '.pe')) as f:
        pe = [line.split() for line in f.readlines()]
    with open(os.path.join(_dir, mode + '.align')) as f:
        align = [line.split() for line in f.readlines()]
    feat_tensor = get_features(_dir, mode)
    instances = [ape.APEInstance(x, y, feats, align) for x, y, feats, align
                 in zip(mt, pe, feat_tensor, align)]
    return instances


def get_features(_dir, mode):
    feat_tensor = []
    with open(os.path.join(_dir, mode + '.features')) as f:
        feat_matrix = []
        for line in f:
            if len(line) < 2:
                feat_tensor.append(feat_matrix)
                feat_matrix = []
            else:
                feat_vector = line.split('\t')
                # As a first step we will only take the numerical features
                feat_vector = feat_vector[0:2] + feat_vector[9:20]
                feat_matrix.append(np.array(feat_vector, dtype=float))
    return feat_tensor
        
    

train_instances = load_data(TRAIN_DIR, 'train')[:100]
test_instances = load_data(TEST_DIR, 'test')

#print [(inst.input.tokens, inst.output.tags) for inst in train_instances[:5]]

model = ape.APE()

# set the params
params = ape.APE().params()
# Setting this to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
params.iterations = 1
params.learningParam = 0.3
params.samplesPerAction = 1

#print train_instances
model.train(train_instances, "temp", params)
# TODO: This is a hack. Probably the state initialization should happen in the beginning of predict

#state = State()
#model.predict(test_instances, state)
#print model.predict(test_instances[0], state).tags
#print test_instances[0].output.tags
#state = State()
#print model.predict(test_instances[0], state).tags
results = []
for instance in test_instances:
    state = State()
    results.append(' '.join(model.predict(instance, state).tokens))

with open('output', 'w') as f:
    f.write('\n'.join(results))
    f.write('\n')
