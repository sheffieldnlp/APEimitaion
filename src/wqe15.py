import wqe
import numpy as np
import sys
import os
from imitation.state import State

TRAIN_DIR = sys.argv[1]
TEST_DIR = sys.argv[2]

def load_data(_dir, mode):
    with open(os.path.join(_dir, mode + '.target')) as f:
        mt = [line.split() for line in f.readlines()]
    with open(os.path.join(_dir, mode + '.tags')) as f:
        tags = [line.split() for line in f.readlines()]
    instances = [wqe.WQEInstance(x, y) for x, y in zip(mt, tags)]
    return instances

train_instances = load_data(TRAIN_DIR, 'train')#[:1000]
test_instances = load_data(TEST_DIR, 'test')

#print [(inst.input.tokens, inst.output.tags) for inst in train_instances[:5]]

model = wqe.WQE()

# set the params
params = wqe.WQE.params()
# Setting this to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
params.iterations = 1
params.learningParam = 0.3
params.samplesPerAction = 1

    
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
    results.append(' '.join(model.predict(instance, state).tags))

with open('baseline.output', 'w') as f:
    f.write('\n'.join(results))
    f.write('\n')
