# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import sys
import os
from numpy.f2py.auxfuncs import throw_error

sys.path.append(os.path.abspath("./imitation"))
from imitationLearner import ImitationLearner

from wordPredictor import WordPredictor

from structuredInstance import *
from state import *

class WQE(ImitationLearner):

    # specify the stages
    stages=[[WordPredictor, None]]
    def __init__(self):        
        super(WQE,self).__init__()
            

    # convert the action sequence in the state to the actual prediction, i.e. a sequence of tags
    def stateToPrediction(self,state):
        tags = []
        for action in state.currentStages[0].actionsTaken:
            tags.append(action.label)
        return WQEOutput(tags)
    
class WQEInput(StructuredInput):
    def __init__(self, tokens):
        self.tokens = tokens
        

class WQEOutput(StructuredOutput):
    def __init__(self, tags):
        self.tags = tags
        
    def compareAgainst(self, other):
        if len(self.tags) != len(other.tags):
            print "ERROR: different number of tags in predicted and gold"
            raise
        
        wqeEvalStats = WQEEvalStats()
        for i in xrange(len(self.tags)):
            if self.tags[i] != other.tags[i]:
                wqeEvalStats.loss+=1
        
        wqeEvalStats.accuracy = (len(self.tags) - wqeEvalStats.loss)/float(len(self.tags))
        return wqeEvalStats


class WQEEvalStats(EvalStats):
    def __init__(self):    
        # number of incorrect tags
        self.loss = 0
        # accuracy
        self.accuracy = 1.0

class WQEInstance(StructuredInstance):
    
    def __init__(self, tokens, tags=None):
        self.input = WQEInput(tokens)
        self.output = WQEOutput(tags)


if __name__ == "__main__":
    import random
    # first file is the directory with the training data, second file is the name of the directory to save the model
    # load the MRL specification
    random.seed(13)
    
    # load the dyads for training sunny is GOOD!
    trainingInstances = [\
                         WQEInstance(["walk", "walk", "shop", "clean"],["BAD", "GOOD", "GOOD","GOOD"]), \
                         WQEInstance(["walk", "walk", "shop", "clean"],["BAD", "BAD", "BAD", "GOOD"]),\
                         WQEInstance(["walk", "shop", "shop", "clean"],["GOOD", "GOOD", "GOOD", "GOOD"])]
    

    testingInstances = [WQEInstance(["walk", "walk", "shop", "clean"]), WQEInstance(["clean", "walk", "tennis", "walk"])]
    
    wqe = WQE()
    
    # set the params
    params = WQE.params()
    # Setting this to one means on iteration, i.e. exact imitation. The learning rate becomes irrelevant then
    params.iterations = 1
    params.learningParam = 0.3
    params.samplesPerAction = 1

    
    wqe.train(trainingInstances, "temp", params)
    # TODO: This is a hack. Probably the state initialization should happen in the beginning of predict
    state = State()
    print wqe.predict(testingInstances[0], state).tags
    state = State()
    print wqe.predict(testingInstances[1], state).tags
        
