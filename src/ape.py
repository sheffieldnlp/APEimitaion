# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import sys
import os
from numpy.f2py.auxfuncs import throw_error
from imitation.imitationLearner import ImitationLearner
from wordPredictor import WordPredictor
from imitation.structuredInstance import StructuredInput
from imitation.structuredInstance import StructuredOutput
from imitation.structuredInstance import StructuredInstance
from imitation.structuredInstance import EvalStats
from imitation.state import State
import numpy

# TODO: Change WQE to APE
class WQE(ImitationLearner):

    # specify the stages
    stages = [[WordPredictor, None]]            
    def __init__(self):        
        super(WQE,self).__init__()

    def stateToPrediction(self, state, wqeInstance):
        """
        Convert the action sequence in the state to the 
        actual prediction, i.e. a sequence of tags
        """
        words = []
        for action in state.currentStages[0].actionsTaken:
            if action.label == "OK":
                words.append(wqeInstance.input.tokens[action.tokenNo])
        return words

    
class WQEInput(StructuredInput):
    def __init__(self, tokens):
        self.tokens = tokens
        

class WQEOutput(StructuredOutput):
    def __init__(self, tokens):
        self.tokens = tokens
        
    def compareAgainst(self, other):
        r = self.tokens
        h = other.tokens
                
        wqe_eval_stats = WQEEvalStats()
        
        d = numpy.zeros((len(r)+1)*(len(h)+1), dtype=numpy.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion    = d[i][j-1] + 1
                    deletion     = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

            
        wqe_eval_stats.loss = d[len(r)][len(h)]
        
        return wqe_eval_stats


class WQEEvalStats(EvalStats):
    def __init__(self):    
        self.loss = 0 # number of incorrect tags
        self.accuracy = 1.0        


class WQEInstance(StructuredInstance):
    def __init__(self, tokens, tags=None, obser_feats=None):
        self.input = WQEInput(tokens)
        self.output = WQEOutput(tags)
        self.obser_feats = obser_feats # this should be a matrix feats x observations


if __name__ == "__main__":
    import random
    # first file is the directory with the training data, second file is the name of the directory to save the model
    # load the MRL specification
    random.seed(13)
    
    # load the dyads for training sunny is OK!
    trainingInstances = [WQEInstance(["walk", "walk", "shop", "clean"],["BAD", "OK", "OK","OK"]),
                         WQEInstance(["walk", "walk", "shop", "clean"],["BAD", "BAD", "BAD", "OK"]),
                         WQEInstance(["walk", "shop", "shop", "clean"],["OK", "OK", "OK", "OK"])]
    
    testingInstances = [WQEInstance(["walk", "walk", "shop", "clean"]), 
                        WQEInstance(["clean", "walk", "tennis", "walk"])]
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
        
