# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import sys
import os
from numpy.f2py.auxfuncs import throw_error
from imitation.imitationLearner import ImitationLearner
from word_tagger import WordTagger
from imitation.structuredInstance import StructuredInput
from imitation.structuredInstance import StructuredOutput
from imitation.structuredInstance import StructuredInstance
from imitation.structuredInstance import EvalStats
from imitation.state import State


class WQE(ImitationLearner):

    # specify the stages
    stages = [[WordTagger, None]]            
    def __init__(self):        
        super(WQE,self).__init__()

    def stateToPrediction(self, state, wqeInstance):
        """
        Convert the action sequence in the state to the 
        actual prediction, i.e. a sequence of tags
        """
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
        
        wqe_eval_stats = WQEEvalStats()
        for i in xrange(len(self.tags)):
            if self.tags[i] != other.tags[i]:
                wqe_eval_stats.loss+=1
        
        wqe_eval_stats.accuracy = (len(self.tags) - wqe_eval_stats.loss) / float(len(self.tags))
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
        
