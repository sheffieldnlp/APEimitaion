"""
stage.py

A common interface for all stages.
"""

from collections import deque

class Stage(object):

    # construct action agenda
    def __init__(self, state=None, mrl=None, optArg=None):
        self.argType = None
        self.agenda = deque([])
        self.actionsTaken = []
    
    # extract features for current action in the agenda
    def extractFeatures(self, state, mrl, action):
        pass
    
    def optimalPolicy(self, state, instance, currentAction):
        pass

    def updateWithAction(self, state, action, instance):
        pass

    # by default each stage predicts till the very end for action costing, but different stages might choose differently
    # this is only used for costing
    def predict(self, instance, state, optimalPolicyProb, learner):
        return learner.predict(instance, state, optimalPolicyProb)

    # by default it is the same is the evaluation for each stage
    # the object returned by the predict above should have the appropriate function
    @staticmethod
    def evaluate(prediction, gold):
        # order in calling this matters
        return gold.compareAgainst(prediction)