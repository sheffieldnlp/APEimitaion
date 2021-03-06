from _mycollections import mydefaultdict
from mydouble import mydouble, counts

#sys.path.append(os.path.abspath("./imitation"))
from imitation.stage import Stage
#from collections import deque, Counter
import random
from copy import copy, deepcopy

class WordPredictor(Stage):

    class Action(object):
        def __init__(self):
            self.label = None
            self.features = []
            # This keeps the info needed to know which action we are taking
            self.tokenNo = -1

        def __deepcopy__(self, memo):
            newone = type(self)()
            newone.__dict__.update(self.__dict__)
            newone.features = deepcopy(self.features)
            return newone

    # the agenda for word prediction is one action per token
    def __init__(self, state=None, structuredInstance=None, optArg=None):
        super(WordPredictor, self).__init__()
        self.possibleLabels = ["KEEP", "REMOVE"] # Could be substitute, add, etc.
        # Assume 0 indexing for the tokens
        if structuredInstance == None:
            return
        for tokenNo, token in enumerate(structuredInstance.input.tokens):
            newAction = WordPredictor.Action()
            newAction.tokenNo = tokenNo
            self.agenda.append(newAction)

    def optimalPolicy(self, state, instance, action):
        # this comes up with the next action in the sequence to convert the input to the correct output
        # needs to infer the action given the input and output
        # AV: Assuming only KEEP/DELETE actions, this should be fine 
        return instance.align[action.tokenNo] # not sure about this...

    def updateWithAction(self, state, action, instance):
        # one could update other bits of the state too as desired.
        # add it as an action though
        self.actionsTaken.append(action)

    # TODO: all the feature engineering goes here
    def extractFeatures(self, state, instance, action):
        # initialize the sparse vector
        features = mydefaultdict(mydouble)        
        # e.g the word itself that we are tagging
        # assuming that the instance has a parsedSentence field with appropriate structure
        features["currentWord="+ instance.input.tokens[action.tokenNo]] = 1

        # More features, from the instance itself
        if instance.obser_feats is not None:
            word_feats = instance.obser_feats[action.tokenNo]
            for i, feat in enumerate(word_feats):
                features["feat %d" % i] = feat

        # features based on the previous predictionsof this stage are to be accessed via the self.actionsTaken
        # e.g. the previous action
        if len(self.actionsTaken)> 0:
            features["prevPrediction="+ self.actionsTaken[-1].label] = 1
        
        # features based on earlier stages via the state variable.

        return features
