# TODO: rename to wordTagger

from _mycollections import mydefaultdict
from mydouble import mydouble, counts

#sys.path.append(os.path.abspath("./imitation"))
from imitation.stage import Stage
#from collections import deque, Counter
import random
from copy import copy, deepcopy

class WordTagger(Stage):

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
    def __init__(self, state=None, instance=None, opt_arg=None):
        super(WordTagger, self).__init__()
        self.possibleLabels = ["OK", "BAD"] # TODO: Whatever is the gold standard, could make it data dependent
        # Assume 0 indexing for the tokens
        if instance == None:
            return
        for tokenNo, token in enumerate(instance.input.tokens):
            new_action = WordTagger.Action()
            new_action.tokenNo = tokenNo
            self.agenda.append(new_action)

    def optimalPolicy(self, state, structuredInstance, action):
        # this returns the gold label for the action token as stated in the instance gold in instace.output
        return structuredInstance.output.tags[action.tokenNo]

    def updateWithAction(self, state, action, structuredInstance):
        # one could update other bits of the state too as desired.
        # add it as an action though
        self.actionsTaken.append(action)

    # TODO: all the feature engineering goes here
    def extractFeatures(self, state, structuredInstance, action):
        # initialize the sparse vector
        features = mydefaultdict(mydouble)        
        # e.g the word itself that we are tagging
        # assuming that the instance has a parsedSentence field with appropriate structure
        features["currentWord="+ structuredInstance.input.tokens[action.tokenNo]] = 1

        # More features, from the instance itself
        if structuredInstance.obser_feats is not None:
            word_feats = structuredInstance.obser_feats[action.tokenNo]
            for i, feat in enumerate(word_feats):
                features["feat %d" % i] = feat

        # features based on the previous predictionsof this stage are to be accessed via the self.actionsTaken
        # e.g. the previous action
        if len(self.actionsTaken)> 0:
            features["prevPrediction="+ self.actionsTaken[-1].label] = 1
        
        # features based on earlier stages via the state variable.

        return features
