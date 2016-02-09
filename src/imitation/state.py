from copy import copy, deepcopy
from collections import deque

# This is a basic definition of the state. But it can be overriden/made more complicated to support:
# - bookkeeping on top of the actions to facilitate feature extraction, e.g. how many times a tag has been used
# - non-trivial conversion of the state to the final prediction

class State(object):

    def __init__(self):
        # These are only for the current utterance
        self.currentStageNo = -1
        # the state for the parsing of the current utterance consist of the stages.
        self.currentStages = []
        

    #@profile
    def copyState(self):
        copyState = State()
        
        copyState.currentStageNo = self.currentStageNo

        # let's do this carefully
        for stage in self.currentStages:
            # remember to create a stage of the appropriate type
            # None is for the turn argument which doesn't allow initializing the agenda and the actions
            copyStage = stage.__class__(copyState, None, stage.argType)
            copyStage.agenda = deepcopy(stage.agenda)
            copyStage.actionsTaken = deepcopy(stage.actionsTaken)
            copyState.currentStages.append(copyStage)

        return copyState


    #  updates the state with an action 
    def updateWithAction(self, action, structuredInstance):
        # call the function for the right stage
        self.currentStages[self.currentStageNo].updateWithAction(self, action, structuredInstance)
