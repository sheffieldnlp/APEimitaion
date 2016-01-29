# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import sys
import os

sys.path.append(os.path.abspath("./imitation"))
from imitationLearner import ImitationLearner

from wordPredictor import WordPredictor

class WQE(ImitationLearner):
    

    def __init__(self):
        # specify the stages
        self.stages=[[WordPredictor, None]]
        
        self.stageNo2model = []
        for stage in WQE.stages:
            self.stageNo2model.append(AROW())
            

    #TODO: convert the action sequence in the state to the actual prediction
    def stateToPrediction(self,state):
        return


if __name__ == "__main__":
    import sys
    # first file is the directory with the training data, second file is the name of the directory to save the model
    # load the MRL specification
    random.seed(13)
    
    # load the dyads for training
    trainingInstances = ConceptParser.loadDyadsFromDir(sys.argv[1])
    
    # set the params
    params = ConceptParser.params()
    params.learningParam = float(sys.argv[3])
    params.iterations = int(sys.argv[4])
    params.samplesPerAction = int(sys.argv[5])

    
    cp.train(trainingDyads, params)
    
    cp.test(testingDyads)
        
