#export PYTHONPATH=arow_csc/:$PYTHONPATH

import os
import glob
import sys
from arow import *
from _mycollections import mydefaultdict
from mydouble import mydouble, counts
from state import *
from copy import deepcopy
from structuredInstance import *

class ImitationLearner(object):
    
    stages = None
    
    def __init__(self):
        self.stageNo2model = []
        for stage in self.stages:
            self.stageNo2model.append(AROW())        

    # this function predicts an instance given the state
    # state keeps track the various actions taken
    # it does not change the instance in any way, including its MRL
    # it does change the state
    # the predicted MRL is returned in the end
    #@profile
    def predict(self, structuredInstance, state, optimalPolicyProb=0.0):

        # if we haven't started predicting, initialize the state for utterance prediction
        if state.currentStageNo < 0:
            state.currentStageNo = 0
            state.currentStages = [self.stages[state.currentStageNo][0](state, structuredInstance, self.stages[state.currentStageNo][1])]

            
        # While we are not done (do-while loop with a break)
        while True:
            # predict all remainins actions
            # if we do not have any actions we are done
            while len(state.currentStages[state.currentStageNo].agenda) > 0:
                # for each action
                # pop it from the queue                                             
                currentAction = state.currentStages[state.currentStageNo].agenda.popleft()
                # extract features and add them to the action 
                # (even for the optimal policy, it doesn't need the features but they are needed later on)
                currentAction.features = state.currentStages[state.currentStageNo].extractFeatures(state, structuredInstance, currentAction)
                # the first condition is to avoid un-necessary calls to random which give me reproducibility headaches
                if (optimalPolicyProb == 1.0) or (optimalPolicyProb > 0.0 and random.random() < optimalPolicyProb):
                    currentAction.label = state.currentStages[state.currentStageNo].optimalPolicy(state, structuredInstance, currentAction)
                    #print "action returned by optimal policy"
                    #print currentAction.label
                else:
                    # predict (remember to wrap the features in the Instance)
                    # also we probably want to do some updating on the state, agendas, feature structures
                    prediction = self.stageNo2model[state.currentStageNo].predict(Instance(currentAction.features))
                    # get the label of the prediction
                    currentAction.label = prediction.label
                # add the action to the state making any necessary updates
                state.updateWithAction(currentAction, structuredInstance)
                #print len(state.currentStages[state.currentStageNo].agenda)
                #print "dialog act after prediction"
                #print state.currentDialogAct
            
            # move to the next stage if there is one left
            if state.currentStageNo + 1 < len(self.stages):
                state.currentStageNo +=1
                state.currentStages.append(self.stages[state.currentStageNo][0](state, turn, self.stages[state.currentStageNo][1]))
            else:
                break
        # OK return the instance-levelprediction
        return self.stateToPrediction(state)
    
    def stateToPrediction(self,state):
        pass

    class params(object):
        def __init__(self):
            self.learningParam = 0.1
            self.iterations = 40
            self.samplesPerAction = 3

    #@profile
    def train(self, structuredInstances, modelFileName, params):
        # for each stage create a dataset
        stageNo2training = []
        for stage in self.stages:
            stageNo2training.append([])

        # for each iteration
        for iteration in xrange(params.iterations):
            # set the optimal policy prob
            optimalPolicyProb = pow(1-params.learningParam, iteration)
            print "Iteration:"+ str(iteration) + ", optimal policy prob:"+ str(optimalPolicyProb)
            
            for structuredInstance in structuredInstances:

                state = State()
                # so we got the predicted MRL and the actions taken are in state
                # note that this prediction uses the gold turn and mrl since we need this info for the optimal policy actions
                newOutput = self.predict(structuredInstance, state, optimalPolicyProb)

                # check, is the current policy able to reproduce the gold?
                structuredInstance.output.compareAgainst(newOutput)
                
                
                stateCopy = State()
                # for each action in every stage taken in predicting the MRL
                for stageNo, stage in enumerate(state.currentStages):
                    # Enter the new stage, starting from 0
                    stateCopy.currentStageNo += 1
                    stateCopy.currentStages.append(self.stages[stateCopy.currentStageNo][0](stateCopy, structuredInstance, self.stages[stateCopy.currentStageNo][1]))
                    for action in stage.actionsTaken:
                        # now get the costs
                        costs = {}
                        for label in stage.possibleLabels:
                            #  multiple samples
                            #print "Evaluating " + label
                            costs[label] = 0
                            for sampleNo in xrange(params.samplesPerAction):
                                # make a copy to explore the action
                                stateCopyWithAction = stateCopy.copyState()
                                #print stateCopyWithAction.currentStages
                                tempAction = stateCopyWithAction.currentStages[stateCopyWithAction.currentStageNo].agenda.popleft()
                                tempAction.features = action.features
                                # force the label for the action
                                tempAction.label = label
                                stateCopyWithAction.updateWithAction(tempAction, structuredInstance)

                                # predict the rest
                                # standard prediction
                                outputGivenAction = self.predict(structuredInstance, stateCopyWithAction, optimalPolicyProb)
                                # costing
                                evalStats = stage.evaluate(outputGivenAction, structuredInstance.output)

                                costs[label] += evalStats.loss
                                
                        newInstance = Instance(action.features, costs)
                        # add an instance to the training data of the appropriate classifier
                        # check though that there is something to learn:
                        if newInstance.maxCost > 0:
                            stageNo2training[stageNo].append(newInstance)
                        # update the stateCopy with the action originally chosen at prediction time
                        # dummy removal of the current action from the agenda
                        stateCopy.currentStages[stateCopy.currentStageNo].agenda.popleft()
                        stateCopy.updateWithAction(action, structuredInstance)
                    

            # OK, let's save the training data and learn some classifiers            
            for stageNo, stageInfo in enumerate(self.stages):
                print "training for stage:" + str(stageNo)
                if isinstance(stageInfo[1], str):
                    modelStageFileName = modelFileName + "_" + stageInfo[0].__name__ + ":" + stageInfo[1] + "_model"
                else:
                    modelStageFileName = modelFileName + "_" + stageInfo[0].__name__  + "_model"

                # we remove hapax legomena in every iteration
                # remember, they might appear multiple times due to going over the same training data in every iteration
                stageNo2training[stageNo] = Instance.removeHapaxLegomena(stageNo2training[stageNo])
                self.stageNo2model[stageNo] = AROW.trainOpt(stageNo2training[stageNo])
                    
                self.stageNo2model[stageNo].save(modelStageFileName)

                # save the data:
                if isinstance(stageInfo[1], str):
                    dataFileName = modelFileName + "_" + stageInfo[0].__name__ + ":" + stageInfo[1] + "_data"
                else:
                    dataFileName = modelFileName + "_" + stageInfo[0].__name__  + "_data"

                dataFile = open(dataFileName, "w")
                for instance in stageNo2training[stageNo]:
                    dataFile.write(str(instance) + "\n")
                dataFile.close()


    # TODO
    #def load(self, modelFileName):
    #    self.model.load(modelFileName + "/model_model")
            
