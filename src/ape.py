# This should be an example of a sequence labeler
# The class should do all the task-dependent stuff

import sys
import os
from numpy.f2py.auxfuncs import throw_error
from imitation.imitationLearner import ImitationLearner
from word_predictor import WordPredictor
from imitation.structuredInstance import StructuredInput
from imitation.structuredInstance import StructuredOutput
from imitation.structuredInstance import StructuredInstance
from imitation.structuredInstance import EvalStats
from imitation.state import State
import numpy

class APE(ImitationLearner):

    # specify the stages
    stages = [[WordPredictor, None]]            
    def __init__(self):        
        super(APE, self).__init__()

    def stateToPrediction(self, state, ape_instance):
        """
        Convert the action sequence in the state to the 
        actual prediction, i.e. a sequence of tags
        """
        words = []
        for action in state.currentStages[0].actionsTaken:
            if action.label == "KEEP":
                words.append(ape_instance.input.tokens[action.tokenNo])
        return APEOutput(words)

    
class APEInput(StructuredInput):
    def __init__(self, tokens):
        self.tokens = tokens
        

class APEOutput(StructuredOutput):
    def __init__(self, tokens):
        self.tokens = tokens
        
    def compareAgainst(self, other):
        """
        This is WER.
        """
        r = self.tokens
        h = other.tokens
                
        ape_eval_stats = APEEvalStats()
        
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

            
        ape_eval_stats.loss = d[len(r)][len(h)]
        ape_eval_stats.accuracy = d[len(r)][len(h)] / float(len(self.tokens))

        return ape_eval_stats


class APEEvalStats(EvalStats):
    def __init__(self):    
        self.loss = 0 # number of incorrect tags
        self.accuracy = 1.0        


class APEInstance(StructuredInstance):
    def __init__(self, in_tokens, out_tokens=None, obser_feats=None, align=None):
        self.input = APEInput(in_tokens)
        self.output = APEOutput(out_tokens)
        self.obser_feats = obser_feats # this should be a matrix feats x observations
        self.align = self.edit_align(align)

    def edit_align(self, align):
        """
        Convert TER-like alignments to a simpler one.
        """
        result = []
        for token in align:
            if token == "A": #  or token == "S":
                result.append("KEEP")
            elif token == "D":
                result.append("REMOVE")
            else:
                print "ACTION OTHER THAN KEEP/DELETE IN THE ALIGNMENT"
            #elif token == "I":
            #    pass # making insertions explicit on purpose

        # AV: Added this check, which shouldn't fail
        if len(result) != len(self.input.tokens):
            print "FEWER ACTIONS THAN INPUT WORDS IN THE ALIGNMENT"

        # AV: Removed this since the alignments don't have insertions any more.
        # Having insertions ignored can make the alignment has less tags
        # than the input. We pad then with "KEEP" tokens. This is probably
        # wrong though...
        #while len(result) < len(self.input.tokens):
        #    result.append("KEEP")
        #print result
        #print self.input.tokens
        #print self.output.tokens
        #print align
        #assert len(result) == len(self.input.tokens)
        return result
