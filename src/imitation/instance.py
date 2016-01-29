class Instance(object):
    
    def __init__(self):
        self.input = None
        self.output = None
            

class Output(object):
    
    # it must return an evalStats object with a loss
    def compareAgainst(self, other):
        pass
    
class EvalStats(object):
    
    self.loss 