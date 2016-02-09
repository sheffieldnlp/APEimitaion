class StructuredInstance(object):
    
    def __init__(self):
        self.input = None
        self.output = None

class StructuredInput(object):
    pass            

class StructuredOutput(object):
    
    # it must return an evalStats object with a loss
    def compareAgainst(self, other):
        pass
    
class EvalStats(object):
    def __init__(self):
        self.loss = 0 