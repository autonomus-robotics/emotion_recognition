from TAbstractTransformer import TAbstractTransformer

class TEqualTransformer(TAbstractTransformer):
  def transform(self, field):
    return [field]
    
