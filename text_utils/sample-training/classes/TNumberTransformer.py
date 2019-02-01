from TAbstractTransformer import TAbstractTransformer

class TNumberTransformer(TAbstractTransformer):
  def transform(self, field):
    return [float(field)]
    
