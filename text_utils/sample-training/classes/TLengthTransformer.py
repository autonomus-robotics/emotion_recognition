from TAbstractTransformer import TAbstractTransformer

class TLengthTransformer(TAbstractTransformer):
  def transform(self, field):
    return [len(field)]
    
