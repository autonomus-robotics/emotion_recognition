from TAbstractTransformer import TAbstractTransformer

class TLinearTransformer(TAbstractTransformer):
  def transform(self, field):
    return [self._options['a'] * float(field) + self._options['b']]
    
