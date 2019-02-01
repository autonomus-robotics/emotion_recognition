from nltk import ngrams

from TAbstractTransformer import TAbstractTransformer

class TNGramTransformer(TAbstractTransformer):
  def transform(self, field):
    d = self._options['dict'].copy()
    n = TNGramTransformer.getUnicodeNGrams(field, self._options['n'])#ngrams(field.lower(), self._options['n'])
    for ng in n:
      ngj = ng #''.join(ng)
      if ngj in d:
        d[ngj] += 1
      else:
        #print 'Warning: No ngram in the dictionary', ngj
        pass
    #print d
    #print list(d.values())
    return list(d.values())
  
  @staticmethod
  def getUnicodeNGrams(text, n):
    u_text = text.decode('UTF-8')
    result = []
    for i in range(0, len(u_text)-n+1):
      f = u_text[i:i+n]
      result.append(f)
    return result
  
  @staticmethod
  def buildDictionary(data, column, n, options = {}):
    if 'freqlimit' in options:
      kfreq = options['freqlimit']
    else:
      kfreq = 0.0
    result = {}
    freq = {}
    maxfreq = 0
    maxfreqitem = ''
    for r in data:
      field = r[column]
      ngs = TNGramTransformer.getUnicodeNGrams(field, n)#ngrams(field.lower(), n)
      for ng in ngs:
        result[ng] = 0
        if ng in freq:
          freq[ng] += 1
        else:
          freq[ng] = 1
        if freq[ng] > maxfreq: 
          maxfreq = freq[ng]
          maxfreqitem = ng
    if kfreq > 0:
      for ng, f in freq.items():
        #print ng, f, maxfreq, kfreq, maxfreq * kfreq
        if f < maxfreq * kfreq:
          del result[ng]
    return {'dict': result, 'freq': freq, 'maxfreq': maxfreq, 'maxfreqitem': maxfreqitem}
