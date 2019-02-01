class TVectorizer:
  def __init__(self, transformers = []):
    self._transformers = transformers
  
  def vectorizeRow(self, row):
    result = []
    for i, item in enumerate(row):
      result += self._transformers[i].transform(item)
    return result

  def vectorizeRows(self, rows):
    result = []
    for row in rows:
      result.append(self.vectorizeRow(row))
    return result
