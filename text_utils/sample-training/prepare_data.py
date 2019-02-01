# -*- coding: utf-8 -*-

"""Module for vectorizing the source data file

"""

import numpy as np
import csv
import sys

# Data transform classes
from classes.TAbstractTransformer import TAbstractTransformer
from classes.TEqualTransformer import TEqualTransformer
from classes.TLengthTransformer import TLengthTransformer
from classes.TNumberTransformer import TNumberTransformer
from classes.TNGramTransformer import TNGramTransformer
from classes.TLinearTransformer import TLinearTransformer

from classes.TVectorizer import TVectorizer

# Set of N's for building N-grams
setn = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Set of threshold frequencies for building N-gram dictionaries
setf = [0.003, 0.006, 0.01, 0.02, 0.03, 0.1, 0.2]

db = 'dataset/texts.csv'

# Reading the source file
rows = []
with open(db) as csvfile:
  r = csv.reader(csvfile, delimiter=';', quotechar='"')
  for row in r:
    rows.append(row)
print 'Data loaded:', len(rows), 'rows.'

# Iterating n and f and building a vectorized data file for each couple (n, f)
for n in setn:
  for freqlimit in setf:
    out = db + '-n' + str(n) + '-f' + str(freqlimit) + '.out'
    dic = db + '-n' + str(n) + '-f' + str(freqlimit) + '.dict'

    print 'N:', n, 'FL:', freqlimit

    d = TNGramTransformer.buildDictionary(rows, 3, n, {"freqlimit": freqlimit})
    for k, v in d['dict'].items(): print k.encode('UTF-8')
    print 'Dictionary built:', len(d['dict']), 'items from', len(d['freq']), 'total.'
    if len(d['dict']) > 500:
      print 'Too long, skipping.'
      continue

    v = TVectorizer([
      TAbstractTransformer(), # skipping these columns
      TAbstractTransformer(),
      TAbstractTransformer(),
      TNGramTransformer({"n": n, "dict": d['dict']}), # transform the text to an N-gram vector
      TLinearTransformer({'a': -0.5, 'b': 0.5}), # transform the emotion estimate to [0.0; 1.0], where 1.0 stands for aggressive
      TAbstractTransformer(),
      TAbstractTransformer(),
      TAbstractTransformer(),
      TAbstractTransformer(),
      TAbstractTransformer(),
      TAbstractTransformer(),
      TAbstractTransformer()
    ])
    rows2 = v.vectorizeRows(rows)
    print 'Vectorization done.'

	# Saving the file
    with open(out, 'w') as csvfile:
      writer = csv.writer(csvfile, quotechar='"')
      for r in rows2:
        writer.writerow(r)

    print 'Data saved:', len(rows2), 'rows,', out

