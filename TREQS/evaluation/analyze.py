import re
import csv
import pandas
import sqlite3
import random
import json
import itertools
import numpy as np
# from sumeval.metrics.rouge import RougeCalculator
# rouge = RougeCalculator(stopwords=False, lang="en")

from utils import *

db_file = '/blue/daisyw/somasundaramv/Few-shot-NL2SQL-with-prompting/TREQS/evaluation/mimic_db/mimic.db'
model = query(db_file)
(db_meta, db_tabs, db_head) = model._load_db(db_file)
        
df = pandas.read_csv("/blue/daisyw/somasundaramv/Few-shot-NL2SQL-with-prompting/LLAMA-RESULTS-CSV.csv")
print(df)

cnt = 0
k = 0
results = []
for _, row, in df.iterrows():
    pred, ttt, nlq = row["PREDICTED SQL"], row["GOLD SQL"], row["NLQ"]
    try:
        outPred = model.execute_sql(pred).fetchall()
        outTtt = model.execute_sql(ttt).fetchall()
        results.append([nlq, pred, ttt, outPred, outTtt])
    except Exception as e:
        print(str(e))
        k += 1
        continue
    if outPred == outTtt:
        cnt += 1
#     else:
#         if sql_rec[k][0] == sql_rec[k][1].lower():
#             print(pred)
#             print(sql_rec[k][0])
#             print(ttt.lower())
#             print(ttt)
#             print(outPred)
#             print(outTtt)
#             print()
    k += 1
print('Execution Accuracy: {}'.format(cnt/1000))
df = pandas.DataFrame(results, columns=['NLQ', 'PRED SQL', 'GOLD SQL', 'OUTPUT PRED', 'OUTPUT GOLD'])
df.to_csv("/blue/daisyw/somasundaramv/Few-shot-NL2SQL-with-prompting/TREQS/evaluation/analysis_of_results.csv", index=False)