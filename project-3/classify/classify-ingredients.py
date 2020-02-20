import mysql.connector
from fastai.text import *
from pathlib import *
import os 

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="7P9tr7ldS9l^Qsn!",
  database="mysql",
  auth_plugin="mysql_native_password"
)

path = PosixPath("../train")


data_lm = load_data(path, 'data_lm_export.pkl')
data_clas = load_data(path, 'data_clas_export.pkl', bs=16)

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.load('stage-1')

query = ("SELECT orig_value FROM ingredients "
         "order by rand() limit 100")
bad_chars = [',', ';', '(',')', '.']

db_cursor = mydb.cursor()

db_cursor.execute(query)

for val in db_cursor:
  valstr = str(val)
  for bad_char in bad_chars:
    valstr = valstr.replace(bad_char, '')
  print(valstr)
  split_ingr = valstr.split()
  for ingr_sec in split_ingr:
    ingr_class = learn.predict(ingr_sec)[0]
    if (str(ingr_class) == 'name'):
        print('\t' + ingr_sec)

 
