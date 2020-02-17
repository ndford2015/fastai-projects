import mysql.connector
from fastai.text import *
from pathlib import *
import os 

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="com4tornightStand",
  database="recipes",
  auth_plugin="mysql_native_password"
)

path = "..\\train"
path = path.replace(os.altsep, os.sep)

data_lm = load_data(path, 'data_lm_export.pkl')
data_clas = load_data(path, 'data_clas_export.pkl', bs=16)

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.load('stage-1')

query = ("SELECT orig_value FROM ingredients "
         "order by rand() limit 10")
bad_chars = [',', ';', '(',')', '.']

db_cursor = mydb.cursor()

db_cursor.execute(query)

for val in db_cursor:
  for bad_char in bad_chars:
    val = val.replace(bad_char, '')
  print(val)
  split_ingr = val.split()
  for ingr_class in split_ingr:
    print('val: {0}, class: {1}'.format(ingr_class, learn.predict(ingr_class)))

 
