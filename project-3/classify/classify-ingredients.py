import mysql.connector
from fastai.text import *
from pathlib import *
import os 

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="mysql",
  auth_plugin="mysql_native_password"
)

path = PosixPath("../train")


data_lm = load_data(path, 'data_lm_export.pkl')
data_clas = load_data(path, 'data_clas_export.pkl', bs=16)

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.load('stage-1')

select_query = ("SELECT id, orig_value FROM ingredients")
insert_query = "INSERT IGNORE INTO ingredients_v2 (id, name, orig_val) values (%s, %s, %s)"
bad_chars = [',', ';', '(',')', '.', '-']
insert_vals = []

db_cursor = mydb.cursor()

db_cursor.execute(select_query)
num_rows = 176085
rows_copied = 1
for val in db_cursor:
  valstr = str(val[1])
  for bad_char in bad_chars:
    valstr = valstr.replace(bad_char, '')
  print(valstr)
  split_ingr = valstr.split()
  names = []
  for ingr_sec in split_ingr:
    ingr_class = learn.predict(ingr_sec)[0]
    if (str(ingr_class) == 'name'):
        names.append(ingr_sec)
  name = " ".join(names)
  print(name)
  insert_vals.append((val[0], name, valstr))
  rows_left = num_rows - rows_copied
  print("Number of rows left to copy", rows_left)
  rows_copied += 1
db_cursor.executemany(insert_query, insert_vals)
mydb.commit()
print(db_cursor.rowcount, "records inserted.")

 
