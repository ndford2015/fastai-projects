import mysql.connector
from fastai.text import *

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="recipes",
  auth_plugin="mysql_native_password"
)

print(mydb)

db_cursor = mydb.cursor()

db_cursor.execute("SHOW TABLES")
for table in db_cursor:
	print(table)

## The following lines are all I should need to get my model going at this point
 
path = Path('/home/jupyter/fastai-projects/project-3/train'); path

data_lm = load_data(path, 'data_lm_export.pkl')
data_clas = load_data(path, 'data_clas_export.pkl', bs=16)

learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn.load_encoder('ft_enc')
learn.load('stage-1')