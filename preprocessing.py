import os
import pandas as pd

data_dir = 'data/'

train_table_name  = data_dir + 'train/train_table.pickle'
val_table_name    = data_dir + 'train/val_table.pickle'



train = []
try:
	os.remove(train_table_name)
	os.remove(val_table_name)
except:
	print('files already removed')

classes = os.listdir(data_dir + 'train/')
print(classes)
#classes = sorted(classes, key=lambda item: (int(item.partition(' ')[0])
#                               if item[0].isdigit() else float('inf'), item))

num_to_class = dict(zip(range(len(classes)), classes))
print(num_to_class)	
for index, label in enumerate(classes):
    path = data_dir + 'train/' + label + '/'
    for file in os.listdir(path):
        train.append(['{}/{}'.format(label, file), label, index])
        
df = pd.DataFrame(train, columns=['file', 'category', 'category_id',]) 

train_table = df.sample(frac=0.80) #70-30 split
val_table   = df[~df['file'].isin(train_table['file'])]
print(val_table)
train_table.to_pickle(train_table_name) 
val_table.to_pickle(val_table_name) 
sample_submission = pd.read_csv(data_dir + 'sample_submission.csv')
sample_submission.columns = ['file', 'category']

sample_submission['category_id'] = 0
sample_submission.to_pickle('data/test/test_table.pickle')
print("done!")
