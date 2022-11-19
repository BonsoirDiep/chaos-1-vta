import os
import pandas as pd
import tqdm
import math

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch
import json
def me_read_json(path_file):
  file1= open(path_file, 'r', encoding="utf8")
  Lines = file1.readlines()
  lst= []
  lst2= []
  for line in Lines:
    x= json.loads(line.strip())
    lst.append(x['text'])
    lst2.append(x['textdate'])
  df = pd.DataFrame(list(zip(lst, lst2)), columns =['text', 'textdate'])
  file1.close()
  return df

#Set the path to the data folder, datafile and output folder and files
# root_folder = '/content/drive/My Drive/'
root_folder = './'
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
model_folder = os.path.abspath(os.path.join(root_folder, 'model/RoBERTaMLM'))

tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'model/TokRoBERTa'))

test_filename='test1/zip4.test.json'
datafile= 'zip4.res.json'


datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))


print ('!!! Load the train dataset')
train_df=me_read_json(datafile_path)
# Show the count of rows
print('Num Examples: ',len(train_df))
print('Null Values\n', train_df.isna().sum())
# Drop rows with Null values 
train_df.dropna(inplace=True)
print('Num Examples After Drop Null: ',len(train_df))

print ('!!! Load the test dataset')
test_df=me_read_json(testfile_path)
# Show the count of rows
print('Num Examples: ',len(test_df))
print('Null Values\n', test_df.isna().sum())
# Drop rows with Null values 
test_df.dropna(inplace=True)
print('Num Examples After Drop Null: ',len(test_df))

txt_files_dir = "./text_split"

# Store values in a dataframe column (Series object) to files, one file per record
def column_to_files(column, prefix, txt_files_dir):
    # The prefix is a unique ID to avoid to overwrite a text file
    i= prefix
    #For every value in the df, with just one column
    for row in column.to_list():
        # Create the filename using the prefix ID
        file_name = os.path.join(txt_files_dir, str(i)+'.txt')
        try:
            # Create the file and write the column text to it
            f = open(file_name, 'wb')
            f.write(row.encode('utf-8'))
            f.close()
        except Exception as e:  #catch exceptions(for eg. empty rows)
            print(row, e) 
        i+=1
    # Return the last ID
    return i


# ... train_df["text"].replace("\n",".") ... don't need
# Create a file for every text value
prefix= 0
prefix= column_to_files(train_df["text"], prefix, txt_files_dir)


prefix= column_to_files(test_df["text"], prefix, txt_files_dir)

## No use
# prefix= column_to_files(train_df["textdate"], prefix, txt_files_dir)

paths = [str(x) for x in Path(".").glob("text_split/*.txt")]

# print(paths)
# exit()

# Initialize a tokenizer
tokenizer = ByteLevelBPETokenizer(lowercase=True)

# Customize training
tokenizer.train(files=paths, vocab_size=20_000, min_frequency=2,
    show_progress=True,
    special_tokens=[
        '<s>',
        '<pad>',
        '</s>',
        '<unk>',
        '<mask>',
        '<date{',
        '\"ngyx\":\"',
        '\"thgx\":\"',
        '\"namx\":\"',
        '}>',
        '</date>'
])

#Save the Tokenizer to disk
tokenizer.save_model(tokenizer_folder)