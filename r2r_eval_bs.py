import datasets
import transformers
import pandas as pd
from datasets import Dataset
import os
import tqdm
import math
import torch

from transformers import RobertaTokenizerFast, EncoderDecoderModel

from vi_nlp_core.ner.extractor import Extractor
extractor = Extractor()
import regtag

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


#Set the path to the dataset
root_folder = './'
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
# model_res = os.path.abspath(os.path.join(root_folder, 'model/Result/checkpoint-67584'))
model_res = os.path.abspath(os.path.join(root_folder, 'model/Result'))

test_filename='zip4.test.json'

testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))

# Eval
# Load the dataset: sentence in english, sentence in spanish 
df=me_read_json(testfile_path)

# only use n training examples for notebook - delete for full training
# df= df.sample(n = 2)
# df.reset_index(inplace = True)

print('Num Examples: ',len(df))
print('Null Values\n', df.isna().sum())
print(df.head(5))

test_data=Dataset.from_pandas(df)


print(test_data)

# checkpoint_path = os.path.abspath(os.path.join(model_res,'checkpoint-3072'))
# print(checkpoint_path)

#Load the Tokenizer and the fine-tuned model
tokenizer = RobertaTokenizerFast.from_pretrained(model_res)

model = EncoderDecoderModel.from_pretrained(model_res)
roberta_shared= model
model.to("cuda")


# Generate the text without setting a decoding strategy
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    #outputs = roberta_shared.generate(input_ids, attention_mask=attention_mask)
    outputs = roberta_shared.generate(input_ids, attention_mask=attention_mask)

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    batch["pred"] = output_str

    return batch

# Generate a text using beams search
def generate_summary_beam_search(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = roberta_shared.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=15,
        repetition_penalty=3.0, 
        length_penalty=2.0, 
        num_return_sequences = 1
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    extents= []
    for i, y in enumerate(outputs):
        res= []
        s1= (y == 5).nonzero(as_tuple=True)[0]
        if len(s1)>0:
            s1= s1[0] -2
            if s1==0:
                s1= 1
            try:
                s2= (y == 9).nonzero(as_tuple=True)[0][0]
                if s2-s1>1:
                    s3= (y == 10).nonzero(as_tuple=True)[0][0]
                    delta= s3- s2 -1
                    res.append(tokenizer.decode(input_ids[i][s1: s1+ delta], skip_special_tokens=True))
            except:
                print(output_str[i])
        extents.append(res)

    batch["pred"] = output_str
    batch["extent"] = extents

    return batch

# Generate a text using beams search
def generate_summary_topk(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = roberta_shared.generate(
        input_ids,
        attention_mask=attention_mask,
        repetition_penalty=3.0, 
        length_penalty=2.0, 
        num_return_sequences = 1,
        do_sample=True,
        top_k=50, 
        top_p=0.95
    )

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)

    batch["pred"] = output_str

    return batch

batch_size = 4

#results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["text"])
# Generate predictions using beam search
results = test_data.map(generate_summary_beam_search, batched=True, batch_size=batch_size, remove_columns=["text"])
results.to_csv('./result_gg.csv')
pred_str_bs = results["pred"]

# # Generate predictions using top-k sampling
# results = test_data.map(generate_summary_topk, batched=True, batch_size=batch_size, remove_columns=["text"])
# pred_str_topk = results["pred"]
# results.to_csv('./result_tk.csv')