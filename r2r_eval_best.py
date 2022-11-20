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


def text_gen(text):
    # create ids of encoded input vectors
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to('cuda')
    # create BOS token
    decoder_input_ids = tokenizer("<s>", add_special_tokens=False, return_tensors="pt").input_ids.to('cuda')
    assert decoder_input_ids[0, 0].item() == model.config.decoder_start_token_id, "`decoder_input_ids` should correspond to `model.config.decoder_start_token_id`"
    for i in range(200):
        # pass input_ids to encoder and to decoder and pass BOS token to decoder to retrieve first logit
        outputs = model(input_ids, decoder_input_ids=decoder_input_ids, return_dict=True)
        # get encoded sequence
        # encoded_sequence = (outputs.encoder_last_hidden_state,)
        # get logits
        lm_logits = outputs.logits
        # sample last token with highest prob
        next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
        # concat
        decoder_input_ids = torch.cat([decoder_input_ids, next_decoder_input_ids], axis=-1)
        if(next_decoder_input_ids.item()==2):
            break
    return decoder_input_ids[0]

def generate_date_tag(batch):
    output_str= []
    for a in batch["text"]:
        output_str.append(text_gen(a))
    batch["pred_text"]= tokenizer.batch_decode(output_str, skip_special_tokens=False)
    batch["pred"] = output_str
    return batch

batch_size = 4

# Generate predictions using beam search
results = test_data.map(generate_date_tag, batched=True, batch_size=batch_size, remove_columns=["text"])
results.to_csv('./score/result_gg.csv')
pred_str_bs = results["pred"]
