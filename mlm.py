import os
import pandas as pd
import tqdm
import math

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import torch
from torch.utils.data.dataset import Dataset
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
root_folder = './'
data_folder = os.path.abspath(os.path.join(root_folder, 'datasets'))
model_folder = os.path.abspath(os.path.join(root_folder, 'model/RoBERTaMLM'))

tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'model/TokRoBERTa2'))

test_filename='zip4.test.json'
datafile= 'zip4.res.json'


datafile_path = os.path.abspath(os.path.join(data_folder,datafile))
testfile_path = os.path.abspath(os.path.join(data_folder,test_filename))

# Load the train dataset
train_df=me_read_json(datafile_path)
# Show the count of rows
print('Num Examples: ',len(train_df))
print('Null Values\n', train_df.isna().sum())
# Drop rows with Null values 
train_df.dropna(inplace=True)
print('Num Examples: ',len(train_df))

# Load the test dataset 
test_df=me_read_json(testfile_path)
print('Num Examples: ',len(test_df))
print('Null Values\n', test_df.isna().sum())
# there are no null values

# # Create the tokenizer using vocab.json and mrege.txt files
# tokenizer = ByteLevelBPETokenizer(
#     os.path.abspath(os.path.join(tokenizer_folder,'vocab.json')),
#     os.path.abspath(os.path.join(tokenizer_folder,'merges.txt'))
# )
# # Prepare the tokenizer
# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)
# print(tokenizer.encode("""<date value="[[[1991]]]">năm 1991</date>""").tokens)


TRAIN_BATCH_SIZE = 16    # input batch size for training (default: 64)
VALID_BATCH_SIZE = 8    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 15        # number of epochs to train (default: 10)
LEARNING_RATE = 1e-4    # learning rate (default: 0.001)
WEIGHT_DECAY = 0.01
SEED = 42               # random seed (default: 42)
MAX_LEN = 200


from transformers import RobertaTokenizerFast
# Create the tokenizer from a trained one
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=MAX_LEN)

print(tokenizer.encode_plus(
    """<date{\"ngyx\":\"+1\",\"thgx\":\"12\"}>chiều ngày mai/12</date> gặp lại""",
    max_length = MAX_LEN,
    truncation=True,
    padding=True
))
# exit()

print('>> len(tokenizer):', len(tokenizer))
from transformers import RobertaConfig

config = RobertaConfig(
    vocab_size=len(tokenizer),
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)
print('Num parameters: ',model.num_parameters())

class CustomDataset(Dataset):
    def __init__(self, pd_df, tokenizer):
        # or use the RobertaTokenizer from `transformers` directly.

        self.examples = []

        df= pd_df['text']
        for example in df.values:
            x=tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding=True)
            self.examples += [x.input_ids]
        # df= pd_df['textdate']
        # for example in df.values:
        #     x=tokenizer.encode_plus(example, max_length = MAX_LEN, truncation=True, padding=True)
        #     self.examples += [x.input_ids]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

# Create the train and evaluation dataset
train_dataset = CustomDataset(train_df, tokenizer)
eval_dataset = CustomDataset(test_df, tokenizer)

from transformers import DataCollatorForLanguageModeling

# Define the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, # không có sử dụng decode, encode, chỉ kiểm tra các token đặc biệt, và tokenizer.pad()
    mlm=True,
    mlm_probability=0.15
)

from transformers import Trainer, TrainingArguments

print(model_folder)
# Define the training arguments
training_args = TrainingArguments(
    output_dir=model_folder,
    overwrite_output_dir=True,

    evaluation_strategy = 'epoch',
    # evaluation_strategy = 'no',

    num_train_epochs=TRAIN_EPOCHS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=VALID_BATCH_SIZE,
    save_steps=8192,
    #eval_steps=4096,
    save_total_limit=1,
    # max_steps=30_000 # when have check point
)
# Create the trainer for our model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    #prediction_loss_only=True,
)


## when have check point
# checkpoint= model_folder+ '/checkpoint-49152/'
# state_dict = torch.load(checkpoint+ 'pytorch_model.bin')
# trainer._load_state_dict_in_model(state_dict)

# Train the model
trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.save_model(model_folder)