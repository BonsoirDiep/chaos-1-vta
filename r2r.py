import datasets
import transformers
import pandas as pd
from datasets import Dataset
import os
import tqdm
import math
import torch

from transformers import RobertaTokenizerFast, EncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
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
model_res = os.path.abspath(os.path.join(root_folder, 'model/Result'))

mlm_pretrainedmodel = os.path.abspath(os.path.join(root_folder, 'model/RoBERTaMLM'))

tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'model/TokRoBERTa2'))

datafile= 'zip4.res.json'


datafile_path = os.path.abspath(os.path.join(data_folder,datafile))


# Load the dataset from a CSV file
df=me_read_json(datafile_path)
print('Num Examples: ',len(df))
print('Null Values\n', df.isna().sum())
df.dropna(inplace=True)
print('Num Examples: ',len(df))

# Splitting the data into training and validation
# Defining the train size. So 90% of the data will be used for training and the rest will be used for validation. 
train_size = 0.99
# Sampling 90% fo the rows from the dataset
train_dataset=df.sample(frac=train_size,random_state = 42)
# Reset the indexes
val_dataset=df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)
print('Length Train dataset: ', len(train_dataset))
print('Length Val dataset: ', len(val_dataset))

# # To limit the training and validation dataset, for testing, delete for full traning
# max_train=28393
# max_val=3155
# # Create a Dataset from a pandas dataframe for training and validation
# train_data=Dataset.from_pandas(train_dataset[:max_train])
# val_data=Dataset.from_pandas(val_dataset[:max_val])

train_data=Dataset.from_pandas(train_dataset)
val_data=Dataset.from_pandas(val_dataset)

TRAIN_BATCH_SIZE = 16   # input batch size for training (default: 64) # change to 16 for full training
VALID_BATCH_SIZE = 4    # input batch size for testing (default: 1000)
TRAIN_EPOCHS = 200       # number of epochs to train (default: 10)
VAL_EPOCHS = 1 
LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
SEED = 42               # random seed (default: 42)
MAX_LEN = 128           # Max length for text
DATETEXT_LEN = 200       # Max length for textdate

# Loading the RoBERTa Tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_folder,  max_len=200)


print(tokenizer.tokenize("""<date{\"ngyx\":\"+1\",\"thgx\":\"12\"}>chiều ngày mai/12</date> gặp lại"""))
# exit()

# model.resize_token_embeddings(len(tokenizer)) # if is pretrained_model
# ###The tokenizer has to be saved if it has to be reused
# tokenizer.save_pretrained(<output_dir>)

xx= """<date{\"ngyx\":\"+1\",\"thgx\":\"12\"}>chiều ngày mai/12</date> gặp lại"""
print(xx)
print(tokenizer(xx, padding="max_length", truncation=True, max_length=12))

# Setting the BOS and EOS token
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

batch_size=TRAIN_BATCH_SIZE
encoder_max_length=MAX_LEN
decoder_max_length=DATETEXT_LEN

def process_data_to_model_inputs(batch):
  # Tokenize the input and target data
  inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["textdate"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  # batch["decoder_input_ids"] = outputs.input_ids # remove from version v4.12.0
  # batch["decoder_attention_mask"] = outputs.attention_mask # remove from version v4.12.0
  batch["labels"] = outputs.input_ids.copy()

  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch
# Preprocessing the training data
train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "textdate"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
# Preprocessing the validation data
val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["text", "textdate"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
# Shuffle the dataset when it is needed
#dataset = dataset.shuffle(seed=42, buffer_size=10, reshuffle_each_iteration=True)

# set encoder decoder tying to True
roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained(
    mlm_pretrainedmodel,
    mlm_pretrainedmodel,
    tie_encoder_decoder=True
)
# Show the vocab size to check it has been loaded
print('Vocab Size: ', roberta_shared.config.encoder.vocab_size)

# set special tokens
roberta_shared.config.decoder_start_token_id = tokenizer.bos_token_id                                             
roberta_shared.config.eos_token_id = tokenizer.eos_token_id
roberta_shared.config.pad_token_id = tokenizer.pad_token_id

# sensible parameters for beam search
# set decoding params                               
roberta_shared.config.max_length = DATETEXT_LEN
roberta_shared.config.early_stopping = True
roberta_shared.config.no_repeat_ngram_size = 1
roberta_shared.config.length_penalty = 2.0
roberta_shared.config.repetition_penalty = 3.0
roberta_shared.config.num_beams = 10
roberta_shared.config.vocab_size = roberta_shared.config.encoder.vocab_size

# load rouge for validation
rouge = datasets.load_metric("rouge")

cc= 0
def compute_metrics(pred):
    global cc
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
    for i, el in enumerate(label_str):
        el= el.replace('<s>', '')
        el= el.replace('</s>', '')
        el= el.replace('<pad>', '')
        el= el.replace('<unk>', '')
        el= el.replace('<mask>', '')
        label_str[i]= el
    for i, el in enumerate(pred_str):
        el= el.replace('<s>', '')
        el= el.replace('</s>', '')
        el= el.replace('<pad>', '')
        el= el.replace('<unk>', '')
        el= el.replace('<mask>', '')
        pred_str[i]= el

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    # Save the predictions to a file:
    cc+= 1
    outputfile_path = os.path.abspath(os.path.join(
        os.path.abspath(os.path.join(root_folder, 'result')),
        'submission_{}.csv'.format(cc))
    )
    final_df = pd.DataFrame({'textdate':pred_str, 'label_str':label_str})
    final_df.to_csv(outputfile_path, index=False)
    print('>> Output Files {} generated for review'.format(cc))


    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=model_res,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    predict_with_generate=True,
    #evaluate_during_training=True,

    # - `"no"`: No evaluation is done during training.
    # - `"steps"`: Evaluation is done (and logged) every `eval_steps`.
    # - `"epoch"`: Evaluation is done at the end of each epoch.

    # evaluation_strategy="epoch",

    evaluation_strategy="no",

    
    do_train=True,
    do_eval=True,
    logging_steps=1024,  
    save_steps=2048, 
    warmup_steps=1024,  
    # max_steps=2, # delete for full training
    num_train_epochs = TRAIN_EPOCHS, #TRAIN_EPOCHS
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True, 
)

# instantiate trainer
trainer = Seq2SeqTrainer(
    tokenizer=tokenizer, # không có sử dụng decode, encode, chỉ kiểm tra các token đặc biệt
    model=roberta_shared,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# trainer._load_state_dict_in_model(torch.load(model_res+ '/pytorch_model.bin'))

# Fine-tune the model, training and evaluating on the train dataset
trainer.train()

# Save the encoder-decoder model just trained
trainer.save_model(model_res)

# eval_results = trainer.evaluate()
# print(eval_results)