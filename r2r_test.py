import datasets
import transformers
import pandas as pd
from datasets import Dataset
import os
import tqdm
import math
import torch

from transformers import RobertaTokenizerFast, EncoderDecoderModel


#Set the path to the dataset
root_folder = './'
model_res = os.path.abspath(os.path.join(root_folder, 'model/Result/checkpoint-67584'))


# checkpoint_path = os.path.abspath(os.path.join(model_res,'checkpoint-3072'))
# print(checkpoint_path)

#Load the Tokenizer and the fine-tuned model
tokenizer = RobertaTokenizerFast.from_pretrained(model_res)

model = EncoderDecoderModel.from_pretrained(model_res)
roberta_shared= model
model.to("cuda")


# Generate a text using beams search
def generate_summary_beam_search(text):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    print('>', input_ids)
    outputs = roberta_shared.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=15,
        repetition_penalty=3.0, 
        length_penalty=2.0, 
        num_return_sequences = 1
    )
    print(outputs[0])

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    output_str= output_str[0]
    output_str= output_str.replace('<s>', '')
    output_str= output_str.replace('</s>', '')
    output_str= output_str.replace('<pad>', '')
    output_str= output_str.replace('<unk>', '')
    output_str= output_str.replace('<mask>', '')

    return output_str

xx= "tiếp bước ca sĩ mỹ linh, hàng loạt dự án của ca sĩ việt được thực hiện, đặc biệt đầu tháng bảy này."
print('>>\t', xx)
print('=>\t', generate_summary_beam_search(xx))
print('!Type .exit to quit')
while xx!= '.exit':
    xx = input()
    print('>>\t', xx)
    print('=>\t', generate_summary_beam_search(xx))