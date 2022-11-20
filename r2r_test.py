import sys
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
model_res = os.path.abspath(os.path.join(root_folder, 'model/Result'))


# checkpoint_path = os.path.abspath(os.path.join(model_res,'checkpoint-3072'))
# print(checkpoint_path)

#Load the Tokenizer and the fine-tuned model
tokenizer = RobertaTokenizerFast.from_pretrained(model_res)

model = EncoderDecoderModel.from_pretrained(model_res)
roberta_shared= model
model.to("cuda")

def generate_date_tag_str(text):
    s= None
    e= []
    m= [] # start, end, mid
    start_append= -1
    # create ids of encoded input vectors
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to('cuda')
    
    res= input_ids[0].clone().tolist()
    print(res)

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
        if(next_decoder_input_ids.item()==5):
            # start_append= i-1
            start_append= i
            s= []
            s.append(next_decoder_input_ids.item())
        elif(next_decoder_input_ids.item()==9):
            s.append(next_decoder_input_ids.item())
            res= res[:start_append]+ s+ res[start_append:]
            s= None
            start_append= -1
        elif(next_decoder_input_ids.item()==10):
            res= res[:i]+ [10]+ res[i:]
        elif start_append> -1:
            s.append(next_decoder_input_ids.item())
    print(res)
    print(decoder_input_ids[0].tolist())
    print('=>\t', ('Recommend', tokenizer.decode(res, skip_special_tokens=False)))
    return ('Seq2seq', tokenizer.decode(decoder_input_ids[0], skip_special_tokens=False))

batch_size = 4

# Generate a text using beams search
def generate_date_tag_beam_search_str(text):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    # cut off at BERT max length 512
    inputs = tokenizer([text], padding="max_length", truncation=True, max_length=200, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    # print('>', input_ids)
    outputs = roberta_shared.generate(
        input_ids,
        attention_mask=attention_mask,
        num_beams=15,
        repetition_penalty=3.0, 
        length_penalty=2.0, 
        num_return_sequences = 1
    )
    # print(outputs[0])

    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=False)
    output_str= output_str[0]
    output_str= output_str.replace('<s>', '')
    output_str= output_str.replace('</s>', '')
    output_str= output_str.replace('<pad>', '')
    output_str= output_str.replace('<unk>', '')
    output_str= output_str.replace('<mask>', '')

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
    print(extents)

    return output_str

if len(sys.argv)>1 and sys.argv[1]=='bs':
    gen= generate_date_tag_beam_search_str
elif len(sys.argv)>1 and sys.argv[1]=='bio':
    gen= generate_date_tag_str
else:
    exit()

xx= "tiếp bước ca sĩ mỹ linh, hàng loạt dự án của ca sĩ việt được thực hiện, đặc biệt đầu tháng bảy này."
print('>>\t', xx)
print('=>\t', gen(xx))
print('!Type .exit to quit')
while xx!= '.exit':
    xx = input()
    print('>>\t', xx)
    print('=>\t', gen(xx))