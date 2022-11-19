from transformers import pipeline, RobertaTokenizerFast, AutoModelForMaskedLM
import os
import torch

#Set the path to the data folder, datafile and output folder and files
root_folder = './'

model_folder = os.path.abspath(os.path.join(root_folder, 'model/RoBERTaMLM'))

tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'model/TokRoBERTa2'))

tokenizer= RobertaTokenizerFast.from_pretrained(tokenizer_folder)

xx= """<date{\"ngyx\":\"+1\",\"thgx\":\"12\"}>chiều ngày mai/12</date> gặp lại"""
print(xx)
print(tokenizer(xx, padding="max_length", truncation=True, max_length=12))



# <mask>


model= AutoModelForMaskedLM.from_pretrained(model_folder)
print(model.config)

# print(model.roberta)

fill_mask = pipeline(
    "fill-mask",
    model=model,
    tokenizer=tokenizer
)

a= fill_mask("""tại thị trường quốc <mask>, king kong thu về 80 triệu usd.""")
for i in a:
    print(i)

print('----')

a= fill_mask("""Chúng tôi chẳng ăn gì hai ba ngày <mask>.""")
for i in a:
    print(i)

print('----')

a= fill_mask("""Chúng tôi chẳng ăn gì hai ba <mask> nay rồi.""")
for i in a:
    print(i['sequence'])


xx= """Chúng tôi chẳng ăn gì hai ba <mask> nay rồi."""
# xx= """Thủ <mask> ra nghị quyết vào sáng nay."""

xx= """Chúng tôi chẳng ăn gì hai ba ngày <mask> rồi."""

xx= tokenizer(xx, padding="max_length", truncation=True, max_length=30, return_tensors='pt')

## convert to tensor
# xx= tokenizer(xx, padding="max_length", truncation=True, max_length=30)
# xx = {k:torch.tensor([v]) for k,v in xx.items()}

word_ids = xx.word_ids()
print(word_ids)


# o1= model.roberta(xx['input_ids'])[0] # sequence_output, pooled_output, (hidden_states), (attentions)
# sequence_output= o1[0]
# print(sequence_output.shape)


input_seq= xx['input_ids']
print(input_seq)
mask_token_index = torch.where(input_seq==tokenizer.mask_token_id)[1]

token_logits = model(input_seq).logits
print('token_logits', token_logits.shape)
masked_token_logits = token_logits[0, mask_token_index, :]

top_5_tokens = torch.topk(masked_token_logits, 5, dim=1).indices[0].tolist()
for token in top_5_tokens:
    print(token, tokenizer.decode([token]))