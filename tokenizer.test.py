from transformers import RobertaTokenizerFast, RobertaTokenizer
import os

root_folder = './'
tokenizer_folder = os.path.abspath(os.path.join(root_folder, 'model/TokRoBERTa'))

tokenizer1 = RobertaTokenizerFast.from_pretrained(tokenizer_folder, max_len=512)
tokenizer2= RobertaTokenizer.from_pretrained(tokenizer_folder, max_len=512)

# test our tokenizer on a simple sentence
xx= """<date{\"ngyx\":\"+1\",\"thgx\":\"12\"}>chiều ngày mai/12</date> gặp lại"""
print(xx)
print(tokenizer1(xx))
print(tokenizer2(xx))


print(len(tokenizer2))
print(tokenizer2.unique_no_split_tokens)
# not working
# tokenizer2.add_tokens([
#     "<date ",
#     "value=\"[[[",
#     "]]]\">",
#     "</date>"
# ], special_tokens=True)

# not working for all situation
tokenizer2.add_special_tokens({
    "additional_special_tokens": [
        '<date{',
        '\"ngyx\":\"',
        '\"thgx\":\"',
        '\"namx\":\"',
        '}>',
        '</date>'
]})


print(len(tokenizer2))
print(tokenizer2.unique_no_split_tokens)
# tokenizer2.save_pretrained("./model/TokRoBERTa3/") # not working

print(len(tokenizer1))
# tokenizer1.add_tokens([
#     "<date ",
#     "value=\"[[[",
#     "]]]\">",
#     "</date>"
# ], special_tokens=True)
tokenizer1.add_special_tokens({
    "additional_special_tokens": [
        '<date{',
        '\"ngyx\":\"',
        '\"thgx\":\"',
        '\"namx\":\"',
        '}>',
        '</date>'
]})

print(len(tokenizer1))

print('#0 test after add special tokens', xx)
print(tokenizer1(xx, padding="max_length", truncation=True, max_length=40))
print(tokenizer2(xx, padding="max_length", truncation=True, max_length=40))

## save tokenizer1
tokenizer1.save_pretrained("./model/TokRoBERTa2/")
exit()


## reload
tokenizer1_new = RobertaTokenizerFast.from_pretrained(tokenizer_folder+ '2', max_len=512)
print(tokenizer1_new(xx))

exit()

# kiểm nghiệm 1 để đánh tag BIO [NOT OK]
def me_token(arr):
    class gg:
        def __init__(self, input_ids):
            self.input_ids= input_ids
        def word_ids(self):
            return [1]
    tokens= []
    tokens+= ['<s>']
    for a in arr:
        word_tokens= tokenizer1.tokenize(a)
        tokens.extend(word_tokens)
    tokens+= ['</s>']
    tokens= tokenizer1.convert_tokens_to_ids(tokens)
    return gg(input_ids=tokens)


def test(xx, tokens=None):
    print('\n>>', xx)

    print('#1 CustomDataset in mlm.py')
    print(tokenizer1.encode_plus(xx, max_length = 12, truncation=True, padding="max_length"))
    print(tokenizer1.encode_plus(xx, max_length = 12, truncation=True, padding=True))

    print('#2 process_data_to_model_inputs in r2r.py')
    print(tokenizer1(xx, padding="max_length", truncation=True, max_length=12))

    if tokens==None:
        tokens= tokenizer1(xx, padding="max_length", truncation=True, max_length=12)
    print('>tokens.word_ids', tokens.word_ids())
    tokens= tokens.input_ids
    print('>tokens', tokens)
    print('#3 compute_metrics rouge.compute')
    print(tokenizer1.batch_decode([tokens], skip_special_tokens=False))
    print(tokenizer1.batch_decode([tokens], skip_special_tokens=True))


aaa= 'Xin chào các bạn nhé!'
# aaa= 'Vào <date value="[[[1991]]]">năm 1991</date>'

test(aaa)


aa= aaa.split()
# aa= ['Vào', '<date ', 'value="[[[', '1991', ']]]">', 'năm', '1991', '</date>']

print(aaa, aa)
test(aaa, me_token(aa))

# kiểm nghiệm 2 để đánh tag BIO [OK]
def me_token_2(text, start, end, text_label):
    tok1= tokenizer1(text, padding="max_length", truncation=True, max_length=12)
    tok1= tok1[0]
    tokens = tok1.tokens
    print(tok1.ids)
    print(tokens)
    print('>>>TEST<<<')
    aligned_labels = [None]*len(tokens) # Make a list to store our labels the same length as our tokens

    for char_ix in range(start,end):
        token_ix = tok1.char_to_token(char_ix)
        # print(text[char_ix:char_ix+1], token_ix)
        if token_ix is not None: # White spaces have no token and will return None
            if aligned_labels[token_ix] is None:
                aligned_labels[token_ix] = text_label
    
    for token,label in zip(tokens,aligned_labels):
        if label is None:
            label= "O"
        print (token,"-",label)

me_token_2("""Vào <date value="[[[19_91]]]">năm 1991</date>""",
    20, # .indexOf('1991')
    20+ 5,
    "VALUE"
)

# https://www.lighttag.io/blog/sequence-labeling-with-transformers/example
text = "I am Tal Perry, founder of LightTag"
annotations = [
    dict(start=5,end=14,text="Tal Perry",label="Person"),
    dict(start=16,end=23,text="founder",label="Title"),
    dict(start=27,end=35,text="LightTag",label="Org")]