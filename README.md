# Vietnamese Temporal Annotation 
- package:
	+ python
		+ tokenizers==0.13.1
		+ transformers==4.16.2
		+ @demdecuong vi-nlp-core==1.1.12 (require pyahocorasick==1.4.4)
		+ @nguyenvulebinh regtag==0.4.0.1 (regex bio taggen)
	+ nodejs
		+ cheerio
#### TokRoBERTa2/vocab.json
```
...
"<date{": 5,
"\"ngyx\":\"": 6,
"\"thgx\":\"": 7,
"\"namx\":\"": 8,
"}>": 9,
"</date>": 10,
...
```
### Mix loss (.. site-packages\transformers\models\encoder_decoder\modeling_encoder_decoder.py)
- line 534: loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.view(-1))
- mix:
```python
input1= logits.reshape(-1, self.decoder.config.vocab_size)
target1= labels.view(-1)
idx_start= 0
multi_x= 1
for i in range(target1.shape[0]):
	if target1[i]==5 or target1[i]==9 or target1[i]==10:
		if target1[i]==5:
			multi_x= 1
		elif target1[i]==9:
			multi_x= 4
		elif target1[i]==10:
			multi_x= 2
		if loss is not None:
			loss += multi_x*loss_fct(input1[idx_start:i+1], target1[idx_start:i+1])
		else:
			loss = multi_x*loss_fct(input1[idx_start:i+1], target1[idx_start:i+1])
	elif i== target1.shape[0]-1:
		multi_x= 1
		if loss is not None:
			loss += multi_x*loss_fct(input1[idx_start:], target1[idx_start:])
		else:
			loss = multi_x*loss_fct(input1[idx_start:], target1[idx_start:])
```

### Tokenizer, Masked-Language Modeling and RoBERTa2RoBERTa Model
```bash
python tokenizer.py
python tokenizer.test.py
python mlm.py
python r2r.py
python r2r_test.py
```

### Eval & Score
- We have 2 dataset for test.
- Copy datasets/test1/zip4.test.json to datasets/ or copy datasets/test1/zip4.test.json to datasets/
#### r2r_eval_best.py
```bash
python r2r_eval_best.py
bash score/s1.sh
```

#### r2r_eval_bs.py
```bash
python r2r_eval_bs.py
bash score/s1.sh
```