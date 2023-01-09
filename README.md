# Vietnamese Temporal Annotation
- Model: https://drive.google.com/drive/folders/1iHYeZpKOjGmwD0BlqeywcgPNwVThAlzN
- Text2text
  - Input:
      + hơn 100 năm qua thế giới đã thay đổi nhiều hơn bất kỳ thế kỷ nào trong lịch sử.
  - Output:
      + `<date{"namx":"-100"}>`hơn 100 năm qua`</date>` thế giới đã thay đổi nhiều hơn bất kỳ thế kỷ nào trong lịch sử.
- Run with "python r2r_test.py bio":
	- Input
		+ "biến thể mới của sars-cov-2 được dự đoán sẽ là chủng vượt trội ở mỹ sau 5 tuần nữa."
	- Ouput:
		- ('Recommend', `'<s>`<b>biến</b> thể mới của sars-cov-2 được dự đoán sẽ là chủng vượt trội ở mỹ`<date{"ngyx":"+35"}>` sau 5 tuần nữa.`</date></s>'`)
		- ('Seq2seq', ```'<s><s>```<b>b đán</b> thể mới của sars-cov-2 được dự đoán sẽ là chủng vượt trội ở mỹ `<date{"ngyx":"+35"}>`sau 5 tuần nữa`</date>`.`</s>`')

### Package
- package:
	+ python: main
		+ tokenizers==0.13.1
		+ transformers==4.16.2
	+ python: refer to similar work
		+ @demdecuong vi-nlp-core==1.1.12 (require pyahocorasick==1.4.4)
		+ @nguyenvulebinh regtag==0.4.0.1 (regex bio taggen)
	+ nodejs
		+ csv-parse
		+ cheerio
		+ romans

### TokRoBERTa2/vocab.json
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

### Tokenizer, Masked-Language Modeling and RoBERTa2RoBERTa Model
- Step by step:
```bash
python tokenizer.py
python tokenizer.test.py
python mlm.py
python r2r.py
```
- Test good at many tag <date>:
```bash
python r2r_test.py bio
```
- Test with beam search good at most 1 tag <date>
```bash
python r2r_test.py bs
```
### Eval & Score
- We have 2 dataset for test.
- Copy datasets/test1/zip4.test.json to datasets/ or copy datasets/test1/zip4.test.json to datasets/
#### r2r_eval_best.py
```bash
python r2r_eval_best.py
bash score/s1.sh # use https://cmder.app/ for window (bash, curl, git and more...)
```

#### r2r_eval_bs.py
```bash
python r2r_eval_bs.py
bash score/s1.sh
```

### More:
- Data Generation:
```bash
node data_generation\zip4.js train
node data_generation\zip4.js train
...
node data_generation\zip4.js test
...
```
- Copy data_generation/zip4.res.json, data_generation/zip4.test.json to folder /datasets