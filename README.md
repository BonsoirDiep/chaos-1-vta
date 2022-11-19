# Vietnamese Temporal Annotation 
	package:
		+ python
			+ tokenizers==0.13.1
			+ transformers==4.16.2
			+ @demdecuong vi-nlp-core==1.1.12 (require pyahocorasick==1.4.4)
			+ @nguyenvulebinh regtag==0.4.0.1 (regex bio taggen)
		+ nodejs
			+ cheerio
### Tokenizer, Masked-Language Modeling and RoBERTa2RoBERTa Model
```bash
python tokenizer.py
python tokenizer.test.py
python mlm.py
python r2r.py
python r2r_test.py
```

### Eval & Score
	We have 2 dataset for test.
	Copy datasets/test1/zip4.test.json to datasets/ or copy datasets/test1/zip4.test.json to datasets/
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