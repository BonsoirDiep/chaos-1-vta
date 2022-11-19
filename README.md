# Vietnamese Temporal Annotation 

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