## Step 1: Install pytorch pretrained BERT from source


```bash
git clone https://github.com/huggingface/pytorch-pretrained-BERT.git
cd pytorch-pretrained-BERT
python setup.py install
```

## Step 2: Move file

```bash
mv run_classifier.py tfidf2bert.py postprocess.py -t pytorch-pretrained-BERT/
```

## Setp 3: Move dataset to pytorch-pretrained-BERT/

### As below...

pytorch-pretrained-BERT/run_classifier.py

pytorch-pretrained-BERT/tiidf2bert.py

pytorch-pretrained-BERT/data/NC_1.csv

pytorch-pretrained-BERT/data/QS_1.csv

pytorch-pretrained-BERT/data/url2content.json

…


## Step 4: Preprocess for your answer

```bash
python3 tfidf2bert.py --data_path data --ans_path ans.csv
```

## Step 5: Finetune and Testing (output_dir need to be empty before Finetune) 

```bash
python3 run_classifier.py \
--task_name sts-b \
--do_train \
--do_eval \
--do_predict \
--do_lower_case \
--data_dir data/ \
--bert_model bert-base-chinese \
--max_seq_length 512 \
--train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3.0 \
--output_dir output/
```

## Step 6: Postprocess 

```bash
python3 postprocess.py --ans_path ans.csv --sorted_ans_path sorted_ans.csv
```


# Finish!