#!/usr/bin/env python
# coding: utf-8

# In[1]:


import evaluate
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


# In[2]:


from config import (
    model_name,
    model_save_path,
    num_epochs,
    alpha,
    per_device_train_batch_size,
    per_device_eval_batch_size,
    weight_decay
)


# In[4]:


from dataset import NewsgroupDataset
def load_and_prepare_data(tokenizer):
    print("Loading dataset")

    remove_headers=('headers','footers','quotes')

    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=remove_headers
    )

    newsgroups_test = fetch_20newsgroups(
        subset='test',
        remove=remove_headers
    )

    train_texts=newsgroups_train.data
    train_labels=newsgroups_train.target
    test_texts=newsgroups_test.data
    test_labels=newsgroups_test.target
 
    labels_list=newsgroups_train.target_names
    print("Tokenizing Data")
    train_encodings = tokenizer(train_texts,truncation=True,padding=True,max_length=512)
    test_encodings = tokenizer(test_texts,truncation = True,padding = True , max_length=512)
    print("Creating custom PyTorch Datasets")
    train_dataset=NewsgroupDataset(train_encodings,train_labels)
    test_dataset=NewsgroupDataset(test_encodings,test_labels)
    return train_dataset,test_dataset,labels_list

def compute_metric (eval_pred):
    accuracy_metric= evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits,axis=1)
    return accuracy_metric.compute(predictions=predictions, references = labels)
def main():
    print(f"Loading Tokenizer : {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset, test_dataset,labels_list = load_and_prepare_data(tokenizer)
    num_labels = len(labels_list)
    print(f"Found {num_labels} topics/labels.")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    model.config.id2label = { i : label for i, label in enumerate(labels_list)}
    model.config.label2id = {label: i for i ,label in enumerate (labels_list)}

    print("Defining Training Arguement")
    training_args= TrainingArguments(
        output_dir='./results',
        num_train_epochs=num_epochs,
        learning_rate=alpha,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        eval_strategy='epoch',
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir='./logs',
        logging_steps=100,
    )

    print("Initializing Trainer")
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metric,
    )

    print("starting")
    trainer.train()
    print("Complete")
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print('Done')
if __name__=="__main__":
    main()

    


# In[ ]:




