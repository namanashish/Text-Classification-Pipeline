#!/usr/bin/env python
# coding: utf-8

# In[ ]:


model_name = 'distilbert-base-uncased'
model_save_path='./saved_model'
num_epochs=4
alpha = 2e-5
per_device_train_batch_size=16
per_device_eval_batch_size=16
weight_decay=0.01

