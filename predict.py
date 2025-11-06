#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# In[5]:


try :
    from config import model_save_path
except ImportError:
    print("Error : config.py")
    model_save_path = "./saved_model"

print("loading model")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)
except EnvironmentError:
    print("model not found")
    exit()
id2label = model.config.id2label




# In[8]:


def classify_text(text):
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation = True,
        padding = True,
        max_length=512
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs=model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits,dim=1).item()
    probablities = torch.softmax(logits,dim=1)
    confidence = probablities[0][predicted_class_id].item()
    predicted_label = id2label[predicted_class_id]
    return predicted_label, confidence

    


# In[ ]:


if __name__ == "__main__" :
    text1 = input("Put a Text : ")
    text2 = input("Put Another Text : ")
    label,score = classify_text(text1)
    print("Result")
    print(f"Text : {text1}")
    print(f"Label : {label}")
    print(f"Confidence : {score:.4f}")

    label2,score2 = classify_text(text2)
    print("Result")
    print(f"Text : {text2}")
    print(f"Label : {label2}")
    print(f"Confidence : {score2:.4f}")
    

