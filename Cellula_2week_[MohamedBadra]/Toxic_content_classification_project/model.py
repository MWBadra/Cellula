import io
import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer,AutoModelForSeq2SeqLM
from peft import LoraModel, LoraConfig,get_peft_model,TaskType,PeftModel
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1) # Convert probabilities to class numbers
    
    # Calculate accuracy
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

   
df = pd.read_csv('D:/books and slides/Cellula26/Cellula/Cellula_2week_[MohamedBadra]/Toxic_content_classification_project/data.csv')

df = df.drop_duplicates(subset=['image descriptions', 'Toxic Category'])
print(df)

unique_cats = df['Toxic Category'].unique()
print("Unique Categories:", unique_cats)

print("\nCounts per Category:")
print(df['Toxic Category'].value_counts())


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",num_labels=9)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)


lora_config = LoraConfig(
    r=8, # Rank Number
    lora_alpha=32, # Alpha (Scaling Factor)
    lora_dropout=0.05, # Dropout Prob for Lora
    target_modules=["q_lin", "k_lin","v_lin"], # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
    bias='none',
    task_type=TaskType.SEQ_CLS # Seqence to Classification Task
)

peft_model = get_peft_model(model, 
                            lora_config)

# Reduced trainble parameters
print(print_number_of_trainable_model_parameters(peft_model))

class ToxicDataset(Dataset):
    
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
       item = {key: val[idx] for key, val in self.encodings.items()}
       item['labels'] = torch.tensor(self.labels[idx])
       return item
    
label2id = {label: i for i, label in enumerate(unique_cats)}


train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_encodings = tokenizer(train_df['image descriptions'].tolist(), truncation=True, padding=True, return_tensors="pt")
test_encodings = tokenizer(test_df['image descriptions'].tolist(), truncation=True, padding=True, return_tensors="pt")


train_labels = train_df['Toxic Category'].map(label2id).tolist()
test_labels = test_df['Toxic Category'].map(label2id).tolist()

train_dataset = ToxicDataset(train_encodings, train_labels)
test_dataset = ToxicDataset(test_encodings, test_labels)

print(len(train_dataset))

print(len(test_dataset))


training_args = TrainingArguments(
    output_dir="./lora_toxic_results",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,   
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",        
    save_strategy="epoch",
    load_best_model_at_end=True,   
    remove_unused_columns=False
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,       
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
print("Starting Training...")
trainer.train()

peft_model.save_pretrained("final_toxic_model")
print("Model saved to folder 'final_toxic_model'")
results = trainer.evaluate()
print(results)