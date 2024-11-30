import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

file_path = "document_final.csv"
df = pd.read_csv(file_path, header=0)
# function to check if a row is valid
def is_valid_row(row):
    return isinstance(row['summary'], str) and isinstance(row['topic'], str)

# Filter rows where the summary and topic are valid
df = df[df.apply(is_valid_row, axis=1)]
df = df[['summary', 'topic']]

unique_topics = df['topic'].unique()
topic_to_index = {label: idx for idx, label in enumerate(unique_topics)}
df['label'] = df['topic'].map(topic_to_index)
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(test_df['topic'].unique())

train_queries = train_df['summary'].tolist()
train_labels = train_df['label'].tolist()
test_queries = test_df['summary'].tolist()
test_labels = test_df['label'].tolist()

# Initialize tokenizer using pretrained bert
bert_tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-small")

# Tokenization function
def convert_summary_into_tokens(texts, tokenizer, max_length = 128):
    tokens = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens = True,
        max_length = max_length,
        truncation = True,
        return_attention_mask = True,
        return_tensors = "pt",
        padding = "max_length"
    )
    return tokens['input_ids'], tokens['attention_mask']

# Encode train and test datasets
train_input_ids, train_attention_masks = convert_summary_into_tokens(train_queries, bert_tokenizer)
test_input_ids, test_attention_masks = convert_summary_into_tokens(test_queries, bert_tokenizer)

# Convert labels to tensors
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Create Dataset class
class QueryClassificationDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }
    
# Create Data Loaders for training
batch_size = 64
train_dataset = QueryClassificationDataset(train_input_ids, train_attention_masks, train_labels)
test_dataset = QueryClassificationDataset(test_input_ids, test_attention_masks, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset))
test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset))

# Initialize the model
model = BertForSequenceClassification.from_pretrained("prajjwal1/bert-small", num_labels=len(unique_topics))

# Optimizer and Scheduler
lr = 1e-6
epochs = 3

optimizer = AdamW(model.parameters(), lr=lr)

# Training the Model 
def train_model(data_loader):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        outputs = model(input_ids= input_ids, attention_mask=attention_mask, labels=labels)
        current_loss = outputs.loss
        total_loss += current_loss.item()
        optimizer.zero_grad()
        current_loss.backward()
        optimizer.step()
    
    return total_loss / len(data_loader)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss = train_model(train_loader)
    print(f"Training Loss: {train_loss:.4f}")


# Evaluating the model

def evaluate_model(data_loader):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    return avg_loss, predictions, true_labels

test_loss, test_predictions, test_true_labels = evaluate_model(test_loader)
print(f"Test Loss: {test_loss:.4f}")

# Classification report
print("Classification Report:")
print(classification_report(test_true_labels, test_predictions, target_names=unique_topics))

# Confusion matrix
cm = confusion_matrix(test_true_labels, test_predictions, normalize="true")
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_topics)
disp.plot(cmap="Blues", values_format=".2f")
plt.title("Confusion Matrix")
plt.show()

# Save the model
model.save_pretrained("bert_topic_classifier")
bert_tokenizer.save_pretrained("bert_topic_classifier")