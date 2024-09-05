import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

# specify GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#On importe le dataset, si spam label = 1 sinon label = 0, encodage latin-1
df = pd.read_csv("./spam.csv", encoding="latin-1", usecols=["v1", "v2"])
df = df.rename(columns={"v1": "label", "v2": "text"})
df["label"] = df["label"].map({"spam": 1, "ham": 0})
print(df.head())

# On vérifie la distribution des labels
distribution = df['label'].value_counts(normalize = True)
print(distribution)

# On divise le dataset en train (70%), validation (15%) et test (15%)
train_text, rest_text, train_labels, rest_labels = train_test_split(
    df['text'], df['label'], 
    random_state=1234, # Seed pour la reproductibilité, on évite un biais
    test_size=0.3, 
    stratify=df['label']
)

val_text, test_text, val_labels, test_labels = train_test_split(
    rest_text, rest_labels, 
    random_state=5678, 
    test_size=0.5, 
    stratify=rest_labels
)

# On importe de huggingface un modèle pre-entrainé et son tokenizer
# bert-base-uncased est un modèle BET de taille standard avec 12 couches, 768 dimensions cachées et 12 têtes d'attention
# Toutes les lettres sont en minuscules et les accents sont supprimés
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased', clean_up_tokenization_spaces=False)

# Historigramme de la longueur des textes
seq_len = [len(i.split()) for i in train_text]

pd.Series(seq_len).hist(bins = 50)

plt.show()

tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True,
)

tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = 25,
    pad_to_max_length=True,
    truncation=True,
)

tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = 25,
    pad_to_max_length = True,
    truncation = True,
)

print(tokens_train[0])

train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids'])
test_mask = torch.tensor(tokens_test['attention_mask'])
test_y = torch.tensor(test_labels.tolist())

print(train_seq.shape)
print(train_mask.shape)
print(train_y.shape)

print(val_seq.shape)
print(val_mask.shape)
print(val_y.shape)

print(test_seq.shape)
print(test_mask.shape)
print(test_y.shape)

batch_size = 32

train_data = TensorDataset(train_seq, train_mask, train_y)

train_sampler = RandomSampler(train_data)

train_dataloader = DataLoader(train_data, sampler = train_sampler, batch_size = batch_size)

val_data = TensorDataset(val_seq, val_mask, val_y)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size = batch_size)

for param in bert.parameters():
    param.requires_grad = False

class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)
    
model = BERT_Arch(bert)
model = model.to(device)
