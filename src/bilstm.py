import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):

    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 hidden_dim=256,
                 num_layers=2,
                 num_classes=6,
                 pad_idx=0,
                 dropout=0.5):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=True, batch_first=True,
                            dropout=0.0 if num_layers == 1 else dropout
                            )
        self.dense = nn.Linear(hidden_dim * 2, num_classes)
        self.drop  = nn.Dropout(dropout)

    def forward(self, ids):
        #embed token ids
        embedded = self.embedding(ids)

        #run through LSTM
        _, (hidden, _) = self.lstm(embedded)

        #concat forward and backward hidden states
        h_cat = torch.cat((hidden[-2], hidden[-1]), 1)

        #return final logits
        return self.dense(self.drop(h_cat))
