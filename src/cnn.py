import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNClassifier(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 num_classes=6,
                 pad_idx=0,
                 filter_sizes=(3,4,5),
                 num_filters=100,
                 dropout=0.5
                 ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=k) for k in filter_sizes
        ])

        self.dense = nn.Linear(num_filters * len(filter_sizes), num_classes)
        self.drop  = nn.Dropout(dropout)


    def forward(self, ids):
        flattened_embeddings = self.embedding(ids).transpose(1, 2)
        features = [F.max_pool1d(F.relu(c(flattened_embeddings)), c(flattened_embeddings).size(2)).squeeze(2) for c in self.convs]

        output = torch.cat(features, 1)
        return self.dense(self.drop(output))
