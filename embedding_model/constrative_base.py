import time
import random
import numpy as np
import argparse
import os
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score, recall_score, accuracy_score
from tqdm import tqdm

# Constants
HIDDEN_SIZE = 768
TEMP = 0.07
MODEL_TYPE =  #'sentence-transformers/all-mpnet-base-v2',
TQDM_DISABLE = False

# Model Definition
class ContrastiveModel(nn.Module):
    def __init__(self, config, ):
        super(ContrastiveModel, self).__init__()
        self.model = AutoModel.from_pretrained(config.model_name)
        self.cos = torch.nn.CosineSimilarity()
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = torch.nn.Dropout(config.hidden_dropout_prob)

    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling over token embeddings.
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids_1, attention_mask_1):
        """
        Forward pass for a single input.
        """
        output1 = self.model(input_ids_1, attention_mask_1)
        return self.mean_pooling(output1['last_hidden_state'], attention_mask_1)

    def forward_train(self, input_ids_1, attention_mask_1):
        """
        Forward pass during training.
        """
        output1 = self.model(input_ids_1, attention_mask_1)
        output2 = self.model(input_ids_1, attention_mask_1)
        average_embs1 = self.mean_pooling(output1['last_hidden_state'], attention_mask_1)
        average_embs2 = self.mean_pooling(output2['last_hidden_state'], attention_mask_1)
        return average_embs1, average_embs2

    def predict_scores(self, input_ids_1, attention_mask_1):
        """
        Predict cosine similarity scores between embeddings.
        """
        average_embs1, average_embs2 = self.forward_train(input_ids_1, attention_mask_1)
        scores = torch.stack([
            self.cos(a_i.reshape(1, a_i.shape[0]), average_embs2) / TEMP
            for a_i in average_embs1
        ])
        return scores

    def forward_cos(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        """
        Compute cosine similarity between two input embeddings.
        """
        output1 = self.model(input_ids_1, attention_mask_1)
        output2 = self.model(input_ids_2, attention_mask_2)
        average_embs1 = self.mean_pooling(output1['last_hidden_state'], attention_mask_1)
        average_embs2 = self.mean_pooling(output2['last_hidden_state'], attention_mask_2)
        return self.cos(average_embs1, average_embs2)

    def __getattr__(self, name: str):
        """
        Forward missing attributes to the wrapped model.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
