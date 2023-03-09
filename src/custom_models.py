import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import numpy as np
import argparse
from itertools import product

from torch.utils.data import DataLoader
from torch import nn
import torch

import transformers
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.deberta.modeling_deberta import ContextPooler, StableDropout
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.activations import ACT2FN

from transformers.utils import logging
logging.set_verbosity(40)

T5_MODEL_NAMES = ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl", "google/ul2", "google/flan-ul2"]

class DebertaV3ForSequenceClassificationCustomClassificationToken(nn.Module):
    def __init__(self, token_pos: list, model_name: str, cache_dir=None, dropout: float = 0.0):
        super().__init__()
        if cache_dir is not None:
            self.full = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1,cache_dir=cache_dir).cuda()
            self.part = AutoModel.from_pretrained(model_name, num_labels=1,cache_dir=cache_dir).cuda()
        else:
            self.full = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).cuda()
            self.part = AutoModel.from_pretrained(model_name, num_labels=1).cuda()
        self.token_pos = token_pos
        self.dropout = StableDropout(dropout) 

    def forward(self, input_ids: torch.Tensor, attention_mask):
        out1 = self.part(input_ids, attention_mask)
        # out2 = out1.last_hidden_state[:, self.token_pos, :]
        out2 = out1.last_hidden_state[:, 0, :]
        pooler = ContextPooler(self.part.config).cuda()
        out3 = pooler(out2)
        out4 = self.dropout(out3)
        classifier = self.full.classifier
        logits = classifier(out4)

        return SequenceClassifierOutput(logits=logits, hidden_states=out1.hidden_states, attentions=out1.attentions)

class ContextPoolerCustomToken(nn.Module):
    def __init__(self, config, custom_tokens: list, dropout: float = None):
        super().__init__()
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        
        if dropout is None:
            self.dropout = StableDropout(config.pooler_dropout)
        else:
            self.dropout = StableDropout(dropout)
        self.config = config
        self.custom_tokens = custom_tokens

    def forward(self, hidden_states):

        # We "pool" the model by simply taking the hidden state corresponding
        # to the token_pos tokens

        
        context_token = hidden_states[:, self.custom_tokens]
        context_token = context_token.mean(dim=1)
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size

class FrozenWrapper(nn.Module):
    def __init__(self, config, model, custom_tokens: list, dropout: float = None):
        super().__init__()
        self.model = model.eval()
        self.dense = nn.Linear(config.pooler_hidden_size, config.num_labels)
        
        if dropout is None:
            self.dropout = StableDropout(config.pooler_dropout)
        else:
            self.dropout = StableDropout(dropout)
        self.config = config
        self.custom_tokens = custom_tokens or [0]

    def forward(self, *inputs, **kwargs):
        
        hidden_states = self.model(output_hidden_states=True, *inputs, **kwargs).hidden_states[-1] # last layer
        context_token = hidden_states[:, self.custom_tokens]
        context_token = context_token.mean(dim=1)
        pooled_output = self.dropout(context_token)
        return (self.dense(pooled_output), )
    
    # modify the eval function
    def train(self, mode):
        self.model.deberta.embeddings.train(mode)

class EncoderWithLogitHead(nn.Module):
    """Adds a logit head to an encoder model. Works with T5 Flan"""
    
    def __init__(self, model: nn.Module, tokenizer, name: str, tokens_to_keep: list=[], attention_mask=None, frozen=False):
        super().__init__()
        if name in T5_MODEL_NAMES:
            self.tokens_to_keep = tokens_to_keep
            self.name = name
            self.model = model.cuda()
            if frozen:
                self.model.eval()
                self.train = lambda self, mode: None
            self.linear = nn.Linear(self.model.model_dim, 1).cuda()
            self.tokenizer = tokenizer
        else:
            raise ValueError("Model name not supported")
    
    def forward(self, input_ids, attention_mask=None):
        if self.name in T5_MODEL_NAMES:
            EMPTY_IDS = self.tokenizer(["" for _ in range(len(input_ids))], return_tensors='pt').input_ids
            input_ids, EMPTY_IDS = input_ids.cuda(), EMPTY_IDS.cuda()
            last_hidden_state = self.model(input_ids=input_ids, decoder_input_ids=EMPTY_IDS, output_hidden_states=True).encoder_hidden_states[-1]
            if self.tokens_to_keep:
                out = last_hidden_state[:, self.tokens_to_keep, :].mean(dim=1)
            else:
                out = (last_hidden_state * attention_mask.unsqueeze(2)).sum(1) / attention_mask.sum(1, keepdim=True) # take average over all tokens
            logits = self.linear(out)
            return logits
        else:
            raise ValueError("Model name not supported")
