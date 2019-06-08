"""
Implementation of the SDSN by Rei et al. (2018) to predict lexical entailment.
"""
# Aurelien Coet, 2019.

import torch
import torch.nn as nn


class SDSN(nn.Module):
    """
    Implementation of the SDSN model presented by Rei et al. in 2018
    in the paper: "Scoring Lexical Entailment with a Supervised 
    Directional Similarity Network".
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim=300,
                 gating_dim=300,
                 mapping_dim=300,
                 hidden_dim=100,
                 embeddings=None,
                 max_score=10.0,
                 embedding_dropout=0.5,
                 combination_dropout=0.0,
                 hidden_layer_dropout=0.0):
        """
        Args:
            vocab_size: The size of the vocabulary used by the model for word
                embeddings.
            embedding_dim: The dimension of the word embeddings used by the
                model. Defaults to 300.
            gating_dim: The dimension of the gating layers used by the model.
                Defaults to 300.
            mapping_dim: The dimension of the mapping layers used by the
                model. Defaults to 300.
            hidden_dim: The dimension of the hidden layer used in the model.
                Defaults to 100.
            embeddings: A tensor containing pre-trained word embeddings to
                use in the model. The expected size of the tensor is
                (vocab_size, embedding_dim). If None, word embeddings are
                initialised randomly. Defaults to None.
            max_score: The maximum score of the model's outputs. Defaults
                to 10.0.
            embedding_dropout: The dropout rate to use on the model's word
                embeddings during training. Defaults to 0.5.
            combination_dropout: The dropout rate to use on the combination
                of the words' representations after the mapping layer
                during training. Defaults to 0.0.
            hidden_layer_dropout: The dropout rate to use on the outputs of
                the model's hidden layer during training. Defaults to 0.0.
        """
        super(SDSN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gating_dim = gating_dim
        self.mapping_dim = mapping_dim
        self.hidden_dim = hidden_dim
        self.max_score = max_score
        self.embedding_dropout = embedding_dropout
        self.combination_dropout = combination_dropout
        self.hidden_layer_dropout = hidden_layer_dropout

        self._word_embedding = nn.Sequential(nn.Embedding(self.vocab_size,
                                                          self.embedding_dim,
                                                          _weight=embeddings),
                                             nn.Dropout(p=self.embedding_dropout))

        self._gating_a = nn.Sequential(nn.Linear(self.embedding_dim,
                                                 self.gating_dim),
                                       nn.Sigmoid())

        self._gating_b = nn.Sequential(nn.Linear(self.embedding_dim,
                                                 self.gating_dim),
                                       nn.Sigmoid())
        
        self._mapping_a = nn.Sequential(nn.Linear(self.gating_dim,
                                                  self.mapping_dim),
                                        nn.Tanh())
        
        self._mapping_b = nn.Sequential(nn.Linear(self.gating_dim,
                                                  self.mapping_dim),
                                        nn.Tanh())
        
        self._hidden = nn.Sequential(nn.Dropout(p=self.combination_dropout),
                                     nn.Linear(self.mapping_dim,
                                               self.hidden_dim),
                                     nn.Tanh(),
                                     nn.Dropout(p=self.hidden_layer_dropout))
        
        self._output = nn.Linear(self.hidden_dim, 1)

        self._a = nn.Parameter(torch.tensor([1.0]))
        self._b = nn.Parameter(torch.tensor([0.0]))
    
    def forward(self,
                premise,
                hypothesis):
        """
        Args:
            premise: A batch of candidate hyponyms for which lexical entailment
                scores must be computed.
            hypothesis: A batch of candidate hypernyms for which lexical entailment
                scores must be computed.
        """
        embedded_premise = self._word_embedding(premise)
        embedded_hypothesis = self._word_embedding(hypothesis)

        gate_a = self._gating_a(embedded_premise)
        gate_b = self._gating_b(embedded_hypothesis)

        gated_premise = embedded_premise * gate_b
        gated_hypothesis = embedded_hypothesis * gate_a

        mapped_premise = self._mapping_a(gated_premise)
        mapped_hypothesis = self._mapping_b(gated_hypothesis)

        combination = mapped_premise * mapped_hypothesis

        hidden_output = self._hidden(combination)

        output = self._output(hidden_output)

        return self.max_score * torch.sigmoid(self._a * (output - self._b))
