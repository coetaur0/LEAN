"""
Definition of the LEAN model.
"""
# Aurelien Coet, 2019.

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, LEAttention
from .utils import get_mask, replace_masked, masked_average, masked_max,\
    masked_min, word2hyp_entailment, lear_entailment


class LEAN1(nn.Module):
    """
    Implementation of the Lexical Entailment Augmented Network (LEAN) for NLI,
    in its first version (LEAN-1).
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 w2h_embeddings=None,
                 lear_embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            w2h_embeddings: A tensor of size (vocab_size, w2h_embedding_dim)
                containing pretrained Word2Hyp embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            lear_embeddings: A tensor of size (vocab_size, lear_embedding_dim)
                containing pretrained LEAR embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(LEAN1, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        self._w2h_embedding = nn.Embedding(self.vocab_size,
                                           200,
                                           padding_idx=padding_idx,
                                           _weight=w2h_embeddings)
        # Freeze the Word2Hyp embeddings.
        self._w2h_embedding.weight.requires_grad = False

        self._lear_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=lear_embeddings)
        # Freeze the LEAR embeddings.
        self._lear_embedding.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size+4,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size+4,
                                                       self.hidden_size+1),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size+1,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_lean_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        w2h_embedded_premises = self._w2h_embedding(premises)
        w2h_embedded_hypotheses = self._w2h_embedding(hypotheses)

        lear_embedded_premises = self._lear_embedding(premises)
        lear_embedded_hypotheses = self._lear_embedding(hypotheses)

        # Computation of lexical entailment scores between words in
        # the premises and hypotheses.
        w2h_scores = word2hyp_entailment(w2h_embedded_premises,
                                         w2h_embedded_hypotheses)
        w2h_scores = w2h_scores[:,
                                :premises_lengths.max(),
                                :hypotheses_lengths.max()]
        w2h_scores /= 100.0

        lear_scores = lear_entailment(lear_embedded_premises,
                                      lear_embedded_hypotheses)
        lear_scores = lear_scores[:,
                                  :premises_lengths.max(),
                                  :hypotheses_lengths.max()]

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)
        
        # Computation of the average, maximum and minimum W2H score for each word
        # in the premise.
        #premise_w2h_avg = masked_average(w2h_scores, hypotheses_mask)
        premise_w2h_max = masked_max(w2h_scores, hypotheses_mask)
        premise_w2h_min = masked_min(w2h_scores, hypotheses_mask)
        w2h_scores = w2h_scores.transpose(2, 1).contiguous()
        # Computation of the average, maximum and minimum W2H score for each word
        # in the hypothesis.
        #hypothesis_w2h_avg = masked_average(w2h_scores, premises_mask)
        hypothesis_w2h_max = masked_max(w2h_scores, premises_mask)
        hypothesis_w2h_min = masked_min(w2h_scores, premises_mask)

        # Computation of the average, maximum and minimum LEAR score for each word
        # in the premise.
        #premise_lear_avg = masked_average(lear_scores, hypotheses_mask)
        premise_lear_max = masked_max(lear_scores, hypotheses_mask)
        premise_lear_min = masked_min(lear_scores, hypotheses_mask)
        lear_scores = lear_scores.transpose(2, 1).contiguous()
        # Computation of the average, maximum and minimum LEAR score for each word
        # in the hypothesis.
        #hypothesis_lear_avg = masked_average(lear_scores, premises_mask)
        hypothesis_lear_max = masked_max(lear_scores, premises_mask)
        hypothesis_lear_min = masked_min(lear_scores, premises_mask)

        # The average, max. and min. LE scores for each metric are concatenated
        # to the vector for each word in the premise.
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises,
                                       premise_w2h_max,
                                       premise_w2h_min,
                                       premise_lear_max,
                                       premise_lear_min],
                                      dim=-1)
        # The same is done for each word in the hypothesis.
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses,
                                         hypothesis_w2h_max,
                                         hypothesis_w2h_min,
                                         hypothesis_lear_max,
                                         hypothesis_lear_min],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        # Computation of the average, max. and min. W2H score between the premise
        # and hypothesis.
        #w2h_avg = masked_average(premise_w2h_avg.transpose(2, 1).contiguous(),
        #                         premises_mask,
        #                         keepdim=False)
        w2h_max = masked_max(premise_w2h_max.transpose(2, 1).contiguous(),
                             premises_mask,
                             keepdim=False)
        w2h_min = masked_min(premise_w2h_min.transpose(2, 1).contiguous(),
                             premises_mask,
                             keepdim=False)
        
        # Computation of the average, max. and min. LEAR score between the premise
        # and hypothesis.
        #lear_avg = masked_average(premise_lear_avg.transpose(2, 1).contiguous(),
        #                          premises_mask,
        #                          keepdim=False)
        lear_max = masked_max(premise_lear_max.transpose(2, 1).contiguous(),
                              premises_mask,
                              keepdim=False)
        lear_min = masked_min(premise_lear_min.transpose(2, 1).contiguous(),
                              premises_mask,
                              keepdim=False)

        # The average, max. and min. LE scores at the sentence pair level are
        # concatenated to the final vector 'v' before classification.
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max,
                       w2h_max, w2h_min,
                       lear_max, lear_min],
                      dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


class LEAN2(nn.Module):
    """
    Implementation of the Lexical Entailment Augmented Network (LEAN) for NLI,
    in its second version (LEAN-2).
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 w2h_embeddings=None,
                 lear_embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            w2h_embeddings: A tensor of size (vocab_size, w2h_embedding_dim)
                containing pretrained Word2Hyp embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            lear_embeddings: A tensor of size (vocab_size, lear_embedding_dim)
                containing pretrained LEAR embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(LEAN2, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        self._w2h_embedding = nn.Embedding(self.vocab_size,
                                           200,
                                           padding_idx=padding_idx,
                                           _weight=w2h_embeddings)
        # Freeze the Word2Hyp embeddings.
        self._w2h_embedding.weight.requires_grad = False

        self._lear_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=lear_embeddings)
        # Freeze the LEAR embeddings.
        self._lear_embedding.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = SoftmaxAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size+6,
                                                       self.hidden_size+3),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size+3,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_lean_weights)

    def forward(self,
                premises,
                premises_lengths,
                le_premises,
                le_premises_lengths,
                hypotheses,
                hypotheses_lengths,
                le_hypotheses,
                le_hypotheses_lengths):
        """
        Args:
            premises: A batch of variable length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            le_premises: A batch of variable length sequences of word indices
                representing premises lowercased and with stopwords removed,
                to compute lexical entailment.
            le_premises_lengths: A 1D tensor containing the lengths of the
                premises in 'le_premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.
            le_hypotheses: A batch of variable length sequences of word indices
                representing hypotheses lowercased and with stopwords removed,
                to compute lexical entailment.
            le_hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'le_hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths)\
            .to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)
        
        le_premises_mask = get_mask(le_premises, le_premises_lengths)\
            .to(self.device)
        le_hypotheses_mask = get_mask(le_hypotheses, le_hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        w2h_embedded_premises = self._w2h_embedding(le_premises)
        w2h_embedded_hypotheses = self._w2h_embedding(le_hypotheses)

        lear_embedded_premises = self._lear_embedding(le_premises)
        lear_embedded_hypotheses = self._lear_embedding(le_hypotheses)

        # Computation of lexical entailment scores between words in
        # the lowercased premises and hypotheses where stopwords
        # have been removed.
        w2h_scores = word2hyp_entailment(w2h_embedded_premises,
                                         w2h_embedded_hypotheses)
        w2h_scores = w2h_scores[:,
                                :le_premises_lengths.max(),
                                :le_hypotheses_lengths.max()]
        w2h_scores /= 100.0

        lear_scores = lear_entailment(lear_embedded_premises,
                                      lear_embedded_hypotheses)
        lear_scores = lear_scores[:,
                                  :le_premises_lengths.max(),
                                  :le_hypotheses_lengths.max()]

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)

        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises],
                                      dim=-1)
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        # Computation of the average, maximum and minimum W2H score.
        w2h_prem_avg = masked_average(w2h_scores, le_hypotheses_mask)
        w2h_prem_max = masked_max(w2h_scores, le_hypotheses_mask)
        w2h_prem_min = masked_min(w2h_scores, le_hypotheses_mask)

        w2h_avg = masked_average(w2h_prem_avg.transpose(2, 1).contiguous(),
                                 le_premises_mask,
                                 keepdim=False)
        w2h_max = masked_max(w2h_prem_max.transpose(2, 1).contiguous(),
                             le_premises_mask,
                             keepdim=False)
        w2h_min = masked_min(w2h_prem_min.transpose(2, 1).contiguous(),
                             le_premises_mask,
                             keepdim=False)
        
        # Computation of the average, max. and min. LEAR scores.
        lear_prem_avg = masked_average(lear_scores, le_hypotheses_mask)
        lear_prem_max = masked_max(lear_scores, le_hypotheses_mask)
        lear_prem_min = masked_min(lear_scores, le_hypotheses_mask)

        lear_avg = masked_average(lear_prem_avg.transpose(2, 1).contiguous(),
                                  le_premises_mask,
                                  keepdim=False)
        lear_max = masked_max(lear_prem_max.transpose(2, 1).contiguous(),
                              le_premises_mask,
                              keepdim=False)
        lear_min = masked_min(lear_prem_min.transpose(2, 1).contiguous(),
                              le_premises_mask,
                              keepdim=False)

        # The average, max. and min. LE scores at the sentence pair level are
        # concatenated to the final vector 'v' before classification.
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max,
                       w2h_avg, w2h_max, w2h_min,
                       lear_avg, lear_max, lear_min],
                      dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


class LEAN3(nn.Module):
    """
    Implementation of the Lexical Entailment Augmented Network (LEAN) for NLI,
    in its third version (LEAN-3).
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 w2h_embeddings=None,
                 lear_embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=3,
                 device="cpu"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            w2h_embeddings: A tensor of size (vocab_size, w2h_embedding_dim)
                containing pretrained Word2Hyp embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            lear_embeddings: A tensor of size (vocab_size, lear_embedding_dim)
                containing pretrained LEAR embeddings to use in LEAN to
                compute lexical entailment between words. If None, word
                embeddings are initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(LEAN3, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)

        self._w2h_embedding = nn.Embedding(self.vocab_size,
                                           200,
                                           padding_idx=padding_idx,
                                           _weight=w2h_embeddings)
        # Freeze the Word2Hyp embeddings.
        self._w2h_embedding.weight.requires_grad = False

        self._lear_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=lear_embeddings)
        # Freeze the LEAR embeddings.
        self._lear_embedding.weight.requires_grad = False

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        self._attention = LEAttention()

        self._projection = nn.Sequential(nn.Linear(4*2*self.hidden_size+6,
                                                   self.hidden_size),
                                         nn.ReLU())

        self._composition = Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size+6,
                                                       self.hidden_size+3),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size+3,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_lean_weights)

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths):
        """
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        premises_mask = get_mask(premises, premises_lengths).to(self.device)
        hypotheses_mask = get_mask(hypotheses, hypotheses_lengths)\
            .to(self.device)

        embedded_premises = self._word_embedding(premises)
        embedded_hypotheses = self._word_embedding(hypotheses)

        w2h_embedded_premises = self._w2h_embedding(premises)
        w2h_embedded_hypotheses = self._w2h_embedding(hypotheses)

        lear_embedded_premises = self._lear_embedding(premises)
        lear_embedded_hypotheses = self._lear_embedding(hypotheses)

        # Computation of lexical entailment scores between words in
        # the premises and hypotheses.
        w2h_scores = word2hyp_entailment(w2h_embedded_premises,
                                         w2h_embedded_hypotheses)
        w2h_scores = w2h_scores[:,
                                :premises_lengths.max(),
                                :hypotheses_lengths.max()]
        w2h_scores /= 100.0

        lear_scores = lear_entailment(lear_embedded_premises,
                                      lear_embedded_hypotheses)
        lear_scores = lear_scores[:,
                                  :premises_lengths.max(),
                                  :hypotheses_lengths.max()]
        lear_scores = 1 - lear_scores

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        encoded_premises = self._encoding(embedded_premises,
                                          premises_lengths)
        encoded_hypotheses = self._encoding(embedded_hypotheses,
                                            hypotheses_lengths)

        attended_premises, attended_hypotheses =\
            self._attention(w2h_scores, lear_scores,
                            encoded_premises, premises_mask,
                            encoded_hypotheses, hypotheses_mask)
        
        # Computation of the average, maximum and minimum W2H score for each word
        # in the premise.
        premise_w2h_avg = masked_average(w2h_scores, hypotheses_mask)
        premise_w2h_max = masked_max(w2h_scores, hypotheses_mask)
        premise_w2h_min = masked_min(w2h_scores, hypotheses_mask)
        w2h_scores = w2h_scores.transpose(2, 1).contiguous()
        # Computation of the average, maximum and minimum W2H score for each word
        # in the hypothesis.
        hypothesis_w2h_avg = masked_average(w2h_scores, premises_mask)
        hypothesis_w2h_max = masked_max(w2h_scores, premises_mask)
        hypothesis_w2h_min = masked_min(w2h_scores, premises_mask)

        # Computation of the average, maximum and minimum LEAR score for each word
        # in the premise.
        premise_lear_avg = masked_average(lear_scores, hypotheses_mask)
        premise_lear_max = masked_max(lear_scores, hypotheses_mask)
        premise_lear_min = masked_min(lear_scores, hypotheses_mask)
        lear_scores = lear_scores.transpose(2, 1).contiguous()
        # Computation of the average, maximum and minimum LEAR score for each word
        # in the hypothesis.
        hypothesis_lear_avg = masked_average(lear_scores, premises_mask)
        hypothesis_lear_max = masked_max(lear_scores, premises_mask)
        hypothesis_lear_min = masked_min(lear_scores, premises_mask)

        # The average, max. and min. LE scores for each metric are concatenated
        # to the vector for each word in the premise.
        enhanced_premises = torch.cat([encoded_premises,
                                       attended_premises,
                                       encoded_premises - attended_premises,
                                       encoded_premises * attended_premises,
                                       premise_w2h_avg,
                                       premise_w2h_max,
                                       premise_w2h_min,
                                       premise_lear_avg,
                                       premise_lear_max,
                                       premise_lear_min],
                                      dim=-1)
        # The same is done for each word in the hypothesis.
        enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                         attended_hypotheses,
                                         encoded_hypotheses -
                                         attended_hypotheses,
                                         encoded_hypotheses *
                                         attended_hypotheses,
                                         hypothesis_w2h_avg,
                                         hypothesis_w2h_max,
                                         hypothesis_w2h_min,
                                         hypothesis_lear_avg,
                                         hypothesis_lear_max,
                                         hypothesis_lear_min],
                                        dim=-1)

        projected_premises = self._projection(enhanced_premises)
        projected_hypotheses = self._projection(enhanced_hypotheses)

        if self.dropout:
            projected_premises = self._rnn_dropout(projected_premises)
            projected_hypotheses = self._rnn_dropout(projected_hypotheses)

        v_ai = self._composition(projected_premises, premises_lengths)
        v_bj = self._composition(projected_hypotheses, hypotheses_lengths)

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                                                .transpose(2, 1), dim=1)\
            / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                                                  .transpose(2, 1), dim=1)\
            / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        # Computation of the average, max. and min. W2H score between the premise
        # and hypothesis.
        w2h_avg = masked_average(premise_w2h_avg.transpose(2, 1).contiguous(),
                                 premises_mask,
                                 keepdim=False)
        w2h_max = masked_max(premise_w2h_max.transpose(2, 1).contiguous(),
                             premises_mask,
                             keepdim=False)
        w2h_min = masked_min(premise_w2h_min.transpose(2, 1).contiguous(),
                             premises_mask,
                             keepdim=False)
        
        # Computation of the average, max. and min. LEAR score between the premise
        # and hypothesis.
        lear_avg = masked_average(premise_lear_avg.transpose(2, 1).contiguous(),
                                  premises_mask,
                                  keepdim=False)
        lear_max = masked_max(premise_lear_max.transpose(2, 1).contiguous(),
                              premises_mask,
                              keepdim=False)
        lear_min = masked_min(premise_lear_min.transpose(2, 1).contiguous(),
                              premises_mask,
                              keepdim=False)

        # The average, max. and min. LE scores at the sentence pair level are
        # concatenated to the final vector 'v' before classification.
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max,
                       w2h_avg, w2h_max, w2h_min,
                       lear_avg, lear_max, lear_min],
                      dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities


def _init_lean_weights(module):
    """
    Initialise the weights of the LEAN model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
