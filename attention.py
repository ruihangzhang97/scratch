import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.Q = nn.Linear(d_in, d_out)
        self.K = nn.Linear(d_in, d_out)
        self.V = nn.Linear(d_in, d_out)
        
        nn.init.xavier_uniform_(self.Q.weight)
        nn.init.xavier_uniform_(self.K.weight)
        nn.init.xavier_uniform_(self.V.weight)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        q = self.Q(x)  # (batch_size, seq_len, d_out)
        k = self.K(x)  # (batch_size, seq_len, d_out)
        v = self.V(x)  # (batch_size, seq_len, d_out)
        
        scores = torch.bmm(q, k.transpose(1, 2))  # (batch_size, seq_len, d_out) @ (batch_size, d_out, seq_len) --> (batch_size, seq_len, seq_len)

        scores = scores / (self.d_out ** 0.5) # (batch_size, seq_len, seq_len)
        
        attention_weights = F.softmax(scores, dim=-1)  # softmax across seq_len
        
        output = torch.bmm(attention_weights, v)  # (batch_size, seq_len, seq_len) @ (batch_size, seq_len, d_out) --> (batch_size, seq_len, d_out)
        
        return output