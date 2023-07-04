import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        energies = torch.bmm(q, k.transpose(1, 2))
        energies = energies / self.temperature

        if mask is not None:
            energies = energies.masked_fill(mask, -np.inf)
        
        attn = self.softmax(energies)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, energies, attn


class TopK_ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super(TopK_ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        energies = torch.bmm(q, k.transpose(1, 2))
        energies = energies / self.temperature
        save = energies
        
        if mask is not None:
            energies = energies.masked_fill(mask, -np.inf)
            save = self.softmax(save)

        attn = self.softmax(energies)

        output = torch.bmm(attn, v)

 

        return output, save, attn
