''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import ipdb
import models.transformer.Constants as Constants
from models.transformer.Layers import EncoderLayer, DecoderLayer, DecoderLayer_nomul
from models.transformer.SubLayers import MHA_crossmodal
from models.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward




def get_pad_mask(seq):
    return (seq != Constants.PAD).unsqueeze(-2)

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return float(position) / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Encoder, self).__init__()

        n_position = len_max_seq + 1

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq_emb, src_pos, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_pos, seq_q=src_pos)
        non_pad_mask = get_non_pad_mask(src_pos)

        # -- Forward
        enc_output = src_seq_emb + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output

class Encoder_emo(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Encoder_emo, self).__init__()

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq_emb, return_attns=False):

        enc_slf_attn_list = []


        # -- Forward
        enc_output = src_seq_emb 

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,

class Decoder_emo(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Decoder_emo, self).__init__()



        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq_emb, enc_output,mask=None, return_attns=True):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = trg_seq_emb

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn,energies = dec_layer(dec_output, enc_output,dec_enc_attn_mask=mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn] if return_attns else []
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list,energies
        return dec_output,energies

class Decoder_emo_nomul(nn.Module):
    ''' A decoder model without self attention mechanism. '''

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super(Decoder_emo_nomul, self).__init__()


        self.layer_stack = nn.ModuleList([
            DecoderLayer_nomul(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, trg_seq_emb, enc_output,mask=None, return_attns=True):

        dec_slf_attn_list, dec_enc_attn_list, dec_eng_list = [], [], []


        # -- Forward
        dec_output = trg_seq_emb 

        for dec_layer in self.layer_stack:
            dec_output, dec_enc_attn,energies = dec_layer(dec_output, enc_output, dec_enc_attn_mask=mask)
            if return_attns:
                dec_enc_attn_list += [dec_enc_attn] if return_attns else []
                dec_eng_list += [energies]

        if return_attns:
            return dec_output, dec_enc_attn_list, dec_eng_list
        return dec_output,energies


class Transformer_emo(nn.Module):

    def __init__(
            self, n_layers, n_head, d_k, d_v,
            d_model, d_inner,d_outer, emo_num, dropout=0.1):

        super(Transformer_emo, self).__init__()

        self.emo_dec = Decoder_emo_nomul(n_layers=n_layers,n_head=n_head,d_k=d_k,d_v=d_v,d_model=d_model,d_inner=d_model)

        self.fc = nn.Linear(d_model, d_inner)        
        self.classifier = nn.Linear(d_inner, 34)

        self.emo_dec2 = Decoder_emo_nomul(n_layers=n_layers,n_head=n_head,d_k=d_k,d_v=d_v,d_model=d_model,d_inner=d_model)

        self.fc2 = nn.Linear(d_model, d_outer)        
        self.classifier2 = nn.Linear(d_outer, emo_num)
        self.emo_num = emo_num

    def padding_seq(self):
        class_len = [2, 5, 6, 5, 7, 7, 4, 6, 7, 5, 6, 2, 6, 3, 5, 4, 5, 7, 3, 11, 5, 2, 4, 5, 3, 4, 6, 5, 5, 6, 7, 9, 5, 7] # 179 
        start_id = [0, 2, 7, 13, 18, 25, 32, 36, 42, 49, 54, 60, 62, 68, 71, 76, 80, 85, 92, 95, 106, 111, 113, 117, 122, 125, 129, 135, 140, 145, 151, 158, 167, 172]
        end_id = [2, 7, 13, 18, 25, 32, 36, 42, 49, 54, 60, 62, 68, 71, 76, 80, 85, 92, 95, 106, 111, 113, 117, 122, 125, 129, 135, 140, 145, 151, 158, 167, 172, 179]
        index = np.zeros([34,179])
        for i in range(34):
            index[i,start_id[i]:end_id[i]] = 1
        return index

    def forward(self, src_seq, trg_seq, src_seq_34):

        batch_size = trg_seq.size(0)

        enc_vt, attn, energies= self.emo_dec(trg_seq, src_seq_34)
        enc_vt = self.fc(enc_vt)
        attn_feats1 = enc_vt.mean(1)
        emo_dist1 = self.classifier(attn_feats1) 

        ####### topK ######
        top_mask = self.padding_seq()
        top_mask = torch.tensor(top_mask, dtype=torch.long).cuda()
        top_mask = torch.eq(top_mask, 0)
        top_mask = top_mask[None,:,:].repeat(batch_size,1,1)
        emo_threshold =1
        _, index = emo_dist1.topk(emo_threshold, dim=-1) 

        dummy = index.unsqueeze(2).expand(index.size(0), index.size(1), top_mask.size(2))
        src_mask = torch.gather(top_mask, 1, dummy).repeat(1,30,1)


        enc_vt, attn, energies = self.emo_dec2(trg_seq, src_seq, mask=src_mask)
        enc_vt = self.fc2(enc_vt)
        attn_feats2 = enc_vt.mean(1)
        emo_dist2 = self.classifier2(attn_feats2) 

        return enc_vt, emo_dist1, emo_dist2
