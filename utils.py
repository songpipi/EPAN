from collections import defaultdict
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from tqdm import tqdm

from torchvision import transforms
from loader.transform import Lowercase, PadFirst, PadLast, PadToLength, RemovePunctuation, \
                             SplitWithWhiteSpace, ToIndex, ToTensor, TrimExceptAscii, Truncate

from loader.MSVD import MSVD
from models.decoder import Decoder
from models.emotion_captioning_network import EmotionCaptioningNetwork as ECN
from models.transformer.Models import Encoder as PhraseEncoder
from models.visual_encoder import VisualEncoder
from models.transformer.Models import Transformer_emo

from EmotionEval.getResult import get_result
from EmotionEval.pycocoevalcap.bleu.bleu import Bleu
from EmotionEval.pycocoevalcap.rouge.rouge import Rouge
from EmotionEval.pycocoevalcap.cider.cider import Cider
from EmotionEval.pycocoevalcap.meteor.meteor import Meteor


class LossChecker:
    def __init__(self, num_losses):
        self.num_losses = num_losses

        self.losses = [ [] for _ in range(self.num_losses) ]

    def update(self, *loss_vals):
        assert len(loss_vals) == self.num_losses

        for i, loss_val in enumerate(loss_vals):
            self.losses[i].append(loss_val)

    def mean(self, last=0):
        mean_losses = [ 0. for _ in range(self.num_losses) ]
        for i, loss in enumerate(self.losses):
            _loss = loss[-last:]
            mean_losses[i] = sum(_loss) / len(_loss)
        return mean_losses


def build_loaders(config):
    corpus = MSVD(config)
    print('#vocabs: {} ({}), #words: {} ({}). Trim words which appear less than {} times.'.format(
        corpus.vocab.n_vocabs, corpus.vocab.n_vocabs_untrimmed, corpus.vocab.n_words,
        corpus.vocab.n_words_untrimmed, config.loader.min_count))
    return corpus.train_data_loader, corpus.val_data_loader, corpus.test_data_loader, corpus.em_train_data_loader, corpus.em_test_data_loader, corpus.vocab


def build_model(config, vocab):
    visual_encoder = VisualEncoder(
        app_feat=config.vis_encoder.app_feat,
        mot_feat=config.vis_encoder.mot_feat,
        app_input_size=config.vis_encoder.app_feat_size,
        mot_input_size=config.vis_encoder.mot_feat_size,
        app_output_size=config.vocab.embedding_size,
        mot_output_size=config.vocab.embedding_size)
    
    phrase_encoder = PhraseEncoder(
        len_max_seq=config.loader.max_caption_len + 2,
        d_word_vec=config.vocab.embedding_size,
        n_layers=config.phr_encoder.SA_num_layers,
        n_head=config.phr_encoder.SA_num_heads,
        d_k=config.phr_encoder.SA_dim_k,
        d_v=config.phr_encoder.SA_dim_v,
        d_model=config.vocab.embedding_size,
        d_inner=config.phr_encoder.SA_dim_inner,
        dropout=config.phr_encoder.SA_dropout)

    decoder = Decoder(
        num_layers=config.decoder.rnn_num_layers,
        emo_word_size=config.vocab.embedding_size,
        vis_feat_size=config.vocab.embedding_size,
        feat_len=config.loader.frame_sample_len,
        embedding_size=config.vocab.embedding_size,
        sem_align_hidden_size=config.decoder.sem_align_hidden_size,
        sem_attn_hidden_size=config.decoder.sem_attn_hidden_size,
        hidden_size=config.decoder.rnn_hidden_size,
        output_size=vocab.n_vocabs)

    emotion_attention = Transformer_emo(
        n_layers=config.emo_transformer.num_layers,
        n_head=config.emo_transformer.num_heads,
        d_k=config.emo_transformer.dim_k,
        d_v=config.emo_transformer.dim_v,
        d_model=config.vocab.embedding_size,
        d_inner=config.emo_transformer.dim_inner,
        d_outer = config.vocab.embedding_size,
        emo_num = config.emo_num)

    model = ECN(visual_encoder, phrase_encoder, decoder, emotion_attention, config.loader.max_caption_len, vocab,
                config.PS_threshold)

    return model


def parse_batch(batch):
    pos, neg = batch
    pos_vids, pos_vis_feats, pos_captions, pos_em_word, pos_em = pos
    neg_vids, neg_vis_feats, neg_captions = neg

    for model in pos_vis_feats:
        pos_vis_feats[model] = pos_vis_feats[model].cuda()
        neg_vis_feats[model] = neg_vis_feats[model].cuda()
    pos_captions = pos_captions.long().cuda()
    pos_em_word = pos_em_word.long().cuda()
    pos_em = pos_em.long().cuda()
    neg_captions = neg_captions.long().cuda()

    pos = ( pos_vids, pos_vis_feats, pos_captions, pos_em_word, pos_em )
    neg = ( neg_vids, neg_vis_feats, neg_captions )
    return pos, neg

        
def transform_caption(vocab,pos_captions):
    transform_sentence = transforms.Compose([
        Lowercase(),
        RemovePunctuation(),
        SplitWithWhiteSpace(),
        Truncate(15), ])
    transform_cap = transforms.Compose([
        transform_sentence,
        ToIndex(vocab.word2idx),
        PadFirst(vocab.word2idx['<SOS>']),
        PadLast(vocab.word2idx['<EOS>']),
        PadToLength(vocab.word2idx['<PAD>'], vocab.max_sentence_len + 2), # +2 for <SOS> and <EOS>
        ToTensor(torch.long), ])   
    return  transform_cap(pos_captions)
 

def train(e, model, optimizer, train_iter, vocab, teacher_forcing_ratio, C, cross_wei):

    model.train()

    loss_checker = LossChecker(4)
    PAD_idx = vocab.word2idx['<PAD>']
    pgbar = tqdm(train_iter)
 
    
    batch_id = 0
 
    for batch in pgbar:
        ( pos_vids, pos_vis_feats, pos_captions, pos_em_word, pos_em ), ( neg_vids, neg_vis_feats, neg_captions ) = parse_batch(batch)

        optimizer.zero_grad()
        output, contrastive_attention, cls_attention1, cls_attention2 = model(
            pos_vis_feats, pos_captions, pos_em_word, neg_vis_feats, neg_captions, batch_id, teacher_forcing_ratio)
 
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        pos_captions[1:].contiguous().view(-1),
                                        weight = cross_wei,
                                        ignore_index=PAD_idx)
        contrastive_attention_loss = F.binary_cross_entropy_with_logits(
            contrastive_attention.mean(dim=0), torch.cuda.FloatTensor([ 1, 0 ]))

 
        cls_loss1 = F.nll_loss(cls_attention1.permute(2,0,1).reshape(-1, 35),
                                pos_em.transpose(0,1).contiguous().view(-1),
                                ignore_index=PAD_idx)              
        cls_loss2 = F.nll_loss(cls_attention2.permute(2,0,1).reshape(-1, C.emo_num+1),
                                pos_em_word.transpose(0,1).contiguous().view(-1),
                                ignore_index=PAD_idx)  
        cls_loss = cls_loss1 + cls_loss2

        loss = cross_entropy_loss + C.CA_lambda * contrastive_attention_loss + C.CA_beta * cls_loss 
    
        loss.backward()
        if C.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.gradient_clip)
        optimizer.step()

        loss_checker.update(loss.item(), cross_entropy_loss.item(), contrastive_attention_loss.item(), cls_loss.item())
        pgbar.set_description("[Epoch #{0}] loss: {3:.3f} = CE {4:.3f} + CTR {1} * {5:.3f} + CLA {2} * {6:.3f}".format(
            e, C.CA_lambda, C.CA_beta, *loss_checker.mean(last=10)))

    total_loss, cross_entropy_loss, contrastive_attention_loss, cls_loss = loss_checker.mean()
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'contrastive_attention': contrastive_attention_loss,
        'emotion_cls': cls_loss
    }
    return loss


def evaluate(model, val_iter, vocab, C, cross_wei):
    model.eval()

    loss_checker = LossChecker(4)
    PAD_idx = vocab.word2idx['<PAD>']
    batch_id=0
    for batch in val_iter:
        ( _, pos_vis_feats, pos_captions, pos_em_word, pos_em ), ( _, neg_vis_feats, neg_captions ) = parse_batch(batch)
        
        output, contrastive_attention, cls_attention1, cls_attention2 = model(
            pos_vis_feats, pos_captions, pos_em_word, neg_vis_feats, neg_captions, batch_id, teacher_forcing_ratio=0.)
        cross_entropy_loss = F.nll_loss(output[1:].view(-1, vocab.n_vocabs),
                                        pos_captions[1:].contiguous().view(-1),
                                        weight = cross_wei,
                                        ignore_index=PAD_idx)
        contrastive_attention_loss = F.binary_cross_entropy_with_logits(
            contrastive_attention.mean(dim=0), torch.cuda.FloatTensor([ 1, 0 ]))

        cls_loss1 = F.nll_loss(cls_attention1.permute(2,0,1).reshape(-1, 35),
                                pos_em.transpose(0,1).contiguous().view(-1),
                                ignore_index=PAD_idx)              
        cls_loss2 = F.nll_loss(cls_attention2.permute(2,0,1).reshape(-1, C.emo_num+1),
                                pos_em_word.transpose(0,1).contiguous().view(-1),
                                ignore_index=PAD_idx)  
        cls_loss = cls_loss1 + cls_loss2
        loss = cross_entropy_loss + C.CA_lambda * contrastive_attention_loss + C.CA_beta * cls_loss

        loss_checker.update(loss.item(), cross_entropy_loss.item(), contrastive_attention_loss.item(), cls_loss.item())
 
    total_loss, cross_entropy_loss, contrastive_attention_loss, cls_loss = loss_checker.mean()
    
    loss = {
        'total': total_loss,
        'cross_entropy': cross_entropy_loss,
        'contrastive_attention': contrastive_attention_loss,
        'emotion_cls': cls_loss
    }
    return loss




def build_YOLO_iter(data_iter, batch_size):
    score_dataset = {}
    for batch in iter(data_iter):
        ( vids, feats, captions, em,_ ), _ = parse_batch(batch)
        for i, vid in enumerate(vids):
            feat = {}
            for model in feats:
                feat[model] = feats[model][i]
            if vid not in score_dataset:
                score_dataset[vid] = feat

    score_iter = []
    vids = list(score_dataset.keys())
    feats = list(score_dataset.values())
    while len(vids) > 0:
        vids_batch = vids[:batch_size]
        feats_batch = defaultdict(lambda: [])
        for feat in feats[:batch_size]:
            for model, f in feat.items():
                feats_batch[model].append(f)
        for model in feats_batch:
            feats_batch[model] = torch.stack(feats_batch[model], dim=0)
        yield ( vids_batch, feats_batch )
        vids = vids[batch_size:]
        feats = feats[batch_size:]



def idxs_to_emo(idxs,em_idx2word,idx2em,type):
    for idx in idxs:
        idx = idx.item()
        if type == 1:            
            word = idx2em[idx]
        else:
            word = em_idx2word[idx]
    return word


def score_full( model, data_iter, vocab):
    def build_refs(data_iter):
        vid2idx = {}
        refs = {}
        for idx, (vid, captions) in enumerate(data_iter.captions.items()):
            vid2idx[vid] = idx
            refs[idx] = captions
        return refs, vid2idx

    model.eval()

    PAD_idx = vocab.word2idx['<PAD>']
    YOLO_iter = build_YOLO_iter(data_iter, batch_size=32)
    refs, vid2idx = build_refs(data_iter)

    hypos = {}
    for vids, feats in tqdm(YOLO_iter, desc='score'):
        captions,_ = model.describe(feats)
        captions = [ idxs_to_sentence(caption, vocab.idx2word, vocab.word2idx['<EOS>']) for caption in captions ]
        for vid, caption in zip(vids, captions):
            hypos[vid2idx[vid]] = [ caption ]
    scores = calc_scores(refs, hypos)
    
    emotion_path = 'EmotionEval/sentiment-words-E/'
    Acc_sw, Acc_c = get_result(hypos, emotion_path, refs)
    scores['Acc_sw'] = Acc_sw
    scores['Acc_c'] = Acc_c
    add_2 = 0.1 * (Acc_sw + Acc_c)
    add_1 = 0.1 * scores['Bleu_1'] + 0.2 * scores['Bleu_2'] + 0.3 * scores['Bleu_3'] + 0.4 * scores['Bleu_4']
    scores['BFS'] = 0.8 * add_1 + add_2
    scores['CFS'] = 0.8 * scores['CIDEr'] + add_2

    return scores, refs, hypos, vid2idx



# refers: https://github.com/zhegan27/SCN_for_video_captioning/blob/master/SCN_evaluation.py
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


# refers: https://stackoverflow.com/questions/52660985/pytorch-how-to-get-learning-rate-during-training
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def idxs_to_sentence(idxs, idx2word, EOS_idx):
    words = []
    for idx in idxs[1:]:
        idx = idx.item()
        if idx == EOS_idx:
            break
        word = idx2word[idx]
        words.append(word)
    sentence = ' '.join(words)
    return sentence


def save_checkpoint(ckpt_fpath, epoch, model, optimizer):
    ckpt_dpath = os.path.dirname(ckpt_fpath)
    if not os.path.exists(ckpt_dpath):
        os.makedirs(ckpt_dpath)

    torch.save(model.state_dict(), ckpt_fpath)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_result(vid2pred, vid2GTs, save_fpath, test_scores, vid2idx):
    assert set(vid2pred.keys()) == set(vid2GTs.keys())

    idx2vid={}
    for vid,idx in vid2idx.items():
        idx2vid[idx] = vid

    save_dpath = os.path.dirname(save_fpath)
    if not os.path.exists(save_dpath):
        os.makedirs(save_dpath)

    vids = vid2pred.keys()
    with open(save_fpath, 'w') as fout:

        fout.write("vid,cap\n")
        for vid in vids:
            pred = vid2pred[vid]
            idx = idx2vid[vid]
            line = ', '.join([ str(idx), pred[0]])
            fout.write("{}\n".format(line))