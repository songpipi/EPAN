import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from models.transformer.Constants import PAD as SelfAttention_PAD

import os


class EmotionCaptioningNetwork(nn.Module):
    def __init__(self, vis_encoder, phr_encoder, decoder, emotion_attention, max_caption_len, vocab, PS_threshold):
        super(EmotionCaptioningNetwork, self).__init__()
        self.vis_encoder = vis_encoder
        self.phr_encoder = phr_encoder
        self.decoder = decoder
        self.emotion_attention = emotion_attention
        self.max_caption_len = max_caption_len
        self.vocab = vocab
        self.PS_threshold = PS_threshold

        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.embedding_weights), freeze=False,
                                                      padding_idx=self.vocab.word2idx['<PAD>'])

        self.embedding_emo = nn.Embedding.from_pretrained(torch.FloatTensor(self.vocab.embedding_weights_em), freeze=False)

    def get_rnn_init_hidden(self, batch_size, num_layers, hidden_size):
        return (
            torch.zeros(num_layers, batch_size, hidden_size).cuda(),
            torch.zeros(num_layers, batch_size, hidden_size).cuda())


    def get_rnn_init_hidden_em(self, batch_size, num_layers, emotion_word):
        hidden = emotion_word.view(num_layers, batch_size, -1)
        return (hidden,hidden)

    def forward_visual_encoder(self, vis_feats):
        app_feats = vis_feats[self.vis_encoder.app_feat]
        vis_feats = self.vis_encoder(app_feats)
        return vis_feats


    def forward_decoder(self, batch_size, vocab_size, pos_vis_feats, pos_captions, pos_em_word, neg_vis_feats, neg_captions,
                        teacher_forcing_ratio, batch_id):
        
        cls_attention1,cls_attention2 = None,None

        emotion_vocab_feats = []
        for idx,word in enumerate(self.vocab.em_vocabs):
            word = Variable(torch.cuda.LongTensor(1).fill_(self.vocab.em_word2idx[word])) 
            weights = self.embedding_emo(word)
            emotion_vocab_feats.append(weights)
        emotion_vocabs = torch.stack(emotion_vocab_feats).transpose(0,1)
        emotion_vocab_feats = emotion_vocabs.repeat(batch_size,1,1) 

        emotion_cate_feats = []
        for idx,word in enumerate(self.vocab.em_cates):
            word = Variable(torch.cuda.LongTensor(1).fill_(self.vocab.em_word2idx[word])) 
            weights = self.embedding_emo(word)
            emotion_cate_feats.append(weights)
        emotion_cates = torch.stack(emotion_cate_feats).transpose(0,1)
        emotion_cate_feats = emotion_cates.repeat(batch_size,1,1) 

        em_embeding, em_logits1, em_logits2 = self.emotion_attention(src_seq=emotion_vocab_feats, trg_seq=pos_vis_feats,src_seq_34=emotion_cate_feats)


        zero_clo = Variable(torch.cuda.FloatTensor(batch_size,1).fill_(0))
        em_logits1 = torch.log_softmax(em_logits1, dim=1)
        em_logits2 = torch.log_softmax(em_logits2, dim=1) 
        cls_attention1 = torch.cat((zero_clo,em_logits1),1).view(batch_size,-1,1).repeat(1,1,3)
        cls_attention2 = torch.cat((zero_clo,em_logits2),1).view(batch_size,-1,1).repeat(1,1,3)
        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.hidden_size)

        
        
        caption_EOS_table = pos_captions == self.vocab.word2idx['<EOS>']
        caption_PAD_table = pos_captions == self.vocab.word2idx['<PAD>']
        caption_end_table = ~(~caption_EOS_table * ~caption_PAD_table)

        outputs = Variable(torch.zeros(self.max_caption_len + 2, batch_size, vocab_size)).cuda()

        caption_lens = torch.zeros(batch_size).cuda().long()
        contrastive_attention_list = []
        output = Variable(torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']))


        for t in range(1, self.max_caption_len + 2):
            embedded = self.embedding(output.view(1, -1)).squeeze(0)
            if t == 1:
                embedded_list = embedded[:, None, :] 
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            elif t == 2:
                embedded_list = embedded[:, None, :]
                caption_lens += 1
                src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
            else:
                embedded_list = torch.cat([ embedded_list, embedded[:, None, :]  ], dim=1)
                caption_lens += ((output.long().squeeze() != self.vocab.word2idx['<PAD>']) * \
                                 (output.long().squeeze() != self.vocab.word2idx['<EOS>'])).long()
                src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                src_pos[src_pos > caption_lens[:, None]] = SelfAttention_PAD
            phr_feats, _ = self.phr_encoder(embedded_list, src_pos, return_attns=True) 

            output, hidden,  sem_align_weights, sem_align_logits = self.decoder(em_embeding, embedded, \
                                                hidden, pos_vis_feats, phr_feats, t)

            if t >= 2:

                pos_sem_align_logits = sem_align_logits 
                _, neg_sem_align_weights, neg_sem_align_logits = self.decoder.semantic_alignment(neg_vis_feats,phr_feats)
                pos_align_logit = pos_sem_align_logits.sum(dim=2) 
                neg_align_logit = neg_sem_align_logits.sum(dim=2)            
                pos_align_logit = pos_align_logit[~caption_end_table[t-1]] 
                neg_align_logit = neg_align_logit[~caption_end_table[t-1]]
                align_logits = torch.stack([ pos_align_logit, neg_align_logit ], dim=2)
                align_logits = align_logits.view(-1, 2) 
                contrastive_attention_list.append(align_logits)


            # Early stop
            if torch.all(caption_end_table[t]).item():
                break

            # Choose the next word
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1 = output.data.max(1)[1]
            output = Variable(pos_captions.data[t] if is_teacher else top1).cuda()


        contrastive_attention = torch.cat(contrastive_attention_list, dim=0)
        return outputs, contrastive_attention, cls_attention1, cls_attention2



    def forward(self, pos_vis_feats, pos_captions, pos_em_word, neg_vis_feats, neg_captions,batch_id, teacher_forcing_ratio=0.):

        batch_size = pos_captions.shape[1] 
        vocab_size = self.decoder.output_size

        pos_vis_feats = self.forward_visual_encoder(pos_vis_feats)
        neg_vis_feats = self.forward_visual_encoder(neg_vis_feats)
        
        captions, CA_logits, cls_attention1, cls_attention2 = self.forward_decoder(batch_size, vocab_size, \
            pos_vis_feats, pos_captions, pos_em_word,neg_vis_feats, neg_captions, teacher_forcing_ratio, batch_id)
        return captions, CA_logits, cls_attention1, cls_attention2

    def describe(self, vis_feats):
        batch_size = vis_feats['clip'].size(0)
        vocab_size = self.decoder.output_size

        vis_feats = self.forward_visual_encoder(vis_feats)
        captions,words = self.beam_search(batch_size, vocab_size, vis_feats)
        return captions,words


    def beam_search(self, batch_size, vocab_size, vis_feats, width=5):
        emotion_vocab_feats = []
        for idx,word in enumerate(self.vocab.em_vocabs):
            word = Variable(torch.cuda.LongTensor(1).fill_(self.vocab.em_word2idx[word]))
            weights = self.embedding_emo(word)
            emotion_vocab_feats.append(weights)
        emotion_vocabs = torch.stack(emotion_vocab_feats).transpose(0,1)
        emotion_vocab_feats = emotion_vocabs.repeat(batch_size,1,1)

        emotion_cate_feats = []
        for idx,word in enumerate(self.vocab.em_cates):
            word = Variable(torch.cuda.LongTensor(1).fill_(self.vocab.em_word2idx[word])) 
            weights = self.embedding_emo(word)
            emotion_cate_feats.append(weights)
        emotion_cates = torch.stack(emotion_cate_feats).transpose(0,1)
        emotion_cate_feats = emotion_cates.repeat(batch_size,1,1) 

        em_embeding, emo_dist1, emo_dist2 = self.emotion_attention(src_seq=emotion_vocab_feats, \
            trg_seq=vis_feats,src_seq_34=emotion_cate_feats)


        words = None
        hidden = self.get_rnn_init_hidden(batch_size, self.decoder.num_layers, self.decoder.hidden_size)

        input_list = [ torch.cuda.LongTensor(1, batch_size).fill_(self.vocab.word2idx['<SOS>']) ]
        hidden_list = [ hidden ]
        cum_prob_list = [ torch.ones(batch_size).cuda() ]
        cum_prob_list = [ torch.log(cum_prob) for cum_prob in cum_prob_list ]
        EOS_idx = self.vocab.word2idx['<EOS>']

        output_list = [ [[]] for _ in range(batch_size) ]
        for t in range(1, self.max_caption_len + 2):
            beam_output_list = []
            normalized_beam_output_list = []
            beam_hidden_list = ( [], [] )
            next_output_list = [ [] for _ in range(batch_size) ]

            assert len(input_list) == len(hidden_list) == len(cum_prob_list)
            for i, (input, hidden, cum_prob) in enumerate(zip(input_list, hidden_list, cum_prob_list)):
                caption_list = [ output_list[b][i] for b in range(batch_size) ]
                if t == 1:
                    words_list = input.transpose(0, 1)
                else:
                    words_list = torch.cuda.LongTensor(caption_list)

                embedded_list = self.embedding(words_list)
                if t == 1:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                elif t == 2:
                    caption_lens = torch.cuda.LongTensor(batch_size).fill_(1)
                    src_pos = torch.arange(1, 2).repeat(batch_size, 1).cuda()
                else:
                    caption_lens = torch.cuda.LongTensor([ [ idx.item() for idx in caption ].index(EOS_idx) if EOS_idx in [ idx.item() for idx in caption ] else t-1 for caption in caption_list ])
                    src_pos = torch.arange(1, t).repeat(batch_size, 1).cuda()
                    src_pos[src_pos > caption_lens[:, None]] = 0
                phr_feats, _ = self.phr_encoder(embedded_list, src_pos, return_attns=True)


                embedded = self.embedding(input.view(1, -1)).squeeze(0)
                output, next_hidden, _, _ = self.decoder(em_embeding, embedded, hidden, vis_feats, phr_feats, t)

                EOS_mask = [ 1 if EOS_idx in [ idx.item() for idx in caption ] else 0 for caption in caption_list ]
                EOS_mask = torch.cuda.BoolTensor(EOS_mask)
                output[EOS_mask] = 0.

                output += cum_prob.unsqueeze(1)
                beam_output_list.append(output)

                caption_lens = [ [ idx.item() for idx in caption ].index(EOS_idx) + 1 if EOS_idx in [ idx.item() for idx in caption ] else t for caption in caption_list ]
                caption_lens = torch.cuda.FloatTensor(caption_lens)
                normalizing_factor = ((5 + caption_lens) ** 1.6) / ((5 + 1) ** 1.6)
                normalized_output = output / normalizing_factor[:, None]
                normalized_beam_output_list.append(normalized_output)
                beam_hidden_list[0].append(next_hidden[0])
                beam_hidden_list[1].append(next_hidden[1])
            beam_output_list = torch.cat(beam_output_list, dim=1) # ( 100, n_vocabs * width )
            normalized_beam_output_list = torch.cat(normalized_beam_output_list, dim=1)
            beam_topk_output_index_list = normalized_beam_output_list.argsort(dim=1, descending=True)[:, :width] # ( 100, width )
            topk_beam_index = beam_topk_output_index_list // vocab_size # ( 100, width )
            topk_output_index = beam_topk_output_index_list % vocab_size # ( 100, width )

            topk_output_list = [ topk_output_index[:, i] for i in range(width) ] # width * ( 100, )
            topk_hidden_list = (
                [ [] for _ in range(width) ],
                [ [] for _ in range(width) ]) # 2 * width * (1, 100, 512)
            topk_cum_prob_list = [ [] for _ in range(width) ] # width * ( 100, )
            for i, (beam_index, output_index) in enumerate(zip(topk_beam_index, topk_output_index)):
                for k, (bi, oi) in enumerate(zip(beam_index, output_index)):
                    topk_hidden_list[0][k].append(beam_hidden_list[0][bi][:, i, :])
                    topk_hidden_list[1][k].append(beam_hidden_list[1][bi][:, i, :])
                    topk_cum_prob_list[k].append(beam_output_list[i][vocab_size * bi + oi])
                    next_output_list[i].append(output_list[i][bi] + [ oi ])
            output_list = next_output_list

            input_list = [ topk_output.unsqueeze(0) for topk_output in topk_output_list ] # width * ( 1, 100 )
            hidden_list = (
                [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[0] ],
                [ torch.stack(topk_hidden, dim=1) for topk_hidden in topk_hidden_list[1] ]) # 2 * width * ( 1, 100, 512 )
            hidden_list = [ ( hidden, context ) for hidden, context in zip(*hidden_list) ]
            cum_prob_list = [ torch.cuda.FloatTensor(topk_cum_prob) for topk_cum_prob in topk_cum_prob_list ] # width * ( 100, )

        SOS_idx = self.vocab.word2idx['<SOS>']
        outputs = [ [ SOS_idx ] + o[0] for o in output_list ]
        return outputs,words

