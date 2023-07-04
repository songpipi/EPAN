from __future__ import print_function, division

from collections import defaultdict
import json
import os
import random
import gensim
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torchvision import transforms

from loader.transform import Lowercase, PadFirst, PadLast, PadToLength, RemovePunctuation, \
                             SplitWithWhiteSpace, ToIndex, ToTensor, TrimExceptAscii, Truncate


class CustomVocab(object):
    def __init__(self, caption_fpath, init_word2idx, min_count=1, transform=str.split, embedding_size=300,
                 pretrained=None):
        self.caption_fpath = caption_fpath
        self.min_count = min_count
        self.transform = transform
        self.pretrained = pretrained
        self.embedding_size = embedding_size

        self.word2idx = defaultdict(lambda: init_word2idx['<UNK>'])
        self.word2idx.update(init_word2idx)
        self.idx2word = { v: k for k, v in self.word2idx.items() }
        self.word_freq_dict = defaultdict(lambda: 0)
        self.max_sentence_len = -1

        self.em_vocabs = []
        self.em_word2idx = {}
        em_words = open('workspace/EmCap/EmotionEval/179_words.txt','r')
        for idx,ddd in enumerate(em_words):
            em_word = ddd.split()[0]
            self.em_vocabs.append(em_word)
            self.em_word2idx[em_word] = idx+1
        self.emo_num = idx+1

        self.em2idx = {}
        self.em_cates = []
        ems = open('workspace/EmCap/EmotionEval/34_emotion.txt','r')
        for idx,ddd in enumerate(ems):
            em = ddd.split()[0]
            self.em_cates.append(em)
            self.em2idx[em] = idx+1

        self.build()

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def load_pretrained_embedding(self, name):
        if name == 'GloVe':
            # 400,000 words, 300 dims
            with open("workspace/EmCap/data/Embeddings/GloVe/GloVe_300.json", 'r') as fin:
                w2v = json.load(fin)
        elif name == 'Word2Vec':
            w2v = gensim.models.KeyedVectors.load_word2vec_format(
                'data/Embeddings/Word2Vec/GoogleNews-vectors-negative300.bin',
                binary=True)
        else:
            raise NotImplementedError("Unknown pretrained word embedding: {}".format(w2v))
        return w2v


    def build(self):
        captions = self.load_captions()
        for caption in captions:
            words = self.transform(caption)
            self.max_sentence_len = max(self.max_sentence_len, len(words))
            for word in words:
                self.word_freq_dict[word] += 1

        self.n_vocabs_untrimmed = len(self.word_freq_dict)
        self.n_words_untrimmed = sum(list(self.word_freq_dict.values()))

        words = list(self.word_freq_dict.keys())
        keep_words = [ word for word in words if self.word_freq_dict[word] >= self.min_count ]

        ########## merge the emotion vocab ############
        for word in self.em_vocabs:
            if word not in keep_words:
                keep_words.append(word)

        word_counts = 'vocabs.txt'
        if not os.path.exists(word_counts):
            file = open(word_counts, 'w')
            file.write("\n".join(["%s %d" % (word, freq) for word, freq in self.word_freq_dict.items() if freq >= self.min_count]))
            file.close()

        word_idx = len(self.word2idx)
        for word in keep_words:
            if word in self.word2idx:
                continue

            self.word2idx[word] = word_idx
            self.idx2word[word_idx] = word
            word_idx += 1
        self.n_vocabs = len(self.idx2word.keys())
        self.n_words = sum([ self.word_freq_dict[word] for word in keep_words ])

        self.embedding_weights = np.zeros(( self.n_vocabs, self.embedding_size ))
        self.pretrained_idxs = []
        if self.pretrained is not None:
            w2v = self.load_pretrained_embedding(self.pretrained)
            for idx, word in self.idx2word.items():
                if word not in w2v:
                    self.embedding_weights[idx] = np.random.normal(size=(self.embedding_size,))
                else:
                    self.embedding_weights[idx] = w2v[word]
                    self.pretrained_idxs.append(idx)

        self.embedding_weights_em = np.zeros(( self.emo_num+1, self.embedding_size ))
        self.pretrained_idxs_em = []
        count = 0
        for word in self.em_vocabs:
            count = count + 1
            if word not in w2v:
                self.embedding_weights_em[count] = np.random.normal(size=(self.embedding_size,))
            else:
                self.embedding_weights_em[count] = w2v[word]
                self.pretrained_idxs_em.append(count)

class CustomDataset(Dataset):
    """ Dataset """

    def __init__(self, C, split, caption_fpath, transform_frame=None, transform_caption=None,transform_em_word=None, transform_em=None):
        self.C = C
        self.split = split
        self.caption_fpath = caption_fpath
        self.transform_frame = transform_frame
        self.transform_caption = transform_caption
        self.transform_em = transform_em
        self.transform_em_word = transform_em_word

        self.video_feats = defaultdict(lambda: {})
        #self.video_len = defaultdict(lambda: {})
        self.captions = defaultdict(lambda: [])
        
        self.em_set = []
        em_words = open('workspace/EmCap/EmotionEval/179_words.txt','r')
        for ddd in em_words:
            em_word = ddd.split()[0]
            self.em_set.append(em_word)
        self.word2emo()

        self.data = []

        self.build_video_caption_pairs()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pos, neg = self.data[idx]
        pos_vid, pos_feat, pos_caption, pos_em_word, pos_em = pos
        neg_vid, neg_feat, neg_caption = neg

        if self.transform_frame:
            pos_feat = { model: self.transform_frame(pos_feat) for model, pos_feat in pos_feat.items() }
            neg_feat = { model: self.transform_frame(neg_feat) for model, neg_feat in neg_feat.items() }
        if self.transform_caption:
            pos_caption = self.transform_caption(pos_caption)
            pos_em_word = self.transform_em_word(pos_em_word)
            pos_em = self.transform_em(pos_em)
            neg_caption = self.transform_caption(neg_caption)

        pos = ( pos_vid, pos_feat, pos_caption, pos_em_word, pos_em )
        neg = ( neg_vid, neg_feat, neg_caption )
        return pos, neg

    def load_video_feats(self):
        raise NotImplementedError("You should implement this function.")

    def word2emo(self):
        self.word_to_34class = {}
        main_path = 'workspace/EmCap/EmotionEval/sentiment-words-E/'
        dir = os.listdir(main_path) #34 emotions
        dir.sort()
        for filename in dir:
            emotion_label = [filename.split('.')[0]]
            open_path = os.path.join(main_path, filename)
            f = open(open_path, 'r')
            for line in f.readlines():
                word = line.split()[0]
                if word==[]: continue
                if word in ['annoying', 'worried', 'hopeful','timid']:
                    if word == 'annoying': self.word_to_34class[word] = ['anger','annoyance']
                    elif word == 'worried': self.word_to_34class[word] = ['annoyance','worry']
                    elif word == 'hopeful': self.word_to_34class[word] = ['anticipation','optimistic']
                    elif word == 'timid': self.word_to_34class[word] = ['fear','shy']
                else:
                    self.word_to_34class[word] = emotion_label
        



    def load_negative_vids(self):
        neg_vids_fpath = self.C.loader.split_negative_vids_fpath.format(self.split)
        neg_emvids_fpath = self.C.loader.split_negative_emvids_fpath.format(self.split)

        if len(self.caption_fpath.split('/')[-1]) < 10:
            with open(neg_vids_fpath, 'r') as fin:
                self.vid2neg_vids = json.load(fin)
        else:
            with open(neg_emvids_fpath, 'r') as fin:
                self.vid2neg_vids = json.load(fin)
            
            # K = 10
            # self.vid2neg_vids = defaultdict(lambda: random.sample(vids, K))

    def load_captions(self):
        raise NotImplementedError("You should implement this function.")

    def build_video_caption_pairs(self):
        self.load_captions()
        self.load_video_feats()
        self.load_negative_vids()

        vids = self.captions.keys()
        for pos_idx, pos_vid in enumerate(vids):
            pos_video_feats = self.video_feats[pos_vid]
            neg_vids = self.vid2neg_vids[pos_vid]

            for pos_caption in self.captions[pos_vid]:
                pos_em_word = list(set(pos_caption.split()).intersection(set(self.em_set)))
                pos_em = []
                for x in pos_em_word:
                    classes = self.word_to_34class[x]
                    for word in classes:
                        pos_em.append(word)
                pos_em = set(pos_em) 
                neg_vid = random.choice(neg_vids)
                neg_video_feats = self.video_feats[neg_vid]
                neg_caption = random.choice(self.captions[neg_vid])
                self.data.append((
                    (pos_vid, pos_video_feats, pos_caption, ' '.join(pos_em_word),' '.join(pos_em)),
                    (neg_vid, neg_video_feats, neg_caption) ))


class Corpus(object):
    """ Data Loader """

    def __init__(self, C, vocab_cls=CustomVocab, dataset_cls=CustomDataset):
        self.C = C
        self.vocab = None
        self.train_dataset = None
        self.train_data_loader = None
        self.val_dataset = None
        self.val_data_loader = None
        self.test_dataset = None
        self.test_data_loader = None

        self.CustomVocab = vocab_cls
        self.CustomDataset = dataset_cls

        self.transform_sentence = transforms.Compose([
            TrimExceptAscii(self.C.corpus),
            Lowercase(),
            RemovePunctuation(),
            SplitWithWhiteSpace(),
            Truncate(self.C.loader.max_caption_len),
        ])

        self.build()

    def build(self):
        self.build_vocab()
        self.build_data_loaders()

    def build_vocab(self):
        self.vocab = self.CustomVocab(
            caption_fpath=self.C.loader.train_caption_fpath,
            init_word2idx=self.C.vocab.init_word2idx,
            min_count=self.C.loader.min_count,
            transform=self.transform_sentence,
            embedding_size=self.C.vocab.embedding_size,
            pretrained=self.C.vocab.pretrained)

    def build_data_loaders(self):
        """ Transformation """
        self.transform_frame = transforms.Compose([
            ToTensor(torch.float),
        ])
        self.transform_caption = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.word2idx),
            PadFirst(self.vocab.word2idx['<SOS>']),
            PadLast(self.vocab.word2idx['<EOS>']),
            PadToLength(self.vocab.word2idx['<PAD>'], self.vocab.max_sentence_len + 2), 
            ToTensor(torch.long),
        ])
        self.transform_em_word = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.em_word2idx),
            Truncate(3),
            PadToLength(self.vocab.word2idx['<PAD>'], 3),
            ToTensor(torch.long),
        ])
        self.transform_em = transforms.Compose([
            self.transform_sentence,
            ToIndex(self.vocab.em2idx),
            Truncate(3),
            PadToLength(self.vocab.word2idx['<PAD>'], 3),
            ToTensor(torch.long),
        ])

        self.train_dataset = self.build_dataset("train", self.C.loader.train_caption_fpath)
        self.val_dataset = self.build_dataset("val", self.C.loader.val_caption_fpath)
        self.test_dataset = self.build_dataset("test", self.C.loader.test_caption_fpath)
        self.train_data_loader = self.build_data_loader(self.train_dataset)
        self.val_data_loader = self.build_data_loader(self.val_dataset)
        self.test_data_loader = self.build_data_loader(self.test_dataset)
      
        self.em_train_dataset = self.build_dataset("train", self.C.loader.em_train_caption_fpath)
        self.em_test_dataset = self.build_dataset("test", self.C.loader.em_test_caption_fpath)        
        self.em_train_data_loader = self.build_data_loader(self.em_train_dataset)
        self.em_test_data_loader = self.build_data_loader_test(self.em_test_dataset)

    def build_dataset(self, split, caption_fpath):
        dataset = self.CustomDataset(
            self.C,
            split,
            caption_fpath,
            transform_frame=self.transform_frame,
            transform_caption=self.transform_caption,
            transform_em_word=self.transform_em_word,
            transform_em=self.transform_em)
        return dataset


    def collate_fn(self, batch):
        pos, neg = zip(*batch)
        pos_vids, pos_video_feats, pos_captions, pos_em_word, pos_em = zip(*pos)
        neg_vids, neg_video_feats, neg_captions = zip(*neg)

        pos_video_feats_list = defaultdict(lambda: [])
        for pos_video_feat in pos_video_feats:
            for model, feat in pos_video_feat.items():
                pos_video_feats_list[model].append(feat)
        neg_video_feats_list = defaultdict(lambda: [])
        for neg_video_feat in neg_video_feats:
            for model, feat in neg_video_feat.items():
                neg_video_feats_list[model].append(feat)
        pos_video_feats_list = dict(pos_video_feats_list)
        neg_video_feats_list = dict(neg_video_feats_list)

        for model in pos_video_feats_list:
            pos_video_feats_list[model] = torch.stack(pos_video_feats_list[model], dim=0).float()
            neg_video_feats_list[model] = torch.stack(neg_video_feats_list[model], dim=0).float()
        pos_captions = torch.stack(pos_captions).float()
        pos_em_word = torch.stack(pos_em_word).float()
        pos_em = torch.stack(pos_em).float()
        neg_captions = torch.stack(neg_captions).float()

        """ (batch, seq, feat) -> (seq, batch, feat) """
        pos_captions = pos_captions.transpose(0, 1)
        neg_captions = neg_captions.transpose(0, 1)
        pos = ( pos_vids, pos_video_feats_list, pos_captions, pos_em_word, pos_em )
        neg = ( neg_vids, neg_video_feats_list, neg_captions )
        return pos, neg

    def build_data_loader(self, dataset):
        data_loader = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False, # If sampler is specified, shuffle must be False.
            sampler=RandomSampler(dataset, replacement=False),
            num_workers=self.C.loader.num_workers,
            collate_fn=self.collate_fn)
        data_loader.captions = { k: [ ' '.join(self.transform_sentence(c)) for c in v   ] for k, v in dataset.captions.items()   }
        return data_loader

    def build_data_loader_test(self, dataset):
        data_loader_test = DataLoader(
            dataset,
            batch_size=self.C.batch_size,
            shuffle=False, # If sampler is specified, shuffle must be False.
            num_workers=self.C.loader.num_workers,
            collate_fn=self.collate_fn)
        data_loader_test.captions = { k: [ ' '.join(self.transform_sentence(c)) for c in v   ] for k, v in dataset.captions.items()   }
        return data_loader_test