import pandas as pd
import h5py
import numpy as np
from loader.data_loader import CustomVocab, CustomDataset, Corpus


class MSVDVocab(CustomVocab):
    """ MSVD Vocaburary """

    def load_captions(self):
        df = pd.read_csv(self.caption_fpath)
        df = df[df['Language'] == 'English']
        df = df[pd.notnull(df['Description'])]
        captions = df['Description'].values

        df_s = pd.read_csv('dataset/dataset_v1(EmVideo-S)/EmVideo_trainval_captions.csv')
        style_s = df_s['EmCaptions'].values

        df_l = pd.read_csv('dataset/dataset_v2_EmVideo-L/EmVideo_trainval_captions.csv','\t')
        style_l = df_l['EmCaptions'].values

        captions = np.concatenate((captions, style_s, style_l),0)
        return captions

class MSVDDataset(CustomDataset):
    """ MSVD Dataset """

    def load_captions(self):
        if len(self.caption_fpath.split('/')[-1]) < 10:
            df = pd.read_csv(self.caption_fpath)
            df = df[df['Language'] == 'English']
            df = df[[ 'VideoID', 'Start', 'End', 'Description' ]]
            df = df[pd.notnull(df['Description'])]
            for video_id, start, end, caption in df.values:
                vid = "{}_{}_{}".format(video_id, start, end)
                self.captions[vid].append(caption)
        else:
            df_l = pd.read_csv(self.caption_fpath,'\t')          
            df_l = df_l[[ 'VideoID','EmCaptions']]
            df_l = df_l[pd.notnull(df_l['EmCaptions'])]
            for video_id, caption in df_l.values:
                self.captions[video_id].append(caption)

            file1 = open('dataset/MSVD/youtube_mapping.txt','r')
            fpth2vid = {}
            for line in file1.readlines():
                line = line.strip()
                vid = line.split(' ')[0]
                fpth = line.split(' ')[1]
                fpth2vid[fpth] = vid
            if self.split == 'train':
                df_s = pd.read_csv('dataset/dataset_v1(EmVideo-S)/EmVideo_trainval_captions.csv')   
            else:
                df_s = pd.read_csv('dataset/dataset_v1(EmVideo-S)/EmVideo_test_captions.csv')            
            df_s = df_s[[ 'VideoID','EmCaptions']]
            df_s = df_s[pd.notnull(df_s['EmCaptions'])]
            for video_id, caption in df_s.values:
                vid = fpth2vid[video_id]
                self.captions[vid].append(caption)                



    def load_video_feats(self):
        models = [ self.C.vis_encoder.app_feat, self.C.vis_encoder.mot_feat  ]
        for model in models:
            if len(self.caption_fpath.split('/')[-1]) < 10:
                fpath = self.C.loader.split_video_feat_fpath_tpl.format(self.C.corpus, model, self.split)
                fin = h5py.File(fpath, 'r')
                for vid in fin.keys():
                    feats = fin[vid].value
                    feats_len = len(feats)

                    # Sample fixed number of frames
                    sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                    feats = feats[sampled_idxs]
                    
                    self.video_feats[vid][model] = feats
                fin.close()

            else: 
                fpath = self.C.loader.split_video_feat_fpath_tpl_em.format(model, self.split)
                fin = h5py.File(fpath, 'r')
                for vid in fin.keys():
                    feats = fin[vid].value
                    feats_len = len(feats)
                    # Sample fixed number of frames
                    sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                    feats = feats[sampled_idxs]
                    vid_re = 'vid'+vid 
                    self.video_feats[vid_re][model] = feats
                fin.close()
                
                fpath_s = self.C.loader.split_video_feat_fpath_tpl.format(self.C.corpus, model, self.split)
                fin = h5py.File(fpath_s, 'r')
                for vid in fin.keys():
                    feats = fin[vid].value
                    feats_len = len(feats)
                    # Sample fixed number of frames
                    sampled_idxs = np.linspace(0, len(feats) - 1, self.C.loader.frame_sample_len, dtype=int)
                    feats = feats[sampled_idxs]
                    self.video_feats[vid][model] = feats
                fin.close()

class MSVD(Corpus):
    """ MSVD Corpus """

    def __init__(self, C):
        super(MSVD, self).__init__(C, MSVDVocab, MSVDDataset)

