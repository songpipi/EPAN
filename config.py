import os
import time

class VocabConfig:
    init_word2idx = { '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3 }
    embedding_size = 300
    pretrained = 'GloVe'


class MSVDLoaderConfig:
    train_caption_fpath = "workspace/EmCap/data/MSVD/metadata/train.csv"
    val_caption_fpath = "workspace/EmCap/data/MSVD/metadata/val.csv"
    test_caption_fpath = "workspace/EmCap/data/MSVD/metadata/test.csv"

    em_train_caption_fpath = "dataset/dataset_v2_EmVideo-L/EmVideo_trainval_captions.csv"
    em_test_caption_fpath = "dataset/dataset_v2_EmVideo-L/EmVideo_test_captions.csv"

    min_count = 1
    max_caption_len = 15

    split_video_feat_fpath_tpl = "workspace/EmCap/data/{}/features/{}_{}.hdf5"
    frame_sample_len = 30

    split_video_feat_fpath_tpl_em = "workspace/EmCap/data/EmVideo-L/features/{}_{}.hdf5"

    split_negative_vids_fpath = "workspace/EmCap/data/MSVD/metadata/neg_vids_{}.json" 
    split_negative_emvids_fpath = "workspace/EmCap/data/MSVD/metadata/L_v2v/neg_Emvids_{}_L.json" 

    num_workers = 1


class VisualEncoderConfig:
    app_feat, app_feat_size = 'clip', 512
    feat_size = app_feat_size

class Emo_transConfig:
    num_layers = 4
    num_heads = 4
    dim_k = 32
    dim_v = 32
    dim_inner = 512
    dropout = 0.1


class PhraseEncoderConfig:
    SA_num_layers = 1; assert SA_num_layers == 1
    SA_num_heads = 1; assert SA_num_heads == 1
    SA_dim_k = 32
    SA_dim_v = 32
    SA_dim_inner = 512
    SA_dropout = 0.1


class DecoderConfig:
    sem_align_hidden_size = 512
    sem_attn_hidden_size = 512
    rnn_num_layers = 1
    rnn_hidden_size = 512
    max_teacher_forcing_ratio = 1.0
    min_teacher_forcing_ratio = 1.0


class Config:
    seed = 0

    corpus = 'MSVD' 
    
    pretrained_fpath = None
    vocab = VocabConfig
    loader = MSVDLoaderConfig
    vis_encoder = VisualEncoderConfig
    phr_encoder = PhraseEncoderConfig
    decoder = DecoderConfig
    emo_transformer = Emo_transConfig

    """ Optimization """
    epochs = 30
    gradient_clip = 5.0 # None if not used
    PS_threshold = 0.2  
      
    emo_num = 179    
    batch_size = 128
    lr = 0.0007
    CA_lambda = 0.2
    CA_beta = 0.2
    em_wei = 0.1

    file_path = 'EmVideo'

    """ Evaluation """
    metrics_full = ['Bleu_1', 'Bleu_2', 'Bleu_3',  'Bleu_4', 'CIDEr', 'METEOR', 'ROUGE_L', 'Acc_sw', 'Acc_c', 'BFS', 'CFS' ]

    """ Log """
    log_dpath = os.path.join("logs", file_path )
    ckpt_fpath_tpl = os.path.join("checkpoints", file_path, "{}.ckpt")
    ckpt_fpath_tpl_em = os.path.join("checkpoints", file_path, "{}.ckpt")
    result_dir = os.path.join("results", file_path)
    score_fpath = os.path.join( result_dir, "results.txt")
    score_fpath_stage1 = os.path.join( result_dir, "results_stage1.txt")

    """ TensorboardX """
    tx_train_loss = "loss/train"
    tx_train_cross_entropy_loss = "loss/train/cross_entropy"
    tx_train_contrastive_attention_loss = "loss/train/contrastive_attention"
    tx_train_cls_loss =  "loss/train/cls_loss"
    tx_val_loss = "loss/val"
    tx_val_cross_entropy_loss = "loss/val/cross_entropy"
    tx_val_contrastive_attention_loss = "loss/val/contrastive_attention"
    tx_val_cls_loss = "loss/val/cls_loss"
    tx_lr = "params/lr"
    tx_teacher_forcing_ratio = "params/teacher_forcing_ratio"


