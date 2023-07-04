import torch
import torch.nn as nn
from models.transformer.Models import Encoder_emo

class VisualEncoder(nn.Module):
    def __init__(self, app_feat, mot_feat, app_input_size, mot_input_size, app_output_size, mot_output_size):
        super(VisualEncoder, self).__init__()
        self.app_feat = app_feat
        self.app_input_size = app_input_size
        self.app_output_size = app_output_size

        self.app_linear = nn.Linear(self.app_input_size, self.app_output_size)
        self.vis_enc = Encoder_emo(n_layers=1, n_head=1, d_k=32, d_v=32, d_model=self.app_output_size, d_inner=512)

    def forward(self, app_feats):
        app_outputs = self.app_linear(app_feats)
        feat, *_ = self.vis_enc(app_outputs)
        return feat

