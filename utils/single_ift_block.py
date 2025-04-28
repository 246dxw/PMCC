import torch
from einops import rearrange
import torch.nn as nn


class IFT_Module(nn.Module):
    """ IFT """

    def __init__(self, clip_model, beta_s=1.0
                 ):
        super().__init__()

        self.softmax = nn.Softmax(-1)
        input_dim = clip_model.text_projection.shape[1]     # 512
        pre_dim1 = input_dim // 8                           # 64
        pre_dim2 = input_dim // 8                           # 64

        self.beta_s = beta_s
        self.scale = 0.1

        self.pre_project = nn.Sequential(  # 3 layers
            nn.Linear(input_dim, pre_dim1),         # [B, 512] -> [B, 64]
            nn.BatchNorm1d(pre_dim1),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim1, pre_dim2),          # [B, 64] -> [B, 64]
            nn.BatchNorm1d(pre_dim2),
            nn.ReLU(inplace=True),

            nn.Linear(pre_dim2, input_dim * 3)      # [B, 64] -> [B, 1536]
        ).half()

        self.post_project = nn.Sequential(  # only one layer
            nn.Linear(input_dim, input_dim)         # [B, 512] -> [B, 512]
        ).half()

        self.logit_scale = clip_model.logit_scale

    def forward(self, Ft, Fv, Fvs_bank):
        '''
        Fvs with shape (batch, C): source visual output w/o attnpool
        Fvt with shape (N, C): classes of target visual output w/o attnpool
        '''
        out_fv = self.pre_project(Fv)  # (batch, 3 * C)
        out_fvs = self.pre_project(Fvs_bank)  # (N, 3 * C)

        q_fv, k_fv, v_fv = tuple(rearrange(out_fv, 'b (d k) -> k b d ', k=3))
        q_fvs, k_fvs, v_fvs = tuple(rearrange(out_fvs, 'b (d k) -> k b d ', k=3))

        # a_q = self.a_q.expand(B, -1)
        # attn_weight = self.softmax(self.scale * a_q @ k_fvs.permute(1, 0))  # (batch, N)
        # v_out = attn_weight @ v_fvs
        #
        # a_k = self.a_k.expand(B, -1)
        # As = self.softmax(self.scale * q_fv @ a_k.permute(1, 0))  # (batch, N)
        #
        # Fsa = Fv + self.post_project(As @ v_out)  # (batch, C)



        As = self.softmax(self.scale * q_fv @ k_fvs.permute(1, 0))  # (batch, N)

        Fsa = Fv + self.post_project(As @ v_fvs)  # (batch, C)

        Fsa = Fsa / Fsa.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = self.beta_s * logit_scale * Fsa @ Ft.t()
        # logits = self.beta_s * logit_scale * torch.einsum('bc,bkc->bk', Fsa, Ft)

        return logits, Fsa