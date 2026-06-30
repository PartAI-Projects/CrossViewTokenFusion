import torch
import torch.nn as nn


class FusionBlock(nn.Module):
    def __init__(self, cfg, embed_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=cfg.model.fusion_heads,
            dropout=cfg.model.fusion_dropout,
            batch_first=True
        )

        self.ln_cc = nn.LayerNorm(embed_dim)
        self.ln_mlo = nn.LayerNorm(embed_dim)

        # Initialize V as identity
        with torch.no_grad():
            E = embed_dim
            self.attn.in_proj_weight[2*E:3*E].copy_(torch.eye(E))
            self.attn.in_proj_bias[2*E:3*E].zero_()

        # Freeze V via gradient hook
        def freeze_vq_grad(grad):
            grad = grad.clone()
            grad[2*E:3*E] = 0
            return grad

        self.attn.in_proj_weight.register_hook(freeze_vq_grad)
        self.attn.in_proj_bias.register_hook(freeze_vq_grad)

    def forward(self, cc_features, mlo_features):
        cc_norm = self.ln_cc(cc_features)
        mlo_norm = self.ln_mlo(mlo_features)
        
        att_cc, _ = self.attn(cc_norm, mlo_norm, mlo_norm)
        att_mlo, _ = self.attn(mlo_norm, cc_norm, cc_norm)
        
        # Pool attention output to a single token
        att_cc_token = att_cc.mean(dim=1, keepdim=True)  # [B,1,D]
        att_mlo_token = att_mlo.mean(dim=1, keepdim=True)  # [B,1,D]
        
        # Append as new tokens
        cc_extended = torch.cat([att_cc_token, cc_features], dim=1)  # [B, L_cc+1, D]
        mlo_extended = torch.cat([att_mlo_token, mlo_features], dim=1)  # [B, L_mlo+1, D]
        
        return cc_extended, mlo_extended
