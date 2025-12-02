import torch
import torch.nn as nn

class DecorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features):
        """
        Args:
            features: (Batch_Size, Channel, Height, Width) または (Batch_Size, Dim)
        """
        # U-Netのボトルネックが (B, C, H, W) の場合、
        # 平均プーリングして (B, C) のベクトルにするのが一般的です。
        # これにより、チャネル間の相関をバッチ全体で抑制します。
        if features.dim() == 4:
            # Global Average Pooling: (B, C, H, W) -> (B, C)
            M = features.mean(dim=[2, 3])
        else:
            M = features

        B, D = M.shape

        # --- Algorithm 2 Line 1: Mean & Var ---
        # 論文ではバッチ次元(B)方向で正規化していると読み取れます
        # M: (B, D)
        mean = M.mean(dim=0, keepdim=True) # (1, D)
        var = M.var(dim=0, keepdim=True, unbiased=False) # (1, D)
        std = torch.sqrt(var + self.eps)

        # --- Algorithm 2 Line 3: Normalization ---
        # 各次元を正規化 (Z-score normalization like)
        M_norm = (M - mean) / std

        # --- Algorithm 2 Line 5: Correlation Matrix ---
        # corr = M.T @ M / B (相関行列の定義に基づきバッチサイズで割る)
        # これにより (D, D) の行列ができる
        # 論文の表記 "M^T . M" は内積ですが、正規化後の内積＝コサイン類似度＝相関です
        corr = (M_norm.T @ M_norm) / B

        # --- Algorithm 2 Line 6-7: Extract Non-diagonal ---
        # 対角成分（自分自身との相関は必ず1）を除外するマスクを作成
        diag_mask = torch.eye(D, device=features.device).bool()

        # 非対角成分のみを取り出す (corrの対角成分を0にする、あるいは無視して平均をとる)
        # ここでは二乗和平均などをとるのが一般的ですが、
        # 論文の Line 8: (corr^2.mean()) とあるので、要素ごとの二乗の平均をとります。
        loss_reg = (corr[~diag_mask].pow(2)).mean()
        return loss_reg, corr