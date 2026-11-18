"""
VAE for polymer-protein contact matrix generation.
Encoder: GCN (amino acid spatial graph) → 1D CNN (monomer chain) → latent
Decoder: latent → 1D CNN + GCN → output
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """单层图卷积: H' = σ(Â @ H @ W)  其中 Â = D^{-1/2}(A+I)D^{-1/2}"""

    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.activation = activation

    def forward(self, H, adj_norm):
        """
        H: (B, N, in_dim), adj_norm: (N, N)
        returns: (B, N, out_dim)
        """
        B, N, _ = H.shape
        adj_expanded = adj_norm.unsqueeze(0).expand(B, -1, -1)
        aggregated = torch.bmm(adj_expanded, H)
        out = aggregated @ self.weight + self.bias
        if self.activation:
            out = F.relu(out)
        return out


class ContactMatrixVAE(nn.Module):
    """
    Encoder: z → GCN(AA graph) → project → 1D CNN(monomer) → pool → μ,σ
    Decoder: z → FC → reshape → 1D CNN(monomer) → GCN(AA graph refine) → output
    """

    def __init__(self, n_monomers=25, n_amino_acids=241, latent_dim=8,
                 gcn_hidden=64, gcn_out=32, cnn_hidden=128):
        super().__init__()
        self.n_monomers = n_monomers
        self.n_aa = n_amino_acids
        self.latent_dim = latent_dim

        # ---- Encoder: GCN on AA ----
        self.gcn_enc1 = GCNLayer(n_monomers, gcn_hidden)   # 25 → 64
        self.gcn_enc2 = GCNLayer(gcn_hidden, gcn_out)       # 64 → 32
        self.aa_to_monomer = nn.Linear(gcn_out, n_monomers)  # 32 → 25

        # ---- Encoder: 1D CNN on monomer axis ----
        self.monomer_conv1 = nn.Conv1d(n_amino_acids, 64, kernel_size=3, padding=1)
        self.monomer_conv2 = nn.Conv1d(64, cnn_hidden, kernel_size=3, padding=1)

        # ---- VAE latent ----
        self.fc_mu = nn.Linear(cnn_hidden, latent_dim)
        self.fc_logvar = nn.Linear(cnn_hidden, latent_dim)

        # ---- Decoder: FC expand ----
        self.fc_dec1 = nn.Linear(latent_dim, 128)
        self.fc_dec2 = nn.Linear(128, cnn_hidden * n_monomers)

        # ---- Decoder: 1D CNN on monomer axis ----
        self.deconv1 = nn.ConvTranspose1d(cnn_hidden, 64, kernel_size=3, padding=1)
        self.deconv2 = nn.ConvTranspose1d(64, n_amino_acids, kernel_size=3, padding=1)

        # ---- Decoder: GCN refinement (symmetric to encoder) ----
        self.gcn_dec1 = GCNLayer(n_monomers, gcn_hidden)   # 25 → 64
        self.gcn_dec2 = GCNLayer(gcn_hidden, n_monomers, activation=False)  # 64 → 25

        # ---- Adjacency matrix ----
        self.register_buffer('adj_norm', torch.eye(n_amino_acids))

        # ---- Data statistics for standardization ----
        self.register_buffer('data_mean', torch.zeros(1))
        self.register_buffer('data_std', torch.ones(1))

    def set_adjacency(self, adj_norm):
        self.adj_norm = adj_norm[:self.n_aa, :self.n_aa].clone().detach()

    def set_data_stats(self, mean, std):
        self.data_mean = torch.tensor(mean, dtype=torch.float32)
        self.data_std = torch.tensor(std, dtype=torch.float32)

    def normalize(self, x):
        return (x - self.data_mean) / (self.data_std + 1e-8)

    def unnormalize(self, x):
        return x * self.data_std + self.data_mean

    def encode(self, x):
        """x: (B, 25, N_aa), normalized"""
        B = x.shape[0]

        # GCN on AA graph
        h = x.transpose(1, 2)                     # (B, N_aa, 25)
        h = self.gcn_enc1(h, self.adj_norm)       # (B, N_aa, 64)
        h = self.gcn_enc2(h, self.adj_norm)       # (B, N_aa, 32)
        h = self.aa_to_monomer(h)                 # (B, N_aa, 25)
        h = h.transpose(1, 2).contiguous()        # (B, 25, N_aa)

        # 1D CNN along monomer axis
        h = h.transpose(1, 2)                     # (B, N_aa, 25)
        h = F.relu(self.monomer_conv1(h))         # (B, 64, 25)
        h = F.relu(self.monomer_conv2(h))         # (B, cnn_hidden, 25)

        # Global pool → latent
        h = h.mean(dim=2)                         # (B, cnn_hidden)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """z: (B, latent_dim) → x_recon: (B, 25, N_aa), normalized"""
        B = z.shape[0]

        # FC expand
        h = F.relu(self.fc_dec1(z))
        h = self.fc_dec2(h)
        h = h.view(B, -1, self.n_monomers)        # (B, cnn_hidden, 25)

        # 1D Transposed CNN
        h = F.relu(self.deconv1(h))               # (B, 64, 25)
        h = self.deconv2(h)                       # (B, N_aa, 25)
        h = h.transpose(1, 2)                     # (B, 25, N_aa)

        # GCN refinement (symmetric to encoder GCN)
        h = h.transpose(1, 2)                     # (B, N_aa, 25)
        h = self.gcn_dec1(h, self.adj_norm)       # (B, N_aa, 64)
        h = self.gcn_dec2(h, self.adj_norm)       # (B, N_aa, 25)
        h = h.transpose(1, 2)                     # (B, 25, N_aa)

        return h

    def forward(self, x):
        x_norm = self.normalize(x)
        mu, logvar = self.encode(x_norm)
        z = self.reparameterize(mu, logvar)
        x_recon_norm = self.decode(z)
        x_recon = self.unnormalize(x_recon_norm)
        return x_recon, mu, logvar

    def generate(self, n_samples, device='cpu'):
        """从先验 N(0,1) 采样生成"""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        with torch.no_grad():
            x_norm = self.decode(z)
            return self.unnormalize(x_norm)

    def generate_empirical(self, n_samples, noise_std=0.1, device='cpu'):
        """
        从经验隐分布采样生成新接触矩阵。
        从存储的训练数据 μ 向量中随机选取，加入高斯噪声进行扰动。
        """
        if not hasattr(self, 'empirical_mu') or self.empirical_mu is None:
            raise RuntimeError("Call build_empirical_latent() first")
        idx = torch.randint(0, len(self.empirical_mu), (n_samples,), device=device)
        mu_base = self.empirical_mu[idx].to(device)
        z = mu_base + noise_std * torch.randn_like(mu_base)
        with torch.no_grad():
            x_norm = self.decode(z)
            return self.unnormalize(x_norm)

    def generate_interpolate(self, n_samples, noise_std=0.05, device='cpu'):
        """
        在经验隐分布中两两插值生成新样本。
        """
        if not hasattr(self, 'empirical_mu') or self.empirical_mu is None:
            raise RuntimeError("Call build_empirical_latent() first")
        n_stored = len(self.empirical_mu)
        idx1 = torch.randint(0, n_stored, (n_samples,), device=device)
        idx2 = torch.randint(0, n_stored, (n_samples,), device=device)
        mu1 = self.empirical_mu[idx1].to(device)
        mu2 = self.empirical_mu[idx2].to(device)
        alpha = torch.rand(n_samples, 1, device=device)
        z = alpha * mu1 + (1 - alpha) * mu2 + noise_std * torch.randn_like(mu1)
        with torch.no_grad():
            x_norm = self.decode(z)
            return self.unnormalize(x_norm)

    @torch.no_grad()
    def build_empirical_latent(self, dataloader, device='cpu'):
        """编码所有训练数据，收集 μ 向量作为经验隐分布"""
        self.eval()
        mu_list = []
        for batch in dataloader:
            x = batch[0].to(device)
            x_norm = self.normalize(x)
            mu, _ = self.encode(x_norm)
            mu_list.append(mu.cpu())
        self.empirical_mu = torch.cat(mu_list, dim=0)
        print(f"Empirical latent built: {self.empirical_mu.shape} "
              f"μ mean={self.empirical_mu.mean().item():.3f} "
              f"std={self.empirical_mu.std().item():.3f}")
        return self.empirical_mu

    def reconstruct(self, x):
        """给定输入的重建（取 μ 而非采样）"""
        x_norm = self.normalize(x)
        mu, _ = self.encode(x_norm)
        with torch.no_grad():
            x_recon_norm = self.decode(mu)
            return self.unnormalize(x_recon_norm)


def vae_loss(x_recon, x, mu, logvar, beta=0.1):
    """VAE loss = MSE + β * KL"""
    B = x.shape[0]
    recon_loss = F.mse_loss(x_recon, x, reduction='sum') / B
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def vae_loss_weighted(x_recon, x, mu, logvar, beta=0.1, temperature=0.5,
                     sample_weights=None):
    """
    距离加权 + 样本加权的 VAE loss。
    sample_weights: (B,) 每个样本的能量权重，None 表示等权。
    """
    B = x.shape[0]

    # 逐元素权重：真实距离越小，权重越大
    with torch.no_grad():
        weight = torch.exp(-x / temperature)  # (B, 25, N_aa)
        weight = weight / weight.mean()

    # 加权 MSE
    sq_error = (x_recon - x) ** 2
    recon_loss_per_sample = (weight * sq_error).sum(dim=(1, 2))  # (B,)

    # KL per sample
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B,)

    # 样本权重
    if sample_weights is not None:
        w = sample_weights / sample_weights.sum() * B  # 归一化保持 loss 量级
        recon_loss = (w * recon_loss_per_sample).sum() / B
        kl_loss = (w * kl_per_sample).sum() / B
    else:
        recon_loss = recon_loss_per_sample.sum() / B
        kl_loss = kl_per_sample.sum() / B
    return recon_loss + beta * kl_loss, recon_loss, kl_loss
