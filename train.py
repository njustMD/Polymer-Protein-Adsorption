"""
训练 ContactMatrixVAE 模型 (v4: 样本能量权重)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from model import ContactMatrixVAE, vae_loss_weighted
import numpy as np


def load_data_with_weights(data_dir='training_samples'):
    """
    加载接触矩阵和对应的能量权重。
    返回: (data, weights, n_aa)
    """
    files = sorted([f for f in os.listdir(data_dir)
                    if f.startswith('dis_matrix.dat_')])
    matrices = []
    weights = []

    for fname in files:
        mat = []
        with open(os.path.join(data_dir, fname)) as f:
            for line in f:
                if not line.strip():
                    continue
                vals = [float(x) for x in line.strip().split()[1:]]
                mat.append(vals)
        matrices.append(mat)

        # 找对应的权重文件: gro/edr_dis_matrix_X_Y.dat
        sample_id = fname.replace('dis_matrix.dat_', '')
        edr_path = os.path.join(data_dir, 'Sample_Weights', f'edr_dis_matrix_{sample_id}.dat')
        w = 1.0  # 默认权重
        if os.path.exists(edr_path):
            with open(edr_path) as f:
                for line in f:
                    if 'weight=' in line:
                        try:
                            w = float(line.split('weight=')[1].split()[0])
                        except (ValueError, IndexError):
                            w = 1.0
                        break
        weights.append(w)

    data = np.array(matrices, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    # 归一化权重使均值为 1
    w_mean = weights.mean()
    if w_mean > 0:
        weights = weights / w_mean

    return data, weights, len(matrices[0][0])


class WeightedDataset(Dataset):
    """带权重的 Tensor Dataset"""
    def __init__(self, data, weights):
        self.data = data
        self.weights = weights

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.weights[idx]


def train_epoch(model, loader, optimizer, beta, temperature, device):
    model.train()
    total_loss = total_recon = total_kl = 0.0
    total_weight = 0.0
    for batch in loader:
        x = batch[0].to(device)
        w = batch[1].to(device)
        optimizer.zero_grad()
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss_weighted(x_recon, x, mu, logvar,
                                             beta=beta, temperature=temperature,
                                             sample_weights=w)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
        total_weight += w.sum().item()
    n = len(loader.dataset)
    return total_loss * len(loader) / n, total_recon * len(loader) / n, total_kl * len(loader) / n


@torch.no_grad()
def eval_epoch(model, loader, beta, temperature, device):
    model.eval()
    total_loss = total_recon = total_kl = 0.0
    for batch in loader:
        x = batch[0].to(device)
        w = batch[1].to(device)
        x_recon, mu, logvar = model(x)
        loss, recon, kl = vae_loss_weighted(x_recon, x, mu, logvar,
                                             beta=beta, temperature=temperature,
                                             sample_weights=w)
        total_loss += loss.item()
        total_recon += recon.item()
        total_kl += kl.item()
    n = len(loader.dataset)
    return total_loss * len(loader) / n, total_recon * len(loader) / n, total_kl * len(loader) / n


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ---- 加载数据 + 权重 ----
    data, weights, n_aa = load_data_with_weights('training_samples')
    data_mean = data.mean()
    data_std = data.std()
    print(f"Data: {data.shape[0]} x ({data.shape[1]}, {n_aa})")
    print(f"  mean={data_mean:.4f}, std={data_std:.4f}, "
          f"range=[{data.min():.3f}, {data.max():.3f}]")
    print(f"Weights: mean={weights.mean():.4f}, std={weights.std():.4f}, "
          f"min={weights.min():.4f}, max={weights.max():.4f}")

    # ---- Train/Val split (固定 seed，跟踪权重) ----
    n_total = len(data)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    rng = np.random.RandomState(42)
    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    tensor_data = torch.from_numpy(data)
    tensor_weights = torch.from_numpy(weights)

    train_ds = Subset(WeightedDataset(tensor_data, tensor_weights), train_idx)
    val_ds = Subset(WeightedDataset(tensor_data, tensor_weights), val_idx)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    print(f"Train: {n_train}, Val: {n_val}")

    # ---- 模型 ----
    model = ContactMatrixVAE(
        n_monomers=25, n_amino_acids=n_aa,
        latent_dim=8, gcn_hidden=64, gcn_out=32, cnn_hidden=128,
    ).to(device)
    model.set_data_stats(data_mean, data_std)

    adj_path = 'processed/protein_adj_norm.pt'
    if os.path.exists(adj_path):
        adj_norm = torch.load(adj_path)
        model.set_adjacency(adj_norm)
        print(f"Adjacency loaded: {model.adj_norm.shape}")
    else:
        print("WARNING: no adjacency, using identity")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    # ---- 训练设置 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=40
    )

    n_epochs = 500
    warmup_epochs = 80
    beta_target = 0.5
    temperature = 0.5
    best_val_loss = float('inf')
    patience = 80
    no_improve = 0

    for epoch in range(1, n_epochs + 1):
        beta = beta_target * min(1.0, epoch / warmup_epochs)

        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, beta, temperature, device)
        val_loss, val_recon, val_kl = eval_epoch(
            model, val_loader, beta, temperature, device)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_recon': val_recon, 'val_kl': val_kl,
            }, 'processed/vae_best.pt')
        else:
            no_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:3d} | β={beta:.3f} T={temperature:.2f} lr={lr:.1e} | "
                  f"train L={train_loss:.2f} R={train_recon:.2f} KL={train_kl:.2f} | "
                  f"val L={val_loss:.2f} R={val_recon:.2f} KL={val_kl:.2f}")

        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}")

    # ---- 加载最佳模型 ----
    checkpoint = torch.load('processed/vae_best.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # ---- 构建经验隐分布 ----
    # 用训练集的 data (不含权重) 构建
    train_data_loader = DataLoader(
        Subset(TensorDataset_from_data(tensor_data), train_idx),
        batch_size=32, shuffle=False)
    model.build_empirical_latent(train_data_loader, device=device)
    torch.save(model.empirical_mu, 'processed/empirical_mu.pt')

    # ---- 生成 ----
    os.makedirs('generated', exist_ok=True)
    n_generate = 10

    gen_emp = model.generate_empirical(n_generate, noise_std=0.1, device=device).cpu().numpy()
    print(f"\n--- Generated (empirical) ---")
    print(f"  mean={gen_emp.mean():.4f}, std={gen_emp.std():.4f}")
    for i in range(n_generate):
        __save_mat(f'generated/empirical_{i+1}.dat', gen_emp[i], n_aa)

    # ---- 评估重建 ----
    print("\nReconstruction quality (val):")
    for i in range(min(5, len(val_ds))):
        x = val_ds[i][0].unsqueeze(0).to(device)
        recon_x = model.reconstruct(x).cpu().numpy()[0]
        orig_x = x.cpu().numpy()[0]
        mse = np.mean((recon_x - orig_x) ** 2)
        mae = np.mean(np.abs(recon_x - orig_x))
        corr = np.corrcoef(orig_x.flatten(), recon_x.flatten())[0, 1]
        print(f"  Sample {i}: MSE={mse:.4f}  MAE={mae:.4f}  r={corr:.4f}")

    return model


class TensorDataset_from_data(Dataset):
    """简单的 TensorDataset 适配（无权重）"""
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return (self.data[idx],)


def __save_mat(fname, mat, n_aa):
    with open(fname, 'w') as f:
        for row_idx in range(mat.shape[0]):
            vals = "\t".join(f"{mat[row_idx, j]:.6f}" for j in range(n_aa))
            f.write(f"{row_idx+1}\t{vals}\n")


if __name__ == '__main__':
    main()
