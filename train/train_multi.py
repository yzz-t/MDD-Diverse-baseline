"""
train_multi.py — 双模态抑郁检测训练脚本（文本 + 音频）
=================================================================
融合方式通过 --fusion_type 切换：
  multi : MULTLAModel 跨模态注意力融合（原有流程）
  early : 早期融合 concat → SVM 或 BiLSTM（--classifier svm / bilstm）
 
用法示例：
  # MulT 跨模态融合
  python train_multi.py \
      --fusion_type multi \
      --audio_feat  wav2vec2 \
      --text_feat   feat_seq \
      --epochs 50 --batch_size 16 --lr 1e-4
 
  # 早期融合 + SVM
  python train_multi.py \
      --fusion_type early \
      --classifier  svm \
      --audio_feat  wav2vec2 \
      --text_feat   feat_mean
 
  # 早期融合 + BiLSTM
  python train_multi.py \
      --fusion_type early \
      --classifier  bilstm \
      --audio_feat  hubert \
      --text_feat   feat_layer4 \
      --epochs 50 --batch_size 32 --lr 1e-4
 
支持的 --audio_feat : logmel | wav2vec2 | hubert | emotion2vec
支持的 --text_feat  :
  multi 模式 → feat_mean | feat_layer4 | feat_seq
  early 模式 → feat_mean | feat_layer4   （池化向量，不支持 feat_seq）
"""
 
# ─────────────────────────────── imports ────────────────────────────────────
import os, sys, math, json, logging, argparse
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  公共常量
# ═══════════════════════════════════════════════════════════════════════════
 
AUDIO_DIMS = {
    'logmel':      80,
    'wav2vec2':    1024,
    'hubert':      1024,
    'emotion2vec': 768,
}
 
TEXT_SUBDIR = {
    'mean':   'mean',
    'layer4': 'layer4',
    'seq':    'seq',
}
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  1.  TransformerEncoder（MulT 用）
# ═══════════════════════════════════════════════════════════════════════════
 
class ScaledDotProductAttention(nn.Module):
    def __init__(self, attn_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(attn_dropout)
 
    def forward(self, q, k, v, key_padding_mask=None):
        d_k    = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = self.drop(torch.softmax(scores, dim=-1))
        return torch.matmul(attn, v)
 
 
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h   = num_heads
        self.d_k = embed_dim // num_heads
        self.w_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_o = nn.Linear(embed_dim, embed_dim, bias=False)
        self.sdp = ScaledDotProductAttention(attn_dropout)
 
    def forward(self, q, k, v, key_padding_mask=None):
        T_q, B, _ = q.shape
 
        def _proj_split(linear, x):
            T = x.size(0)
            return linear(x).view(T, B, self.h, self.d_k).permute(1, 2, 0, 3)
 
        Q = _proj_split(self.w_q, q)
        K = _proj_split(self.w_k, k)
        V = _proj_split(self.w_v, v)
 
        out = self.sdp(Q, K, V, key_padding_mask)
        out = out.permute(2, 0, 1, 3).contiguous().view(T_q, B, -1)
        return self.w_o(out)
 
 
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0):
        super().__init__()
        self.mha    = MultiHeadAttention(embed_dim, num_heads, attn_dropout)
        self.fc1    = nn.Linear(embed_dim, 4 * embed_dim)
        self.fc2    = nn.Linear(4 * embed_dim, embed_dim)
        self.norm1  = nn.LayerNorm(embed_dim)
        self.norm2  = nn.LayerNorm(embed_dim)
        self.drop_r = nn.Dropout(relu_dropout)
        self.drop_s = nn.Dropout(res_dropout)
 
    def forward(self, q, k=None, v=None, key_padding_mask=None):
        if k is None:
            k = v = q
        residual = q
        x = self.mha(q, k, v, key_padding_mask)
        x = self.norm1(residual + self.drop_s(x))
        residual = x
        x = self.fc2(self.drop_r(F.relu(self.fc1(x))))
        x = self.norm2(residual + self.drop_s(x))
        return x
 
 
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, layers,
                 attn_dropout=0.0, relu_dropout=0.0,
                 res_dropout=0.0, embed_dropout=0.0, attn_mask=False):
        super().__init__()
        self.embed_drop = nn.Dropout(embed_dropout)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads,
                                    attn_dropout, relu_dropout, res_dropout)
            for _ in range(layers)
        ])
 
    def forward(self, q, k=None, v=None, key_padding_mask=None):
        q = self.embed_drop(q)
        if k is not None:
            k = self.embed_drop(k)
            v = self.embed_drop(v)
        for layer in self.layers:
            q = layer(q, k, v, key_padding_mask)
        return q
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  2.  MULTLAModel（multi 融合）
# ═══════════════════════════════════════════════════════════════════════════
 
class MULTLAModel(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.orig_d_l      = hp.orig_d_l
        self.orig_d_a      = hp.orig_d_a
        self.d_l = self.d_a = hp.d_common
        self.num_heads     = hp.num_heads
        self.layers        = hp.layers
        self.attn_dropout  = hp.attn_dropout
        self.relu_dropout  = hp.relu_dropout
        self.res_dropout   = hp.res_dropout
        self.out_dropout   = hp.out_dropout
        self.embed_dropout = hp.embed_dropout
 
        combined_dim = 2 * self.d_l
 
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, bias=False)
 
        self.trans_l_with_a = self._make_enc(self.d_l)
        self.trans_a_with_l = self._make_enc(self.d_a)
        self.trans_l_mem    = self._make_enc(self.d_l, layers=3)
        self.trans_a_mem    = self._make_enc(self.d_a, layers=3)
 
        self.proj1     = nn.Linear(combined_dim, combined_dim)
        self.proj2     = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, hp.output_dim)
 
    def _make_enc(self, embed_dim, layers=None):
        return TransformerEncoder(
            embed_dim=embed_dim, num_heads=self.num_heads,
            layers=layers if layers else self.layers,
            attn_dropout=self.attn_dropout, relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,   embed_dropout=self.embed_dropout,
        )
 
    def forward(self, x_l, x_a, pad_mask_l=None, pad_mask_a=None):
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = x_a.transpose(1, 2)
 
        proj_l = self.proj_l(x_l).permute(2, 0, 1)
        proj_a = self.proj_a(x_a).permute(2, 0, 1)
 
        h_l = self.trans_l_with_a(proj_l, proj_a, proj_a, pad_mask_a)
        h_l = self.trans_l_mem(h_l)
        last_h_l = h_l[-1]
 
        h_a = self.trans_a_with_l(proj_a, proj_l, proj_l, pad_mask_l)
        h_a = self.trans_a_mem(h_a)
        last_h_a = h_a[-1]
 
        last_hs = torch.cat([last_h_l, last_h_a], dim=1)
        out = self.proj2(F.dropout(F.relu(self.proj1(last_hs)),
                                   p=self.out_dropout, training=self.training))
        out = out + last_hs
        logits = self.out_layer(out)
        return logits, last_hs
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  3.  早期融合 BiLSTM 分类器
# ═══════════════════════════════════════════════════════════════════════════
 
class EarlyFusionBiLSTM(nn.Module):
    """
    输入  x      : (B, 1, concat_dim)
                   concat_dim = text_dim(1024) + audio_dim(D_a)
    输出  logits : (B, 2)
    """
    def __init__(self, concat_dim: int, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(concat_dim)
        self.lstm = nn.LSTM(
            input_size=concat_dim, hidden_size=hidden,
            num_layers=num_layers, bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn       = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, seq_len, concat_dim)"""
        x      = self.input_norm(x)
        out, _ = self.lstm(x)
        w      = torch.softmax(self.attn(out), dim=1)
        ctx    = (out * w).sum(dim=1)
        return self.classifier(ctx)
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  4.  Dataset
# ═══════════════════════════════════════════════════════════════════════════
 
class DepressionDataset(Dataset):
    """
    通用 Dataset，同时支持 multi / early 两种融合模式。
 
    multi 模式：返回 (text_feat, audio_feat, label)
      text_feat  : (T_l, 1024)  feat_seq 时 T_l>1，池化向量时 T_l=1
      audio_feat : (T_a, D_a)
 
    early 模式：返回 (concat_vec, label)
      concat_vec : (1, 1024+D_a)  音频均值池化后与文本向量 concat
    """
 
    def __init__(self, list_file, data_root, audio_feat, text_feat,
                 fusion_type='multi'):
        assert fusion_type in ('multi', 'early')
        if fusion_type == 'early':
            assert text_feat in ('mean', 'layer4'), \
                "early 融合只支持 feat_mean / feat_layer4"
 
        self.audio_feat  = audio_feat
        self.text_feat   = text_feat
        self.fusion_type = fusion_type
        self.data_root   = Path(data_root)
        self.audio_root  = self.data_root / 'a_features' / audio_feat
        self.text_root   = self.data_root / ('bert_features_' + TEXT_SUBDIR[text_feat])
 
        self.samples = []
        with open(list_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts    = line.split()
                rel_stem = str(Path(parts[0]).with_suffix(''))
                label    = int(parts[1])
                self.samples.append((rel_stem, label))
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        rel_stem, label = self.samples[idx]
 
        text_np  = np.load(self.text_root  / (rel_stem + '.npy')).astype(np.float32)
        audio_np = np.load(self.audio_root / (rel_stem + '.npy')).astype(np.float32)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
 
        if self.fusion_type == 'multi':
            if text_np.ndim == 1:
                text_np = text_np[np.newaxis, :]          # (1, 1024)
            return (
                torch.from_numpy(text_np),                # (T_l, 1024)
                torch.from_numpy(audio_np),               # (T_a, D_a)
                torch.tensor(label, dtype=torch.long),
            )
        else:
            # early: 音频均值池化 → concat
            text_vec  = text_np.flatten()[:1024]          # (1024,)
            audio_vec = audio_np.mean(axis=0)             # (D_a,)
            concat    = np.concatenate([text_vec, audio_vec])[np.newaxis, :]
            return (
                torch.from_numpy(concat),                 # (1, 1024+D_a)
                torch.tensor(label, dtype=torch.long),
            )
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  5.  collate_fn
# ═══════════════════════════════════════════════════════════════════════════
 
def collate_fn_multi(batch):
    texts, audios, labels = zip(*batch)
 
    text_lens  = torch.tensor([t.shape[0] for t in texts])
    texts_pad  = pad_sequence(texts, batch_first=True, padding_value=0.0)
    text_mask  = torch.arange(texts_pad.shape[1]).unsqueeze(0) >= text_lens.unsqueeze(1)
 
    audio_lens = torch.tensor([a.shape[0] for a in audios])
    audios_pad = pad_sequence(audios, batch_first=True, padding_value=0.0)
    audio_mask = torch.arange(audios_pad.shape[1]).unsqueeze(0) >= audio_lens.unsqueeze(1)
 
    return texts_pad, audios_pad, text_mask, audio_mask, torch.stack(labels)
 
 
def collate_fn_early(batch):
    feats, labels = zip(*batch)
    return torch.stack(feats), torch.stack(labels)
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  6.  评估工具
# ═══════════════════════════════════════════════════════════════════════════
 
def _compute_metrics(all_labels, all_preds, avg_loss=None):
    p, r, f, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=[0, 1], average=None, zero_division=0)
    mp, mr, mf, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='macro', zero_division=0)
    m = {
        'HC_precision': float(p[0]), 'HC_recall': float(r[0]), 'HC_f1': float(f[0]),
        'DP_precision': float(p[1]), 'DP_recall': float(r[1]), 'DP_f1': float(f[1]),
        'macro_precision': float(mp), 'macro_recall': float(mr), 'macro_f1': float(mf),
    }
    if avg_loss is not None:
        m['loss'] = avg_loss
    return m
 
 
def _log_metrics(logger, metrics, prefix='Val'):
    logger.info(
        f"  [{prefix}] "
        f"HC  P={metrics['HC_precision']:.4f} "
        f"R={metrics['HC_recall']:.4f} "
        f"F1={metrics['HC_f1']:.4f} | "
        f"DP  P={metrics['DP_precision']:.4f} "
        f"R={metrics['DP_recall']:.4f} "
        f"F1={metrics['DP_f1']:.4f}"
    )
    logger.info(
        f"  [{prefix}] macro  "
        f"P={metrics['macro_precision']:.4f}  "
        f"R={metrics['macro_recall']:.4f}  "
        f"F1={metrics['macro_f1']:.4f}"
    )
 
 
def evaluate_multi(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for texts, audios, text_mask, audio_mask, labels in loader:
            texts      = texts.to(device);      audios     = audios.to(device)
            text_mask  = text_mask.to(device);  audio_mask = audio_mask.to(device)
            labels     = labels.to(device)
            logits, _  = model(texts, audios, text_mask, audio_mask)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return _compute_metrics(all_labels, all_preds, total_loss / len(all_labels))
 
 
def evaluate_bilstm(model, loader, device, criterion):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(device), labels.to(device)
            logits        = model(feats)
            total_loss   += criterion(logits, labels).item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return _compute_metrics(all_labels, all_preds, total_loss / len(all_labels))
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  7.  输出目录 & 日志初始化
# ═══════════════════════════════════════════════════════════════════════════
 
def setup_output(args):
    if args.fusion_type == 'multi':
        run_name = f"multi_{args.audio_feat}_{args.text_feat}"
    else:
        run_name = f"early_{args.classifier}_{args.audio_feat}_{args.text_feat}"
 
    output_dir = Path(args.output_root) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
 
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'train.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ]
    )
    logger = logging.getLogger('train')
    logger.info(f"Run        : {run_name}")
    logger.info(f"Output dir : {output_dir}")
    logger.info(json.dumps(vars(args), indent=2, ensure_ascii=False))
    return output_dir, logger
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  8.  加载所有样本为 numpy（SVM 专用）
# ═══════════════════════════════════════════════════════════════════════════
 
def load_flat_features(list_file, data_root, audio_feat, text_feat, logger):
    audio_root = Path(data_root) / 'a_features' / audio_feat
    text_root  = Path(data_root) / ('bert_features_' + TEXT_SUBDIR[text_feat])
 
    X, y, missing = [], [], 0
    with open(list_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
 
    for line in lines:
        parts    = line.split()
        rel_stem = str(Path(parts[0]).with_suffix(''))
        label    = int(parts[1])
        t_path   = text_root  / (rel_stem + '.npy')
        a_path   = audio_root / (rel_stem + '.npy')
        if not t_path.exists() or not a_path.exists():
            missing += 1
            continue
        text_vec  = np.load(t_path).astype(np.float32).flatten()[:1024]
        audio_arr = np.load(a_path).astype(np.float32)
        audio_vec = audio_arr.mean(axis=0) if audio_arr.ndim > 1 else audio_arr
        X.append(np.concatenate([text_vec, audio_vec]))
        y.append(label)
 
    if missing:
        logger.warning(f"{missing} samples skipped (file not found)")
    logger.info(f"Loaded {len(X)} samples, feat_dim={X[0].shape[0] if X else '?'}")
    return np.stack(X, axis=0), y
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  9.  训练：MulT（multi）
# ═══════════════════════════════════════════════════════════════════════════
 
def train_multi(args, output_dir, logger):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")
 
    train_list = '202x_train.txt'
    val_list   =  '202x_val.txt'
 
    train_ds = DepressionDataset(train_list, args.data_root,
                                  args.audio_feat, args.text_feat, 'multi')
    val_ds   = DepressionDataset(val_list,   args.data_root,
                                  args.audio_feat, args.text_feat, 'multi')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn_multi,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn_multi,
                               num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train : {len(train_ds)}  Val : {len(val_ds)}")
 
    class HP: pass
    hp = HP()
    hp.orig_d_l = 1024;  hp.orig_d_a = AUDIO_DIMS[args.audio_feat]
    hp.d_common = args.d_common;   hp.num_heads = args.num_heads
    hp.layers   = args.layers;     hp.output_dim = 2
    hp.attn_dropout  = args.attn_dropout;  hp.relu_dropout = args.relu_dropout
    hp.res_dropout   = args.res_dropout;   hp.out_dropout  = args.out_dropout
    hp.embed_dropout = args.embed_dropout
 
    model = MULTLAModel(hp).to(device)
    logger.info(f"MULTLAModel params : "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
    labels_all = [s[1] for s in train_ds.samples]
    n_hc, n_dp = labels_all.count(0), labels_all.count(1)
    total  = n_hc + n_dp
    weight = torch.tensor([total/(2*n_hc), total/(2*n_dp)],
                           dtype=torch.float).to(device)
    logger.info(f"Class weight  HC={weight[0]:.3f}  DP={weight[1]:.3f}")
 
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
 
    best_f1, best_epoch = -1.0, -1
    ckpt_path = output_dir / 'best_model.pt'
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        for texts, audios, text_mask, audio_mask, labels in train_loader:
            texts      = texts.to(device);      audios     = audios.to(device)
            text_mask  = text_mask.to(device);  audio_mask = audio_mask.to(device)
            labels     = labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(texts, audios, text_mask, audio_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n_samples  += labels.size(0)
 
        train_loss  = total_loss / n_samples
        val_metrics = evaluate_multi(model, val_loader, device, criterion)
        scheduler.step(val_metrics['macro_f1'])
 
        is_best = val_metrics['macro_f1'] > best_f1
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}"
            + ("  ← best" if is_best else ""))
        _log_metrics(logger, val_metrics)
 
        if is_best:
            best_f1, best_epoch = val_metrics['macro_f1'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_metrics': val_metrics, 'args': vars(args)}, ckpt_path)
 
        with open(output_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train_loss': train_loss,
                                **val_metrics}, ensure_ascii=False) + '\n')
 
    logger.info(f"Done. Best macro-F1={best_f1:.4f} @ epoch {best_epoch} → {ckpt_path}")
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  10.  训练：早期融合 SVM
# ═══════════════════════════════════════════════════════════════════════════
 
def train_early_svm(args, output_dir, logger):
    train_list = '202x_train.txt'
    val_list   =  '202x_val.txt'
 
    logger.info("Loading train features ...")
    X_train, y_train = load_flat_features(
        train_list, args.data_root, args.audio_feat, args.text_feat, logger)
    logger.info("Loading val features ...")
    X_val, y_val = load_flat_features(
        val_list, args.data_root, args.audio_feat, args.text_feat, logger)
 
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
 
    logger.info("Fitting SVM (RBF, C=1.0, class_weight=balanced) ...")
    svm = SVC(kernel='rbf', C=1.0, gamma='scale',
              class_weight='balanced', random_state=42)
    svm.fit(X_train, y_train)
 
    train_metrics = _compute_metrics(y_train, svm.predict(X_train).tolist())
    logger.info("── Train ──")
    _log_metrics(logger, train_metrics, prefix='Train')
 
    y_val_pred  = svm.predict(X_val).tolist()
    val_metrics = _compute_metrics(y_val, y_val_pred)
    logger.info("── Val ──")
    _log_metrics(logger, val_metrics, prefix='Val')
 
    report = classification_report(y_val, y_val_pred,
                                    target_names=['HC', 'DP'], digits=4)
    logger.info('\n' + report)
    (output_dir / 'classification_report.txt').write_text(report, encoding='utf-8')
 
    joblib.dump(svm,    output_dir / 'svm_model.pkl')
    joblib.dump(scaler, output_dir / 'svm_scaler.pkl')
    with open(output_dir / 'best_metrics.json', 'w') as f:
        json.dump({k: float(v) for k, v in val_metrics.items()}, f, indent=4)
    logger.info(f"SVM saved → {output_dir}")
 
 
# ═══════════════════════════════════════════════════════════════════════════
#  11.  训练：早期融合 BiLSTM
# ═══════════════════════════════════════════════════════════════════════════
 
def train_early_bilstm(args, output_dir, logger):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device : {device}")
 
    train_list =  '202x_train.txt'
    val_list   = '202x_val.txt'
 
    train_ds = DepressionDataset(train_list, args.data_root,
                                  args.audio_feat, args.text_feat, 'early')
    val_ds   = DepressionDataset(val_list,   args.data_root,
                                  args.audio_feat, args.text_feat, 'early')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn_early,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn_early,
                               num_workers=args.num_workers, pin_memory=True)
    logger.info(f"Train : {len(train_ds)}  Val : {len(val_ds)}")
 
    concat_dim = 1024 + AUDIO_DIMS[args.audio_feat]
    logger.info(f"Concat dim : 1024 + {AUDIO_DIMS[args.audio_feat]} = {concat_dim}")
 
    model = EarlyFusionBiLSTM(concat_dim=concat_dim,
                               hidden=args.bilstm_hidden,
                               num_layers=args.bilstm_layers,
                               dropout=args.bilstm_dropout).to(device)
    logger.info(f"BiLSTM params : "
                f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
 
    labels_all = [s[1] for s in train_ds.samples]
    n_hc, n_dp = labels_all.count(0), labels_all.count(1)
    total  = n_hc + n_dp
    weight = torch.tensor([total/(2*n_hc), total/(2*n_dp)],
                           dtype=torch.float).to(device)
    logger.info(f"Class weight  HC={weight[0]:.3f}  DP={weight[1]:.3f}")
 
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
 
    best_f1, best_epoch = -1.0, -1
    ckpt_path = output_dir / 'best_model.pt'
 
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, n_samples = 0.0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            n_samples  += labels.size(0)
 
        train_loss  = total_loss / n_samples
        val_metrics = evaluate_bilstm(model, val_loader, device, criterion)
        scheduler.step(val_metrics['macro_f1'])
 
        is_best = val_metrics['macro_f1'] > best_f1
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_metrics['loss']:.4f}"
            + ("  ← best" if is_best else ""))
        _log_metrics(logger, val_metrics)
 
        if is_best:
            best_f1, best_epoch = val_metrics['macro_f1'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_metrics': val_metrics, 'args': vars(args)}, ckpt_path)
 
        with open(output_dir / 'metrics.jsonl', 'a') as f:
            f.write(json.dumps({'epoch': epoch, 'train_loss': train_loss,
                                **val_metrics}, ensure_ascii=False) + '\n')
 
    logger.info(f"Done. Best macro-F1={best_f1:.4f} @ epoch {best_epoch} → {ckpt_path}")
 

# ═══════════════════════════════════════════════════════════════════════════
#  6.  argparse
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='MULTLAModel bimodal depression classification')

    # 路径
    p.add_argument('--data_root',    type=str,
                   default='202_baseline',
                   help='数据根目录')
    p.add_argument('--output_root',  type=str,
                   default='./outputs',
                   help='实验输出根目录，子目录按特征组合自动命名')
    # 融合方式
    p.add_argument('--fusion_type', type=str, default='multi',
                   choices=['multi', 'early'],
                   help='multi=MulT跨模态注意力 | early=concat早期融合')
    p.add_argument('--classifier',  type=str, default='bilstm',
                   choices=['svm', 'bilstm'],
                   help='early 模式下的分类器')

    # 特征选择
    p.add_argument('--audio_feat', type=str, default='wav2vec2',
                   choices=['logmel', 'wav2vec2', 'hubert', 'emotion2vec'],
                   help='使用的音频特征类型')
    p.add_argument('--text_feat',  type=str, default='seq',
                   choices=['mean', 'layer4', 'seq'],
                   help='使用的文本特征类型')

    # 训练超参
    p.add_argument('--epochs',       type=int,   default=15)
    p.add_argument('--batch_size',   type=int,   default=8)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip',         type=float, default=1.0,
                   help='梯度裁剪阈值')
    p.add_argument('--num_workers',  type=int,   default=8)
    p.add_argument('--device',       type=str,   default='cuda')

    # 模型超参
    p.add_argument('--d_common',      type=int,   default=32,
                   help='模态投影后的公共维度')
    p.add_argument('--num_heads',     type=int,   default=4,
                   help='多头注意力头数，需整除 d_common')
    p.add_argument('--layers',        type=int,   default=3,
                   help='Transformer 层数')
    p.add_argument('--attn_dropout',  type=float, default=0.1)
    p.add_argument('--relu_dropout',  type=float, default=0.1)
    p.add_argument('--res_dropout',   type=float, default=0.1)
    p.add_argument('--out_dropout',   type=float, default=0.1)
    p.add_argument('--embed_dropout', type=float, default=0.1)

     # 早期融合 BiLSTM 专属超参
    p.add_argument('--bilstm_hidden',  type=int,   default=256)
    p.add_argument('--bilstm_layers',  type=int,   default=2)
    p.add_argument('--bilstm_dropout', type=float, default=0.3)

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  7.  入口
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    args = parse_args()
 
    if args.fusion_type == 'early' and args.text_feat == 'seq':
        raise ValueError(
            "early 融合不支持 feat_seq，请使用 --text_feat feat_mean 或 feat_layer4")
 
    output_dir, logger = setup_output(args)
 
    if args.fusion_type == 'multi':
        train_multi(args, output_dir, logger)
    elif args.classifier == 'svm':
        train_early_svm(args, output_dir, logger)
    else:
        train_early_bilstm(args, output_dir, logger)
