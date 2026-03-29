"""
train_seq.py
============
片段级分类，每个片段 = 一个样本，帧序列直接送入 BiLSTM/BiGRU。

特征输入：离线提取的 (T_frames, D) 帧序列 .npy
  features/<feat_type>/data/<id>/<seg>.npy

数据流：
  (T_frames, D) → pack_padded_sequence → BiLSTM/BiGRU
  → valid mean pooling（只对有效帧）→ 分类头 → HC / DP

SVM：
  对 (T_frames, D) 做 mean+std+max 统计池化 → (3D,) 定长向量

结果保存：
  <savedir>/<model>_<feat_type>_seed<seed>/
    ├── log.txt
    ├── plot.png
    ├── best_metrics.json
    └── classification_report.txt

运行示例：
  python train_seq.py --model bilstm --feat_type logmel     --gpu 0
  python train_seq.py --model bigru  --feat_type wav2vec2   --gpu 0
  python train_seq.py --model svm    --feat_type hubert     --gpu -1
"""

import os
import json
import random
import logging
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.optim as optim

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from collections import defaultdict # 用于最后计算平均指标

# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
FEAT_ROOT = 'features'
TRAIN_TXT = 'train.txt'
VAL_TXT   = 'val.txt'


# ─────────────────────────────────────────────────────────────────────────────
# 工具
# ─────────────────────────────────────────────────────────────────────────────
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def setup_logger(savedir: str, name: str = 'train') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
    fh  = logging.FileHandler(os.path.join(savedir, 'log.txt'), encoding='utf-8')
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_txt(txt_path: str):
    """解析 '<rel_wav_path> <label>'，返回 (paths, labels)。"""
    paths, labels = [], []
    with open(txt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rel_path, label = line.rsplit(' ', 1)
            paths.append(rel_path)
            labels.append(int(label))
    return paths, labels


def wav_to_npy(rel_wav_path: str, feat_root: str, feat_type: str) -> str:
    return os.path.join(
        feat_root, feat_type,
        str(Path(rel_wav_path).with_suffix('.npy'))
    )


def compute_metrics(y_true, y_pred) -> dict:
    target_names = ["HC", "DP"]
    try:
        report = classification_report(
            y_true, y_pred, target_names=target_names, output_dict=True
        )
    except Exception:
        report = classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        )
    return {
        'accuracy'        : accuracy_score(y_true, y_pred),
        'macro_f1'        : report['macro avg']['f1-score'],
        'macro_precision' : report['macro avg']['precision'],
        'macro_recall'    : report['macro avg']['recall'],
        'HC_f1'           : report.get('HC', {}).get('f1-score', 0),
        'HC_precision'    : report.get('HC', {}).get('precision', 0),
        'HC_recall'       : report.get('HC', {}).get('recall', 0),
        'DP_f1'           : report.get('DP', {}).get('f1-score', 0),
        'DP_precision'    : report.get('DP', {}).get('precision', 0),
        'DP_recall'       : report.get('DP', {}).get('recall', 0),
        'micro_f1'        : f1_score(y_true, y_pred, average='micro'),
        'micro_precision' : precision_score(y_true, y_pred, average='micro'),
        'micro_recall'    : recall_score(y_true, y_pred, average='micro'),
    }


def log_metrics(metrics: dict, prefix: str, logger):
    logger.info(
        f"[{prefix}] acc={metrics['accuracy']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"macro_P={metrics['macro_precision']:.4f} | "
        f"macro_R={metrics['macro_recall']:.4f}"
    )
    logger.info(
        f"[{prefix}] HC: f1={metrics['HC_f1']:.4f} "
        f"P={metrics['HC_precision']:.4f} R={metrics['HC_recall']:.4f} | "
        f"DP: f1={metrics['DP_f1']:.4f} "
        f"P={metrics['DP_precision']:.4f} R={metrics['DP_recall']:.4f}"
    )
    logger.info(
        f"[{prefix}] micro: f1={metrics['micro_f1']:.4f} "
        f"P={metrics['micro_precision']:.4f} R={metrics['micro_recall']:.4f}"
    )


def save_metrics(metrics: dict, savedir: str):
    with open(os.path.join(savedir, 'best_metrics.json'), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset：片段级，每个样本 = 一个 .npy 帧序列
# ─────────────────────────────────────────────────────────────────────────────
class SegmentDataset(Dataset):
    """
    每个样本 = 一个片段的帧序列。
    __getitem__ 返回：
      feat  : Tensor (T_frames, D)  — 帧序列，长度可变
      label : int
    """

    def __init__(self, paths: list, labels: list,
                 feat_root: str, feat_type: str, logger):
        self.samples = []
        missing = 0
        for rel_path, label in zip(paths, labels):
            npy_path = wav_to_npy(rel_path, feat_root, feat_type)
            if not os.path.exists(npy_path):
                missing += 1
                continue
            self.samples.append((npy_path, label))
        if missing:
            logger.warning(f'{missing} npy files not found and skipped.')
        logger.info(f'Dataset size: {len(self.samples)} segments')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        feat = torch.tensor(
            np.load(npy_path).astype(np.float32)
        )                                          # (T_frames, D)
        return feat, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    变长帧序列 padding。
    返回：
      padded      : (B, T_max, D)   — 帧级 padding
      frame_lens  : (B,)            — 每个样本的真实帧数（用于 pack）
      labels      : (B,)
    """
    feats, labels = zip(*batch)
    frame_lens = torch.LongTensor([f.shape[0] for f in feats])
    padded     = pad_sequence(feats, batch_first=True)   # (B, T_max, D)
    labels     = torch.stack(list(labels))
    return padded, frame_lens, labels


# ─────────────────────────────────────────────────────────────────────────────
# 模型：单层 BiLSTM / BiGRU，帧级输入
# ─────────────────────────────────────────────────────────────────────────────
class FrameSeqClassifier(nn.Module):
    """
    输入  x          : (B, T_max, D)   — padding 后的帧序列
          frame_lens : (B,)            — 每个样本的真实帧数
    输出  logits     : (B, 2)

    pack_padded_sequence 确保 LSTM 只在有效帧上计算，
    padding 帧不参与梯度，不影响结果。
    """

    def __init__(self, feat_dim: int, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2, rnn_type: str = 'lstm'):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat_dim)

        rnn_cls  = nn.LSTM if rnn_type == 'lstm' else nn.GRU
        self.rnn = rnn_cls(
            input_size   = feat_dim,
            hidden_size  = hidden,
            num_layers   = num_layers,
            bidirectional= True,
            batch_first  = True,
            dropout      = dropout if num_layers > 1 else 0.0,
        )

        # attention 加权：对有效帧做加权聚合
        self.attn = nn.Linear(hidden * 2, 1)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor,
                frame_lens: torch.Tensor) -> torch.Tensor:
        """
        x          : (B, T_max, D)
        frame_lens : (B,)  CPU tensor（pack 要求）
        """
        x = self.input_norm(x)

        # pack：LSTM 只计算有效帧，跳过 padding
        packed     = pack_padded_sequence(
            x, frame_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.rnn(packed)
        out, _     = pad_packed_sequence(packed_out, batch_first=True)
        # out: (B, T_max, hidden*2)，padding 位置为 0

        # attention 加权（mask 掉 padding 帧）
        T_max       = out.shape[1]
        valid       = torch.arange(T_max, device=out.device).unsqueeze(0) \
                      < frame_lens.unsqueeze(1).to(out.device)   # (B, T_max)
        attn_scores = self.attn(out).squeeze(-1)                 # (B, T_max)
        attn_scores = attn_scores.masked_fill(~valid, float('-inf'))
        w           = torch.softmax(attn_scores, dim=1).unsqueeze(-1)
        ctx         = (out * w).sum(dim=1)                       # (B, hidden*2)

        return self.classifier(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# 评估
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for padded, frame_lens, labels in loader:
            padded     = padded.to(device)
            labels     = labels.to(device)

            logits     = model(padded, frame_lens)
            loss       = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_preds)
    return avg_loss, metrics, all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# LSTM / GRU 训练
# ─────────────────────────────────────────────────────────────────────────────
def run_rnn(args, savedir: str, logger,train_paths, train_labels, val_paths, val_labels):
    device = torch.device(
        f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available()
        else 'cpu'
    )
    logger.info(f'Device: {device}')

    # train_paths, train_labels = load_txt(args.train_txt)
    # val_paths,   val_labels   = load_txt(args.val_txt)

    train_ds = SegmentDataset(
        train_paths, train_labels, args.feat_root, args.feat_type, logger
    )
    val_ds = SegmentDataset(
        val_paths, val_labels, args.feat_root, args.feat_type, logger
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=8, pin_memory=True
    )

    # 自动推断 feat_dim
    sample_feat, _, _ = next(iter(train_loader))
    feat_dim = sample_feat.shape[-1]
    logger.info(f'feat_dim: {feat_dim}')

    rnn_type  = 'lstm' if args.model == 'bilstm' else 'gru'
    model     = FrameSeqClassifier(
        feat_dim   = feat_dim,
        hidden     = args.hidden,
        num_layers = args.layers,
        dropout    = args.dropout,
        rnn_type   = rnn_type,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    logger.info(
        f'Params: {sum(p.numel() for p in model.parameters()):,} | '
        f'Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}'
    )

    loss_pic_tr  = []
    loss_pic_val = []
    acc_pic_tr   = []
    acc_pic_val  = []
    best_macro_f1   = -1.0
    best_dp_f1      = -1.0
    best_state_dict = None
    best_metrics    = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for padded, frame_lens, labels in train_loader:
            padded = padded.to(device)
            labels = labels.to(device)
            # frame_lens 留在 CPU（pack_padded_sequence 要求）

            logits = model(padded, frame_lens)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=-1) == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        loss_pic_tr.append(train_loss)
        acc_pic_tr.append(train_acc)
        logger.info(
            f'Epoch [{epoch}/{args.epochs}] '
            f'Loss: {train_loss:.4f}  Acc: {train_acc:.4f}'
        )

        val_loss, metrics, _, _ = evaluate(model, val_loader, criterion, device)
        loss_pic_val.append(val_loss)
        acc_pic_val.append(metrics['accuracy'])
        log_metrics(metrics, f'Epoch {epoch} Val', logger)
        scheduler.step(metrics['macro_f1'])

        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1   = metrics['macro_f1']
            best_metrics    = metrics
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(
                f"★ Best  macro_f1={best_macro_f1:.4f}  "
                f"acc={metrics['accuracy']:.4f}"
            )
        if metrics['DP_f1'] > best_dp_f1:
            best_dp_f1 = metrics['DP_f1']
            logger.info(f"★ Best DP_f1={best_dp_f1:.4f}")

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_pic_tr,  label='Training Loss')
    plt.plot(loss_pic_val, label='Validation Loss')
    plt.plot(acc_pic_tr,   label='Training Accuracy')
    plt.plot(acc_pic_val,  label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title(f'{args.model.upper()} + {args.feat_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'plot.png'))
    plt.close()

    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, os.path.join(savedir, 'best_model.pt'))
    logger.info(f'Best model saved → {savedir}/best_model.pt')

    _, final_metrics, y_true, y_pred = evaluate(
        model, val_loader, criterion, device
    )
    report_str = classification_report(
        y_true, y_pred, target_names=["HC", "DP"], digits=4
    )
    logger.info('\n' + report_str)
    with open(os.path.join(savedir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    save_metrics(best_metrics, savedir)
    logger.info(f'All results saved → {savedir}/')
    return best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# SVM：统计池化处理变长帧序列
# ─────────────────────────────────────────────────────────────────────────────
def stat_pool(frames: np.ndarray) -> np.ndarray:
    """
    (T_frames, D) → (3D,)
    mean / std / max 沿帧轴计算，输出定长向量。
    """
    mean = frames.mean(axis=0)
    std  = frames.std(axis=0) if frames.shape[0] > 1 else np.zeros_like(mean)
    maxv = frames.max(axis=0)
    return np.concatenate([mean, std, maxv], axis=0).astype(np.float32)


def load_svm_features(paths, labels, feat_root, feat_type, logger):
    X, y  = [], []
    missing = 0
    for rel_path, label in zip(paths, labels):
        npy_path = wav_to_npy(rel_path, feat_root, feat_type)
        if not os.path.exists(npy_path):
            missing += 1
            continue
        frames = np.load(npy_path)     # (T_frames, D)
        X.append(stat_pool(frames))    # (3D,)
        y.append(label)
    if missing:
        logger.warning(f'{missing} npy files not found and skipped.')
    return np.stack(X, axis=0), np.array(y)


def run_svm(args, savedir: str, logger,train_paths, train_labels, val_paths, val_labels):
    logger.info("=" * 60)
    logger.info(f"Model: SVM  |  Feature: {args.feat_type}  |  Pool: mean+std+max")
    logger.info("=" * 60)

    # train_paths, train_labels = load_txt(args.train_txt)
    # val_paths,   val_labels   = load_txt(args.val_txt)

    logger.info('Building features ...')
    X_train, y_train = load_svm_features(
        train_paths, train_labels, args.feat_root, args.feat_type, logger
    )
    X_val, y_val = load_svm_features(
        val_paths, val_labels, args.feat_root, args.feat_type, logger
    )
    logger.info(f'Train: {X_train.shape} | Val: {X_val.shape}')

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    logger.info('Fitting SVM ...')
    svm = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', probability=True, random_state=args.seed
    )
    svm.fit(X_train, y_train)

    for split, X, y in [('Train', X_train, y_train), ('Val', X_val, y_val)]:
        preds   = svm.predict(X)
        metrics = compute_metrics(y, preds)
        log_metrics(metrics, split, logger)

    y_val_pred = svm.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    report_str  = classification_report(
        y_val, y_val_pred, target_names=["HC", "DP"], digits=4
    )
    logger.info('\n' + report_str)
    with open(os.path.join(savedir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    joblib.dump(svm,    os.path.join(savedir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(savedir, 'svm_scaler.pkl'))
    save_metrics(val_metrics, savedir)
    logger.info(f'All results saved → {savedir}/')
    return val_metrics


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        required=True,
                        choices=['bilstm', 'bigru', 'svm'])
    parser.add_argument('--feat_type',    required=True,
                        choices=['logmel', 'wav2vec2', 'hubert', 'emotion2vec'])
    parser.add_argument('--gpu',          type=int,   default=0)
    parser.add_argument('--seed',         type=int,   default=42)
    parser.add_argument('--savedir',      type=str,   default='a_5k')
    parser.add_argument('--feat_root',    type=str,   default=FEAT_ROOT)
    parser.add_argument('--train_txt',    type=str,   default=TRAIN_TXT)
    parser.add_argument('--val_txt',      type=str,   default=VAL_TXT)
    parser.add_argument('--hidden',       type=int,   default=256)
    parser.add_argument('--layers',       type=int,   default=2)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--lr',           type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    args = parser.parse_args()

    setup_seed(args.seed)

    exp_name = f'{args.model}_{args.feat_type}_seed{args.seed}'
    savedir  = os.path.join(args.savedir, exp_name)
    os.makedirs(savedir, exist_ok=True)

    logger = setup_logger(savedir, name=exp_name)
    logger.info(f'Experiment : {exp_name}')
    logger.info(f'savedir    : {savedir}')


    # 1. 读取并合并所有的 txt 数据
    train_paths, train_labels = load_txt(args.train_txt)
    val_paths, val_labels     = load_txt(args.val_txt)

    all_paths  = np.array(train_paths + val_paths)
    all_labels = np.array(train_labels + val_labels)

    # 2. 从路径中提取受试者 ID 作为 groups
    # 假设路径格式是 data/<id>/<seg>.wav，ID 就是倒数第二级目录
    groups = np.array([Path(p).parent.name for p in all_paths])
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    fold_metrics = [] # 用来存每一折的最优结果
    
    # 开始 3 折循环
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(all_paths, all_labels, groups), 1):
        # 为当前折创建独立的保存目录（例如：a/bilstm_logmel_seed42/fold_1）
        fold_savedir = os.path.join(savedir, f'fold_{fold}')
        os.makedirs(fold_savedir, exist_ok=True)
        
        # 给当前折设置专属的 logger
        fold_logger = setup_logger(fold_savedir, name=f'{exp_name}_fold{fold}')
        fold_logger.info(f"========== Start Fold {fold} ==========")
        
        # 切分当前折的数据（注意转回 list 给 Dataset 用）
        fold_train_paths  = all_paths[train_idx].tolist()
        fold_train_labels = all_labels[train_idx].tolist()
        fold_val_paths    = all_paths[val_idx].tolist()
        fold_val_labels   = all_labels[val_idx].tolist()
        
        # 运行模型
        if args.model == 'svm':
            best = run_svm(args, fold_savedir, fold_logger, fold_train_paths, fold_train_labels, fold_val_paths, fold_val_labels)
        else:
            best = run_rnn(args, fold_savedir, fold_logger, fold_train_paths, fold_train_labels, fold_val_paths, fold_val_labels)
            
        fold_metrics.append(best)
    
    # 统计所有折的平均结果
    logger = setup_logger(savedir, name='average_results') # 用主目录的 logger
    logger.info("========== 5-Fold Cross Validation Final Results ==========")
    
    avg_metrics = defaultdict(float)
    for metrics in fold_metrics:
        for k, v in metrics.items():
            avg_metrics[k] += v
            
    for k in avg_metrics:
        avg_metrics[k] /= 5.0 # 除以折数
        
    for k, v in avg_metrics.items():
        logger.info(f'  Average {k}: {v:.4f}')
        
    # 保存最终平均指标
    save_metrics(avg_metrics, savedir)
