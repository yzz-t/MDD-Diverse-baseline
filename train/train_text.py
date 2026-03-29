"""
train_bert_cls.py
=================
基于 MacBERT pooler 特征（1024 维）的抑郁检测分类器：
  - SVM    : 片段级，每个片段 (1024,) 作为独立样本
  - BiLSTM : 片段级，每个片段 (k, 1024) 送入 LSTM，attention 加权输出

结果保存：
  <savedir>/svm_seed{seed}/    ← SVM 结果
  <savedir>/bilstm_seed{seed}/ ← BiLSTM 结果
  各自包含 log.txt / plot.png / model文件

运行示例：
  python train_text.py --model svm    --savedir results --seed 42
  python train_text.py --model bilstm --savedir results --seed 42
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

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import joblib


# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
BERT_ROOT = 'bert_features_mean'
BERT_ROOT_SEQ  = 'bert_features_seq'                # BiLSTM seq 用
TRAIN_TXT = 'train.txt'
VAL_TXT   = 'val.txt'
FEAT_DIM  = 1024


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def setup_logger(savedir: str, name: str = 'cls') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(message)s")
    fh  = logging.FileHandler(os.path.join(savedir, 'log.txt'), encoding='utf-8')
    fh.setFormatter(fmt)
    sh  = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_txt(txt_path: str):
    """解析 '<rel_wav_path> <label>'，返回 (paths, labels) 列表。"""
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


def wav_to_npy(rel_wav_path: str, bert_root: str) -> str:
    """将相对 wav 路径映射到对应的 .npy 绝对路径。"""
    return os.path.join(bert_root, str(Path(rel_wav_path).with_suffix('.npy')))


def compute_metrics(y_true, y_pred) -> dict:
    """与 train_202x.py 完全一致的指标集合。"""
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


def log_metrics(metrics: dict, prefix: str, logger: logging.Logger):
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


def save_metrics(metrics: dict, savedir: str, filename: str = 'best_metrics.json'):
    with open(os.path.join(savedir, filename), 'w') as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# 特征加载（SVM 用）
# ─────────────────────────────────────────────────────────────────────────────
def load_features(paths: list, labels: list, bert_root: str, logger: logging.Logger):
    """读取所有 .npy，跳过缺失文件，返回 (X: np.ndarray N×1024, y: list)。"""
    feats, labs = [], []
    missing = 0
    for rel_path, label in zip(paths, labels):
        npy_path = wav_to_npy(rel_path, bert_root)
        if not os.path.exists(npy_path):
            missing += 1
            continue
        feats.append(np.load(npy_path).astype(np.float32))
        labs.append(label)
    if missing:
        logger.warning(f"{missing} npy files not found and skipped.")
    return np.stack(feats, axis=0), labs


# ─────────────────────────────────────────────────────────────────────────────
# SVM
# ─────────────────────────────────────────────────────────────────────────────
def run_svm(savedir: str, logger: logging.Logger):
    logger.info("=" * 60)
    logger.info("Model: SVM  |  Feature: MacBERT pooler (1024-d)")
    logger.info("=" * 60)

    # 读取数据
    train_paths, train_labels = load_txt(TRAIN_TXT)
    val_paths,   val_labels   = load_txt(VAL_TXT)

    logger.info("Loading train features ...")
    X_train, y_train = load_features(train_paths, train_labels, BERT_ROOT, logger)
    logger.info("Loading val features ...")
    X_val,   y_val   = load_features(val_paths,   val_labels,   BERT_ROOT, logger)
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # 标准化
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)

    # 训练
    logger.info("Fitting SVM (linear, C=1.0, class_weight=balanced) ...")
    svm = SVC(
        kernel='rbf', C=1.0, gamma='scale',
        class_weight='balanced', random_state=42
    )
    svm.fit(X_train, y_train)

    # 训练集指标
    y_train_pred = svm.predict(X_train)
    train_metrics = compute_metrics(y_train, y_train_pred)
    log_metrics(train_metrics, "Train", logger)

    # 验证集指标
    y_val_pred = svm.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    log_metrics(val_metrics, "Val", logger)

    # 详细报告
    report_str = classification_report(y_val, y_val_pred,
                                       target_names=["HC", "DP"], digits=4)
    logger.info("\n" + report_str)
    with open(os.path.join(savedir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    # 保存模型 & 指标
    joblib.dump(svm,    os.path.join(savedir, 'svm_model.pkl'))
    joblib.dump(scaler, os.path.join(savedir, 'svm_scaler.pkl'))
    save_metrics(val_metrics, savedir)
    logger.info(f"SVM model + scaler saved to {savedir}/")

    return val_metrics


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM Dataset
# ─────────────────────────────────────────────────────────────────────────────
class BertSegDataset(Dataset):
    def __init__(self, paths, labels, bert_root, logger, use_seq=False):
        self.use_seq = use_seq
        self.samples = []
        missing = 0
        for rel_path, label in zip(paths, labels):
            npy_path = wav_to_npy(rel_path, bert_root)
            if not os.path.exists(npy_path):
                missing += 1
                continue
            self.samples.append((npy_path, label))
        if missing:
            logger.warning(f"{missing} npy files not found and skipped.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        feat = torch.tensor(np.load(npy_path).astype(np.float32))
        # use_seq=False: (1024,) → (1, 1024)
        # use_seq=True:  (T_token, 1024) 直接用
        if not self.use_seq:
            feat = feat.unsqueeze(0)
        return feat, torch.tensor(label, dtype=torch.long)

def collate_fn_seq(batch):
    feats, labels = zip(*batch)
    # feats: list of (T_i, 1024)，T_i 各不同
    from torch.nn.utils.rnn import pad_sequence
    feats_pad = pad_sequence(feats, batch_first=True, padding_value=0.0)  # (B, T_max, 1024)
    lengths   = torch.tensor([f.shape[0] for f in feats])
    labels    = torch.stack(labels)
    return feats_pad, lengths, labels
# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM 模型
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    """
    输入  x      : (B, seq_len=1, 1024)
    输出  logits : (B, 2)
      
    """

    def __init__(self, feat_dim: int = 1024, hidden: int = 256,
                 num_layers: int = 2, dropout: float = 0.3,
                 num_classes: int = 2):
        super().__init__()
        self.input_norm = nn.LayerNorm(feat_dim)
        self.lstm = nn.LSTM(
            input_size    = feat_dim,
            hidden_size   = hidden,
            num_layers    = num_layers,
            bidirectional = True,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0.0,
        )
        self.attn       = nn.Linear(hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden * 2),
            nn.Linear(hidden * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, lengths=None):
        x = self.input_norm(x)
        if lengths is not None:
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(out, batch_first=True)   # (B, T_max, hidden*2)
        else:
            out, _ = self.lstm(x)

        w   = torch.softmax(self.attn(out), dim=1)   # (B, T_max, 1)
        ctx = (out * w).sum(dim=1)                   # (B, hidden*2)
        return self.classifier(ctx)


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM 评估
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_bilstm(model, loader, criterion, device, use_seq=False):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            if use_seq:
                feats, lengths, labels = batch
                feats, lengths, labels = feats.to(device), lengths, labels.to(device)
                logits = model(feats, lengths)
            else:
                feats, labels = batch
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    return total_loss / len(loader), compute_metrics(all_labels, all_preds)


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM 训练
# ─────────────────────────────────────────────────────────────────────────────
def run_bilstm(savedir: str, seed: int, logger: logging.Logger):
    logger.info("=" * 60)
    logger.info("Model: BiLSTM  |  Feature: MacBERT pooler (1024-d)")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 超参
    batch_size   = 32
    epochs       = 45
    lr           = 1e-4
    weight_decay = 0.01

    # 数据
    train_paths, train_labels = load_txt(TRAIN_TXT)
    val_paths,   val_labels   = load_txt(VAL_TXT)

    use_seq = (args.text_feat == 'seq')
    bert_root = BERT_ROOT_SEQ if use_seq else BERT_ROOT

    train_dataset = BertSegDataset(train_paths, train_labels, bert_root, logger, use_seq=use_seq)
    train_loader  = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn_seq if use_seq else None,
        num_workers=8, pin_memory=True)
    val_dataset = BertSegDataset(val_paths, val_labels, bert_root, logger, use_seq=use_seq)
    val_loader  = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn_seq if use_seq else None,
        num_workers=8, pin_memory=True)
    # train_dataset = BertSegDataset(train_paths, train_labels, BERT_ROOT, logger)
    # val_dataset   = BertSegDataset(val_paths,   val_labels,   BERT_ROOT, logger)
    logger.info(
        f"Train: {len(train_dataset)} segments | Val: {len(val_dataset)} segments"
    )

    # train_loader = DataLoader(train_dataset, batch_size=batch_size,
    #                           shuffle=True,  num_workers=8, pin_memory=True)
    # val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
    #                           shuffle=False, num_workers=8, pin_memory=True)

    # 模型
    model     = BiLSTMClassifier(feat_dim=FEAT_DIM).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    logger.info(
        f"Params: {sum(p.numel() for p in model.parameters()):,} total | "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable"
    )

    loss_pic_tr  = []
    loss_pic_val = []
    acc_pic_tr   = []
    acc_pic_val  = []

    best_macro_f1   = -1.0
    best_dp_f1      = -1.0
    best_state_dict = None
    best_metrics    = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct    = 0
        total      = 0

        for batch in train_loader:
            if use_seq:
                feats, lengths, labels = batch
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats, lengths)
            else:
                feats, labels = batch
                feats, labels = feats.to(device), labels.to(device)
                logits = model(feats)
            # logits = model(feats)
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(dim=-1) == labels).sum().item()
            total      += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_acc  = correct / total
        loss_pic_tr.append(train_loss)
        acc_pic_tr.append(train_acc)
        logger.info(
            f"Epoch [{epoch}/{epochs}] Loss: {train_loss:.4f}  Acc: {train_acc:.4f}"
        )

        val_loss, metrics = evaluate_bilstm(model, val_loader, criterion, device,use_seq)
        loss_pic_val.append(val_loss)
        acc_pic_val.append(metrics['accuracy'])
        log_metrics(metrics, f"Epoch {epoch} Val", logger)

        # 以 macro_f1 为准保存最优模型
        if metrics['macro_f1'] > best_macro_f1:
            best_macro_f1   = metrics['macro_f1']
            best_metrics    = metrics
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            logger.info(
                f"★ Best model updated  macro_f1={best_macro_f1:.4f}  "
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
    plt.title('BiLSTM Training and Validation Loss / Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(savedir, 'plot.png'))
    plt.close()

    # 保存最优模型
    model.load_state_dict(best_state_dict)
    torch.save(best_state_dict, os.path.join(savedir, 'bilstm_best.pt'))
    logger.info(f"Best model saved to {savedir}/bilstm_best.pt")

    # 最终详细报告（用最优权重跑一遍 val）
    _, final_metrics = evaluate_bilstm(model, val_loader, criterion, device,use_seq)
    report_str = classification_report(
        *_get_preds(model, val_loader, device),
        target_names=["HC", "DP"], digits=4
    )
    logger.info("\n" + report_str)
    with open(os.path.join(savedir, 'classification_report.txt'), 'w') as f:
        f.write(report_str)

    save_metrics(best_metrics, savedir)
    logger.info(f"All results saved to {savedir}/")
    return best_metrics


def _get_preds(model, loader, device):
    """辅助函数：返回 (y_true, y_pred) 用于 classification_report。"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats  = feats.to(device)
            preds  = model(feats).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.numpy().tolist())
    return all_labels, all_preds


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',   type=str, required=True,
                        choices=['svm', 'bilstm'], help='选择分类器')
    parser.add_argument('--savedir', type=str, default='t_seq',
                        help='结果根目录，子目录按模型和seed自动创建')
    parser.add_argument('--text_feat', type=str, default='seq',
                    choices=['mean', 'layer4', 'seq'])
    parser.add_argument('--seed',    type=int, default=42)
    args = parser.parse_args()

    setup_seed(args.seed)

    # 每个模型独立子目录
    savedir = f'./{args.savedir}_{args.model}'
    os.makedirs(savedir, exist_ok=True)

    logger = setup_logger(savedir, name=args.model)
    logger.info(f"savedir : {savedir}")
    logger.info(f"seed    : {args.seed}")
    
    
    if args.model == 'svm':
        best = run_svm(savedir, logger)
    else:
        best = run_bilstm(savedir, args.seed, logger)

    logger.info("\n最终最优指标:")
    for k, v in best.items():
        logger.info(f"  {k}: {v:.4f}")
