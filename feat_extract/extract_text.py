"""
extract_text.py
========================
从 Whisper 转录文本重新提取 MacBERT 特征，替换原有 pooler_output。

输出两种版本（分别存两个目录，原有 bert_features/ 保持不变）：
  bert_features_mean/   : last_hidden_state mean pooling        → (1024,)
  bert_features_layer4/ : 最后 4 层先平均再 mean pooling        → (1024,)

目录结构镜像 transcripts/：
  transcripts/root/<id>/<seg>.txt
  → bert_features_mean/data/<id>/<seg>.npy
  → bert_features_layer4/data/<id>/<seg>.npy

运行：
  python extract_text.py
  python extract_text.py --batch_size 64 --device cuda:1
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
TRANSCRIPT_ROOT = 'transcripts'
MACBERT_PATH    = '/huggingface/hub/chinese-macbert-large'
OUT_MEAN        = '/bert_features_mean'
OUT_LAYER4      = '/bert_features_layer4'
MAX_LENGTH      = 512


# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 特征提取
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_batch(texts: list, tokenizer, model, device: torch.device):
    """
    返回两种特征，均为 (B, 1024) numpy float32：
      feat_mean   : last hidden state mean pooling（只对非 padding token 取均值）
      feat_layer4 : 最后 4 层平均后再 mean pooling
    """
    enc = tokenizer(
        texts,
        padding        = True,
        truncation     = True,
        max_length     = MAX_LENGTH,
        return_tensors = 'pt',
    )
    input_ids      = enc['input_ids'].to(device)
    attention_mask = enc['attention_mask'].to(device)

    outputs = model(
        input_ids            = input_ids,
        attention_mask       = attention_mask,
        output_hidden_states = True,
    )

    mask = attention_mask.unsqueeze(-1).float()   # (B, seq, 1)

    # ── mean pooling on last hidden state ──────────────────────────────────
    last_hidden = outputs.last_hidden_state        # (B, seq, 1024)
    feat_mean   = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    # (B, 1024)

    # ── last 4 layers 平均 → mean pooling ──────────────────────────────────
    # hidden_states: tuple[num_layers+1]，每项 (B, seq, 1024)，含 embedding 层
    last4       = torch.stack(outputs.hidden_states[-4:], dim=0)   # (4, B, seq, 1024)
    last4_avg   = last4.mean(dim=0)                                 # (B, seq, 1024)
    feat_layer4 = (last4_avg * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
    # (B, 1024)

    # last_hidden_state: 去掉 padding token，只保留真实 token
    # mask: (B, seq)，值为1的位置是真实token
    seq_lens = attention_mask.sum(dim=1)  # 每条样本的真实token数

    last_hidden_states = []
    for i in range(last_hidden.shape[0]):
        real_len = seq_lens[i].item()
        last_hidden_states.append(
            last_hidden[i, :real_len, :].cpu().float().numpy()  # (T_token_i, 1024)
        )

    return (
        feat_mean.cpu().float().numpy(),
        feat_layer4.cpu().float().numpy(),
        last_hidden_states,  # list of (T_token_i, 1024
    )


# ─────────────────────────────────────────────────────────────────────────────
# 收集所有 transcript 文件
# ─────────────────────────────────────────────────────────────────────────────
def collect_txt_files(transcript_root: str):
    """
    返回 list of (abs_txt_path, rel_stem)。
    
    """
    root    = Path(transcript_root)
    records = []
    for txt_path in sorted(root.rglob('*.txt')):
        rel_stem = str(txt_path.relative_to(root).with_suffix(''))
        records.append((str(txt_path), rel_stem))
    return records


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device        : {device}")
    logger.info(f"MacBERT       : {args.macbert_path}")
    logger.info(f"Transcripts   : {args.transcript_root}")
    logger.info(f"Out mean      : {args.out_mean}")
    logger.info(f"Out layer4    : {args.out_layer4}")

    # ── 加载模型 ──────────────────────────────────────────────────────────────
    logger.info("Loading MacBERT ...")
    tokenizer = AutoTokenizer.from_pretrained(args.macbert_path)
    model     = AutoModel.from_pretrained(args.macbert_path)
    model.eval().to(device)
    logger.info("Model loaded.")

    # ── 收集文件 ──────────────────────────────────────────────────────────────
    records = collect_txt_files(args.transcript_root)
    logger.info(f"Total transcript files : {len(records)}")

    # ── 断点续跑：跳过两个输出都已存在的文件 ──────────────────────────────────
    todo, skip = [], 0
    for txt_path, rel_stem in records:
        npy_mean   = os.path.join(args.out_mean,   rel_stem + '.npy')
        npy_layer4 = os.path.join(args.out_layer4, rel_stem + '.npy')
        npy_seq    = os.path.join(args.out_seq,    rel_stem + '.npy')  # 新增
        if os.path.exists(npy_mean) and os.path.exists(npy_layer4) and os.path.exists(npy_seq):
            skip += 1
        else:
            todo.append((txt_path, rel_stem))
    logger.info(f"Already done : {skip} | To process : {len(todo)}")

    if not todo:
        logger.info("All files already extracted. Done.")
        return

    # ── 按 batch 提取 ─────────────────────────────────────────────────────────
    errors = 0
    for batch_start in tqdm(range(0, len(todo), args.batch_size),
                             desc="Extracting", unit="batch"):
        batch = todo[batch_start: batch_start + args.batch_size]

        # 读文本，空文本用占位符防止 tokenizer 报错
        texts = []
        for txt_path, _ in batch:
            try:
                text = Path(txt_path).read_text(encoding='utf-8').strip()
                texts.append(text if text else '[UNK]')
            except Exception as e:
                logger.warning(f"Read error {txt_path}: {e}")
                texts.append('[UNK]')

        try:
            feats_mean, feats_layer4, feats_seq = extract_batch(  # 改：解包三个
                texts, tokenizer, model, device)
        except Exception as e:
            logger.error(f"Extraction failed at batch {batch_start}: {e}")
            errors += len(batch)
            continue

        for i, (_, rel_stem) in enumerate(batch):
            for out_root, feat in [
                (args.out_mean,   feats_mean[i]),
                (args.out_layer4, feats_layer4[i]),
                (args.out_seq,    feats_seq[i]),    # 新增，shape (T_token_i, 1024)
            ]:
                npy_path = os.path.join(out_root, rel_stem + '.npy')
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                np.save(npy_path, feat.astype(np.float32))

    logger.info(f"Extraction complete. Errors: {errors}")
    logger.info(f"mean   features → {args.out_mean}")
    logger.info(f"layer4 features → {args.out_layer4}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript_root', default=TRANSCRIPT_ROOT)
    parser.add_argument('--macbert_path',    default=MACBERT_PATH)
    parser.add_argument('--out_mean',        default=OUT_MEAN)
    parser.add_argument('--out_layer4',      default=OUT_LAYER4)
    parser.add_argument('--out_seq',         default=OUT_LAYER4.replace('layer4', 'seq'))  # 新增
    parser.add_argument('--batch_size',      type=int, default=32)
    parser.add_argument('--device',          default='cuda:5')
    args = parser.parse_args()
    main(args)
