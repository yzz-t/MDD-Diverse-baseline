"""
extract_features.py
===================
离线特征提取，支持四种特征类型，输出 (T, D) 帧序列 .npy。
不做任何池化，完整保留时序信息，供后续 LSTM / SVM 使用。

输出目录结构镜像音频目录：
  features/<type>/root/<id>/<seg>.npy  → shape (T, D)

特征维度：
  logmel    : (T, n_mels=80)
  wav2vec2  : (T', 768)   下采样约 20ms/帧
  hubert    : (T', 768)   下采样约 20ms/帧
  emotion2vec: (T', 768)

运行示例：
  python extract_features.py --type logmel    --gpu 0
  python extract_features.py --type wav2vec2  --gpu 0 --batch_size 8
  python extract_features.py --type hubert    --gpu 1
  python extract_features.py --type emotion2vec --gpu 0
"""

import os
import wave
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────────────────────
# 路径配置
# ─────────────────────────────────────────────────────────────────────────────
AUDIO_ROOT   = 'your_audio_root_path'
FEAT_ROOT    = 'your_save_root'
TRAIN_TXT    = 'your train split txt'
VAL_TXT      = 'your val split txt'
SAMPLE_RATE  = 16000
MAX_FRAMES   = 48000   

MODEL_PATHS  = {
    'wav2vec2'    : 'model pt path',
    'hubert'      : 'model pt path',
    'emotion2vec' : {
        'model_dir' : 'model dir',
        'checkpoint': 'model pt path',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# 日志
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def pool_features(feat: np.ndarray, pool_type: str) -> np.ndarray:
    """
    将 (T, D) 的帧级别特征池化为固定维度的句子级别特征。
    """
    if pool_type == 'mean':
        # 输出形状 (D,)
        return feat.mean(axis=0).astype(np.float32)
    elif pool_type == 'stat':
        # 输出形状 (3D,)
        mean_v = feat.mean(axis=0)
        std_v  = feat.std(axis=0) if feat.shape[0] > 1 else np.zeros_like(mean_v)
        max_v  = feat.max(axis=0)
        return np.concatenate([mean_v, std_v, max_v], axis=0).astype(np.float32)
    else:
        # 默认不池化 (frame 级别)
        return feat
# ─────────────────────────────────────────────────────────────────────────────
# 音频加载
# ─────────────────────────────────────────────────────────────────────────────
def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """加载音频，返回 float32 单声道波形 (T,)，统一采样率。"""
    try:
        with wave.open(path, 'rb') as wf:
            orig_sr = wf.getframerate()
            n       = wf.getnframes()
            data    = np.frombuffer(
                wf.readframes(n), dtype=np.int16
            ).astype(np.float32) / 32768.0
        # 双声道取均值
        if wf.getnchannels() == 2:
            data = data.reshape(-1, 2).mean(axis=1)
    except Exception:
        data, orig_sr = sf.read(path, always_2d=False)
        data = data.astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)

    if orig_sr != sr:
        data = librosa.resample(data, orig_sr=orig_sr, target_sr=sr)
    return data


# ─────────────────────────────────────────────────────────────────────────────
# 特征提取函数（每种返回 (T, D) numpy float32）
# ─────────────────────────────────────────────────────────────────────────────
def extract_logmel(wav_path: str, sr: int = SAMPLE_RATE,
                   n_mels: int = 80, hop_length: int = 160,
                   win_length: int = 400) -> np.ndarray:
    """LogMel 帧序列，(T, n_mels)。"""
    y   = load_audio(wav_path, sr)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=hop_length, win_length=win_length,
        n_fft=512,
    )
    feat = np.log(np.maximum(1e-6, mel)).T   # (T, n_mels)
    # CMVN：零均值单位方差
    feat = (feat - feat.mean(axis=0)) / (feat.std(axis=0) + 1e-8)
    return feat.astype(np.float32)


def build_wav2vec2_extractor(model_path: str, device: torch.device):
    """返回 wav2vec2 提取函数（闭包，共用模型）。"""
    from transformers import Wav2Vec2Model, AutoProcessor
    logger.info(f'Loading Wav2Vec2 from {model_path} ...')
    processor = AutoProcessor.from_pretrained(model_path)
    model     = Wav2Vec2Model.from_pretrained(model_path).to(device).eval()
    logger.info('Wav2Vec2 loaded.')

    @torch.no_grad()
    def extract(wav_path: str) -> np.ndarray:
        y      = load_audio(wav_path)
        inputs = processor(
            y, sampling_rate=SAMPLE_RATE, return_tensors='pt'
        ).to(device)
        out  = model(**inputs)
        feat = out.last_hidden_state.squeeze(0)   # (T', 768)
        return feat.cpu().numpy().astype(np.float32)

    return extract


def build_hubert_extractor(model_path: str, device: torch.device):
    from transformers import HubertModel, Wav2Vec2FeatureExtractor  
    logger.info(f'Loading HuBERT from {model_path} ...')
  
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = HubertModel.from_pretrained(model_path).to(device).eval()
    logger.info('HuBERT loaded.')

    @torch.no_grad()
    def extract(wav_path: str) -> np.ndarray:
        y      = load_audio(wav_path)
        inputs = feature_extractor(
            y,
            sampling_rate    = SAMPLE_RATE,
            return_tensors   = 'pt',
            padding          = True,
        ).to(device)
        out  = model(**inputs)
        feat = out.last_hidden_state.squeeze(0)   # (T', 1024)
        return feat.cpu().numpy().astype(np.float32)

    return extract

from dataclasses import dataclass
import fairseq

@dataclass
class UserDirModule:
    user_dir: str

def load_emotion2vec(model_path, checkpoint_path):
    model_path = UserDirModule(model_path)
    fairseq.utils.import_user_module(model_path)
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = models[0]
    return model, task
def load_finetuned_using_pretrained_config(model_path, checkpoint_path):
    
    
    model_path = UserDirModule(model_path)
    fairseq.utils.import_user_module(model_path)
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(['/home/zyy/.cache/huggingface/hub/emotion2vec_base/emotion2vec_base.pt'])
    model = models[0]
    
    # 4. 加载你自己微调后的权重 (state_dict)
    finetuned_state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    # 5. 处理可能存在的 Key 不匹配问题 (例如 DataParallel 产生的 module. 前缀)
    
    new_state_dict = {}
    for k, v in finetuned_state_dict.items():
        # 移除 'module.' 前缀
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
     
    try:
        model.load_state_dict(new_state_dict, strict=True)
        print("Success: Fine-tuned weights loaded into Pre-trained structure.")
    except RuntimeError as e:
        print("Error loading state_dict. Keys might not match.")
        print(e)
        raise e

    model.eval()
    return model, task

def build_emotion2vec_extractor(model_dir: str, checkpoint: str,
                                device: torch.device):
    """返回 Emotion2Vec 提取函数（闭包，共用模型）。"""
    
    if checkpoint == '/huggingface/hub/emotion2vec_base/emotion2vec_base.pt':
        logger.info(f'Loading Emotion2Vec from {checkpoint} ...')
        model,task = load_emotion2vec(model_dir, checkpoint)
    else:

        model, task = load_finetuned_using_pretrained_config(model_dir, checkpoint)
    logger.info('Loading Emotion2Vec ...')
    
    model = model.to(device).eval()
    logger.info('Emotion2Vec loaded.')

    @torch.no_grad()
    def extract(wav_path: str) -> np.ndarray:
        y = load_audio(wav_path)
        t = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)
        # # 截断/补零到 MAX_FRAMES
        # T = t.shape[1]
        # if T > MAX_FRAMES:
        #     t = t[:, :MAX_FRAMES]
        # elif T < MAX_FRAMES:
        #     t = torch.cat(
        #         [t, torch.zeros(1, MAX_FRAMES - T, device=device)], dim=1
        #     )
        out  = model.extract_features(t, padding_mask=None)
        feat = out['x'].squeeze(0)               # (T', 768)
        return feat.cpu().numpy().astype(np.float32)

    return extract


# ─────────────────────────────────────────────────────────────────────────────
# 收集待提取文件列表
# ─────────────────────────────────────────────────────────────────────────────
def collect_wav_files(audio_root: str, txt_paths: list):
    """从 txt 文件中收集所有不重复的 (abs_path, rel_path) 对。"""
    seen    = set()
    records = []
    for txt_path in txt_paths:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path = line.rsplit(' ', 1)[0]
                if rel_path in seen:
                    continue
                seen.add(rel_path)
                records.append((
                    os.path.join(audio_root, rel_path),
                    rel_path,
                ))
    return records

def pad_or_truncate_features(feat: np.ndarray, feat_type: str) -> np.ndarray:
    """
    将特征 (T, D) 统一截断或补齐到 3秒 对应的帧数。
    """
    # 确定目标帧数
    target_frames = 300 if feat_type == 'logmel' else 150
    
    T, D = feat.shape
    if T > target_frames:
        # 超过3秒，截取前3秒
        return feat[:target_frames, :]
    elif T < target_frames:
        # 不足3秒，末尾补0
        pad_width = target_frames - T
        # np.pad 在第0轴(时间轴)末尾补 pad_width，第1轴(特征轴)不补
        return np.pad(feat, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    return feat
# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────
def main(args):
    # GPU 设置
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    logger.info(f'Device : {device}')
    logger.info(f'Feature: {args.type}')

    out_root = os.path.join(args.feat_root, args.type)
    logger.info(f'Output : {out_root}')

    # 构建提取函数
    if args.type == 'logmel':
        extract_fn = extract_logmel
    elif args.type == 'wav2vec2':
        extract_fn = build_wav2vec2_extractor(
            args.model_path or MODEL_PATHS['wav2vec2'], device
        )
    elif args.type == 'hubert':
        extract_fn = build_hubert_extractor(
            args.model_path or MODEL_PATHS['hubert'], device
        )
    elif args.type == 'emotion2vec':
        cfg = MODEL_PATHS['emotion2vec']
        extract_fn = build_emotion2vec_extractor(
            args.model_path or cfg['model_dir'],
            args.checkpoint  or cfg['checkpoint'],
            device,
        )
    else:
        raise ValueError(f'Unknown feature type: {args.type}')

    # 收集文件
    txt_list = [args.train_txt, args.val_txt]
    records  = collect_wav_files(args.audio_root, txt_list)
    logger.info(f'Total wav files: {len(records)}')

    # 断点续跑：跳过已提取文件
    todo, skip = [], 0
    for abs_path, rel_path in records:
        npy_path = os.path.join(
            out_root, str(Path(rel_path).with_suffix('.npy'))
        )
        if os.path.exists(npy_path):
            skip += 1
        else:
            todo.append((abs_path, rel_path, npy_path))
    logger.info(f'Already done: {skip} | To process: {len(todo)}')

    if not todo:
        logger.info('All files already extracted.')
        return

    # 提取
    errors = 0
    for abs_path, rel_path, npy_path in tqdm(todo, desc=f'[{args.type}]', ncols=80):
        try:
            feat = extract_fn(abs_path)          # (T, D) float32
            # ========== 新增：池化操作 ==========
            
            feat = pad_or_truncate_features(feat, args.type)
            feat = pool_features(feat, args.pool_type)

            os.makedirs(os.path.dirname(npy_path), exist_ok=True)
            np.save(npy_path, feat)
        except Exception as e:
            logger.warning(f'Error {rel_path}: {e}')
            errors += 1

    logger.info(f'Done. Errors: {errors}')
    logger.info(f'Features saved to: {out_root}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',        required=True,
                        choices=['logmel', 'wav2vec2', 'hubert', 'emotion2vec'])
    parser.add_argument('--gpu',         type=int, default=0,
                        help='GPU id，-1 表示 CPU')
    parser.add_argument('--audio_root',  default=AUDIO_ROOT)
    parser.add_argument('--feat_root',   default=FEAT_ROOT)
    parser.add_argument('--train_txt',   default=TRAIN_TXT)
    parser.add_argument('--val_txt',     default=VAL_TXT)
    parser.add_argument('--model_path',  default=None,
                        help='覆盖默认模型路径（wav2vec2/hubert/emotion2vec model_dir）')
    parser.add_argument('--checkpoint',  default=None,
                        help='emotion2vec checkpoint 路径')
    
    parser.add_argument('--pool_type',   default='mean', choices=['mean', 'stat','frame'],
                        help='如果是 utterance 级别，使用哪种池化方式')
    args = parser.parse_args()
    main(args)
