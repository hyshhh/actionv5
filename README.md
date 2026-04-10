# 水上行为识别系统

基于 YOLO + 多模态大模型的实时行为分析系统

## 环境准备

```bash
pip install -r requirements.txt
```

## 下载开源模型

> 需要查看 [vLLM 官方支持的模型列表](https://docs.vllm.com.cn/en/latest/usage/)

```bash
# 1. Qwen3-VL-4B-Instruct (FP16)
pip install modelscope
modelscope download --model Qwen/Qwen3-VL-4B-Instruct --local_dir /media/ddc/新加卷/hys/qmy/qwen

# 2. Qwen3-VL-4B-Instruct-AWQ-4bit (INT4 量化版)
python -c "
from modelscope import snapshot_download
snapshot_download('cyankiwi/Qwen3-VL-4B-Instruct-AWQ-4bit', local_dir='/media/ddc/新加卷/hys/hysnew/Qwen3-VL-4B-Instruct-AWQ-4bit')
"

# 3. Qwen3.5-2B (FP16)
modelscope download --model Qwen/Qwen3.5-2B --local_dir /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B

# 4. Qwen3.5-2B-AWQ-4bit (INT4 量化版)
modelscope download --model cyankiwi/Qwen3.5-2B-AWQ-4bit --local_dir /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ

# 5. Qwen3.5-0.8B (轻量版)
modelscope download --model unsloth/Qwen3.5-0.8B --local_dir /media/ddc/新加卷/hys/hysnew/Qwen3.5-0.8B
```

## 启动 vLLM 推理服务

> 注意 `--api-key` 和 `--served-model-name` 要和 `config.yaml` 中保持一致

```bash
# 1. Qwen3-VL-4B-Instruct (FP16)
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/qmy/qwen \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-Instruct \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25

# 2. Qwen3-VL-4B-Instruct-AWQ-4bit (INT4)
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3-VL-4B-Instruct-AWQ-4bit \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25

# 3. Qwen3.5-2B (FP16)
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen/Qwen3.5-2B \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.25

# 4. Qwen3.5-2B-AWQ-4bit (INT4)
CUDA_VISIBLE_DEVICES=1 vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-2B-AWQ \
  --api-key abc123 \
  --served-model-name Qwen/Qwen3-VL-4B-AWQ \
  --max-model-len 1024 \
  --port 7890 \
  --gpu-memory-utilization 0.15 \
  --max-num-seqs 10

# 5. Qwen3.5-0.8B (轻量版)
CUDA_VISIBLE_DEVICES=1 VLLM_USE_MODELSCOPE=true vllm serve /media/ddc/新加卷/hys/hysnew/Qwen3.5-0.8B \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 1024 \
  --trust-remote-code
```

## 运行

```bash
python main.py --config config.yaml
```

## 配置参数详解

### 模型运行模式

```yaml
model_mode: "local"    # "api" = 阿里云百炼千问 | "local" = 本地 vLLM
```

### 检测器 (detector)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model` | `yolo11n.pt` | YOLO 模型路径，首次运行自动下载 |
| `confidence` | `0.1` | 检测置信度阈值，低于此值的框直接丢弃 |
| `device` | `cuda:0` | 推理设备：`cpu` / `cuda:0` |
| `class_ids` | `[0, 1, 2]` | 要检测的类别 ID 列表，全部当作"人"处理 |
| `detect_width` | `640` | 推理宽度，`0` = 保持原始分辨率 |
| `detect_height` | `640` | 推理高度，`0` = 保持原始分辨率 |
| `nms_iou` | `0.1` | NMS IoU 阈值，控制合并重叠框的严格程度 |

**`class_ids` 说明：**
- 支持单个值（如 `0`）或列表（如 `[0, 1, 2]`）
- 列表中所有类别统一当作"人"处理，合并输出

### 跟踪器 (tracker)

```yaml
tracker:
  enabled: false               # true = 启用跟踪 | false = 仅检测（无ID）
  tracker_type: "bytetrack"    # "bytetrack" | "botsort"
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `track_high_thresh` | `0.6` | 高分框门槛 |
| `track_low_thresh` | `0.3` | 低分框门槛（ByteTrack 二次匹配） |
| `match_thresh` | `0.8` | IoU 匹配阈值 |
| `track_buffer` | `30` | 跟踪丢失后保留帧数 |
| `with_reid` | `false` | BoT-SORT 是否启用 ReID 外观特征 |

### 关键帧提取 (frame_extractor)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `padding_ratio` | `0.15` | 裁剪 padding 倍率 |
| `keyframe_interval` | `1` | 每隔几帧提取一帧 |
| `keyframe_count` | `1` | 关键帧数量上限 |
| `min_region_size` | `32` | 最小有效区域像素 |
| `adaptive_padding` | `true` | 小目标自动放大 padding |
| `pixel_threshold` | `20000` | 最小裁剪面积阈值（像素²） |

### 行为类别

| ID | 中文 | 英文 | 严重等级 |
|----|------|------|---------|
| 0 | 溺水 | drowning | critical |
| 1 | 游泳 | swimming | normal |
| 2 | 攀爬栏杆 | climbing | warning |
| 3 | 正常行走 | normal_walking | normal |
| 4 | 正在救援 | waterhelping | normal |
| 5 | 在船上 | abord | normal |

### 流水线 (pipeline)

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `process_every_n_frames` | `1` | 每 N 帧触发一次推理 |
| `max_concurrent` | `5` | 最大并发 API 请求数 |
| `alert_cooldown` | `5` | 同一行为告警冷却（秒） |
| `display` | `true` | 是否显示实时画面 |
| `sustained_detection_frames` | `1` | 连续 N 帧检测到目标才触发 |

## 项目结构

```
.
├── main.py                    # 入口
├── config.yaml                # 全局配置
├── core/
│   ├── detector.py            # YOLO 人体检测 + ByteTrack 跟踪
│   ├── pipeline.py            # 推理流水线
│   ├── behavior_classifier.py # 行为分类器
│   ├── frame_extractor.py     # 关键帧提取
│   └── video_source.py        # 视频源管理
├── models/
│   └── schemas.py             # 数据模型定义
├── utils/
│   ├── logger.py              # 日志
│   └── image_utils.py         # 图像工具
├── finetune_yolo.py           # YOLO 微调脚本
├── tracker_guide.md           # 跟踪器调参指南
└── requirements.txt           # 依赖
```

## 微调 YOLO（可选）

```bash
python finetune_yolo.py --data dataset.yaml --pretrained yolov8n.pt --epochs 100
```

微调完成后在 `config.yaml` 中更新模型路径：
```yaml
detector:
  model: "runs/finetune/finetune_xxxx/weights/best.pt"
```
