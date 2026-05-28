# SD-reScripts (lora-rescripts)

## 项目概述

SD-reScripts 是基于 [lora-scripts](https://github.com/aaaki/lora-scripts) / SD-Trainer 的增强分支，提供 LoRA 训练的 Web GUI 和一键环境配置。本仓库是为 Arch Linux + uv + CUDA 13 + RTX 5090 本地部署适配的 fork。

- **上游**: https://github.com/WhitecrowAurora/lora-rescripts
- **入口文件**: `gui.py`（FastAPI + uvicorn）
- **默认端口**: 28000（主服务），6006（TensorBoard），28001（Tag Editor）
- **本地环境**: `.local-runtime/venv`（Python 3.13），`run_gui.sh` 自动检测

## 快速启动

```bash
# 激活本地环境
source .local-runtime/venv/bin/activate
# 或 direnv（自动加载 .envrc）

# 启动 GUI
python gui.py
# 或用启动脚本（自动处理环境检测和依赖检查）
bash run_gui.sh
```

## 目录结构

| 目录/文件 | 说明 |
|-----------|------|
| `gui.py` | GUI 入口，FastAPI 服务器启动器 |
| `mikazuki/` | 后端核心逻辑（FastAPI app、训练流程、插件系统、工具集） |
| `mikazuki/app/` | API 路由层（训练、配置、状态、UI） |
| `mikazuki/utils/` | 工具模块（运行时管理、训练辅助、环境检测） |
| `mikazuki/plugins/` | 插件系统 |
| `mikazuki/tagger/` | WD14 标签器 |
| `mikazuki/aesthetic_labeling/` | 美学评分标注 |
| `mikazuki/scripts/` | 运行时自检脚本 |
| `frontend/dist/` | 预构建前端页面（HTML） |
| `config/` | 训练配置 TOML、镜像设置、预设、benchmark |
| `scripts/` | 训练脚本（kohya/sd-scripts 派生） |
| `scripts/stable/` | 稳定版训练脚本（SD, SDXL, SD3, Flux, Anima, Newbie 等） |
| `scripts/dev/` | 开发版训练脚本（来自上游 sd-scripts） |
| `dataset/` | 训练数据集入口（含 `labels.db`） |
| `sd-models/` | 放入 Stable Diffusion 基础模型 |
| `output/` | 训练输出目录（LoRA safetensors） |
| `logs/` | TensorBoard 日志 |
| `.local-runtime/` | **本地部署目录**（git ignored） |
| `.local-runtime/venv/` | Python 虚拟环境（uv 创建） |
| `.local-runtime/docs/` | 本地部署和维护文档 |
| `.local-runtime/cache/` | Triton/TorchInductor 编译缓存 |
| `.local-runtime/wheels/` | 本地编译的 wheel（如 sageattention） |
| `tools/` | 开发和运维辅助工具 |
| `plugin/` | 第三方插件安装目录 |
| `launcher/` | 桌面启动器（Tauri） |

## 训练路线

所有支持的训练类型定义在 `mikazuki/utils/trainer_registry.py` 的 `TRAINER_REGISTRY` 字典中。

| 训练类型 | 描述 | 训练脚本 |
|----------|------|----------|
| `sd-lora` | SD1.5 LoRA | `scripts/stable/train_network.py` |
| `sdxl-lora` | SDXL LoRA | `scripts/stable/sdxl_train_network.py` |
| `sd-dreambooth` | DreamBooth | `scripts/stable/train_db.py` |
| `sd3-lora` | SD3 LoRA | `scripts/dev/sd3_train_network.py` |
| `flux-lora` | Flux LoRA | `scripts/dev/flux_train_network.py` |
| `anima-lora` | Anima LoRA | `scripts/stable/anima_train_network.py` |
| `newbie-lora` | Newbie LoRA | `scripts/stable/newbie_lora_train.py` |
| `hunyuan-image-lora` | 混元图像 LoRA | `scripts/stable/hunyuan_image_train_network.py` |
| `lumina-lora` / `lumina2-lora` | Lumina LoRA | `scripts/stable/lumina_train_network.py` |

## 环境配置

本 fork 的核心改动：使用项目本地 `uv venv`，不污染系统 Python。

### 环境变量

- `VIRTUAL_ENV` → `.local-runtime/venv`（通过 `.envrc` 或 direnv 自动加载）
- `MIKAZUKI_ALLOW_SYSTEM_PYTHON=1` — 允许使用系统 Python（开发调试用）
- `MIKAZUKI_SKIP_REQUIREMENTS_VALIDATION=1` — 跳过依赖检查（`.envrc` 已设）
- `HF_HOME` — HuggingFace 缓存目录
- `TRITON_CACHE_DIR` / `TORCHINDUCTOR_CACHE_DIR` — 编译缓存目录

### 本地部署基线

- Python 3.13.13
- PyTorch 2.12.0+cu130
- SageAttention 2.2.0（从源码编译，RTX 5090 sm120）
- xformers 0.0.35
- 详细部署步骤见 `.local-runtime/docs/archlinux-uv-cuda13-deploy.md`

### Attention 后端选择（config/lora.toml）

```toml
# SageAttention 优先（RTX 5090 推荐）
xformers = false
sdpa = true
sageattn = true

# 全精度 SDPA（最省显存）
xformers = false
sdpa = true
sageattn = false
```

## 配置文件

- `config/default.toml` — 默认训练参数模板
- `config/lora.toml` — LoRA 训练参数模板
- `config/presets/*.toml` — 训练预设
- `config/benchmarks/*.toml` — 速度 benchmark 配置
- `config/china_mirror.json` — 中国大陆镜像设置
- `config/sample_prompts.txt` — 采样提示词模板

## GUI 启动参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `28000` | 监听端口 |
| `--listen` | — | 监听所有接口（0.0.0.0） |
| `--skip-prepare-environment` | — | 跳过环境准备 |
| `--disable-tensorboard` | — | 禁用 TensorBoard |
| `--disable-tageditor` | — | 禁用标签编辑器 |
| `--tensorboard-port` | `6006` | TensorBoard 端口 |

详细参数列表见 `.local-runtime/docs/gui-args.md`。

## 关键训练参数（SDXL LoRA）

详见 `.local-runtime/docs/sdxl-lora-training-config.md`。核心参数速查：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `network_dim` | 32-128 | LoRA 秩，dim 越大表达能力越强 |
| `network_alpha` | ≤ dim | 缩放系数，通常 = dim |
| `unet_lr` | 1e-4 (dim=32) | UNet 学习率，dim 增大时减小 |
| `text_encoder_lr` | 1e-5 (dim=32) | Text Encoder LR，学画风时不能太低 |
| `optimizer_type` | AdamW8bit | 默认推荐 |
| `lr_scheduler` | cosine_with_restarts | 余弦退火+重启 |
| `mixed_precision` | bf16 | RTX 5090 完全支持 |
| `cache_latents` | true | 强烈建议开启 |
| `network_train_unet_only` | true | 学概念时开启，学画风时关闭 |

## 常见操作

```bash
# 训练一次后查看日志
ls logs/
tensorboard --logdir ./logs

# 使用 direnv 自动加载环境
direnv allow

# 查看 nvidia-smi
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## 注意事项

1. **环境隔离**：所有 Python 依赖在 `.local-runtime/venv` 内，删除该目录即可干净重建
2. **模型路径**：基础模型放在 `sd-models/`，训练集放在 `dataset/`（或 `train_data_dir` 指定的路径）
3. **SageAttention**：RTX 5090 需从源码编译，wheel 保存在 `.local-runtime/wheels/`
4. **tag editor**：需要独立运行时（`install_tageditor.sh`），Python 3.13 不兼容时自动回退到主运行时
5. **TensorBoard**：集成在 GUI 内，可通过 `/proxy/tensorboard` 访问
