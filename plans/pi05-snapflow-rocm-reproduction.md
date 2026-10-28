# pi0.5 SnapFlow 在 rocm-nightly 上复现：计划与输入准备

日期：2026-10-26。此阶段仅准备输入、核对代码并生成计划，不启动训练。

## 后续执行更新（以实际进程和日志为准）

最新用户要求：下载缺少的LIBERO-90并重新启动完整流程，单独统计学生90任务成功率。
官方90任务数据已下载并逐文件验证，4500 demonstrations、62.08 GiB，位于
`/data/tether/datasets/LIBERO-datasets/libero_90`。原383步中止记录已归档，
新10k任务从教师权重重新开始，保留原40任务训练输入。新增学生90任务×5次评估，
单独输出summary.json及报告指标，不混入40任务统计。最新相关122项测试通过。

用户已将目标扩展为：完整蒸馏、loss/日志整理、LIBERO 指标、1/10-step
真实耗时对比、英文领导汇报，三项全部验证后才算完成。
最新运行说明见 `docs/pi05-rocm-distillation.md`；下文初始环境状态为准备时快照。

- 教师、数据、tokenizer 均已下载；输入共 8.78 GiB，manifest 已补齐。
- Tokenizer 通过公开副本获取，五个文件与原始 Google 版本的哈希一致，
  原权限问题已解决，无需更换词表。
- 已创建 `/data/tether/rocm-venv`，复用 `rocm-nightly` 的 ROCm 10.1 PyTorch，
  隔离安装 LeRobot 0.6.1、Transformers 5.5.4 和非 torch 依赖。
- 真实 20-step/batch1 和 100-step/batch4 训练均完成；100-step 前/后50步
  平均 loss 4.428 → 2.370，GPU峰值分配37.17 GiB，稳定4.35 s/step。
- 已修复默认 pi05 单步推理的 fp32 timestep/bf16 time MLP 类型不匹配。
  相关117项测试通过；保存后加载与 rollout 继续进行真实验证。
- 正式训练保留 default expert-only 梯度范围，显式冻结原本已截断梯度的VLM。
  计划5000/10000步保存两个checkpoint；实测单个学生checkpoint约7.72 GiB。
- TensorBoard 已启动在127.0.0.1:6006；完整后台流程、日志及最终报告见运行说明。
- 最终评估：40任务×5初态，每模型200 episodes；latency每模型50样本。
  正式结果和英文报告尚未完成，不把短程验证当作最终蒸馏交付。

## 目标与复现边界

以 `lerobot/pi05_libero_finetuned_v044` 为未蒸馏教师，使用项目当前
SnapFlow backend、LeRobot 格式 LIBERO，训练 default（state-in-language）
1-NFE 学生，目标 10,000 steps、batch 4、bf16、consistency_alpha 1.0。
不需要先微调 pi05_base，也不需要下载已蒸馏学生或教师 ONNX。
教师需要完整 PyTorch checkpoint，包括 config、权重和 processor/归一化文件。

原学生公开 provenance 未记录 dataset revision、seed、学习率或代码 commit。
因此这是基于公开实现与输入的复现实验，不能承诺逐位复现原权重或原成功率。
学习率暂用 backend 默认 1e-4，seed=42；必须在短程训练后检查稳定性。

## 当前机器与环境

- `/data` 总容量约 477.57 GiB，准备前可用 338.99 GiB。
- GPU：AMD Radeon 8060S Graphics，PyTorch 报告共享 GPU 内存 120 GiB。
- 系统物理内存约 123 GiB，目前 available 81 GiB；GPU 与 CPU 共享内存，
  不能把 120 GiB 视为独立可用显存。正式运行前重新检查其他负载。
- conda `rocm-nightly`：Python 3.12.14，torch
  `2.15.0a0+rocm10.1.0a20260902`，torch.version.hip=7.16.26351，GPU 可用。
- torchvision 为对应 ROCm nightly，现有 transformers=5.17.0。
- 当前项目需要 transformers=5.5.4、lerobot=0.6.1；环境没有 LeRobot。
- 训练代码用 torch.cuda API，ROCm PyTorch 可以走该路径。
  还没有完成真实模型 forward/backward，不能据此宣布训练已兼容。

## 输入下载与目录

执行（可重复执行；Hugging Face 会复用已完成文件）：

```bash
cd /home/haojiema/code/amd_robot/models/tether
conda activate rocm-nightly
source scripts/pi05_distill_rocm_env.sh
python scripts/download_pi05_distill_assets.py --inputs-only
# 如已拥有 Google 模型的 HF 访问权限，下面补齐 tokenizer。
python scripts/download_pi05_distill_assets.py
```

固定版本和路径：

| 输入 | revision | 路径 | 大小 |
|---|---|---|---|
| 教师完整仓库 | 7152f13e34c6c115d808aea0e5d08aca74eac14b | /data/tether/checkpoints/pi05_libero_finetuned_v044 | 6.960 GiB |
| LIBERO v3.0 | a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4 | /data/tether/lerobot/lerobot/libero | 1.803 GiB |
| PaliGemma tokenizer 文件，不含模型权重 | 35e4f46485b4d07967e7e9935bc3786aad50687c | /data/tether/tokenizers/paligemma-3b-pt-224 | 0.020 GiB |

数据有 1693 episodes、273465 frames、40 tasks、10 fps。
两路 256×256 AV1 视频，state=8、action=7，与教师 processor 字段一致。
训练必须读取真实 50-step action chunk；现有 backend 已设置 delta_timestamps。
无需将视频解码成图像文件，解码缓存应有界。
脚本将下载清单及文件大小验证结果写入 `/data/tether/download_manifest.json`。
实际状态：教师 9 个文件和 LIBERO 457 个文件已下载并验证。
Google tokenizer 在官方源返回 GatedRepoError 401，当前没有取得访问权限；
tokenizer 尚未下载完成，训练准备入口会明确报错。需在具有
`google/paligemma-3b-pt-224` 访问权限的 HF 账户登录后重新执行完整下载脚本。
下载脚本禁用曾返回 401 的 Xet 通道，并显式使用 huggingface.co；
系统默认的 hf-mirror.com 对 tokenizer 返回 403。没有自动登录或修改凭据。

## 环境适配计划（尚未执行安装）

先导出现有包列表，再查看依赖解析结果。保留现有 torch/torchvision，
禁止直接安装 `[gpu]`、`[gpu-min]` 或 NVIDIA CUDA/TensorRT 依赖。
现有 Transformers 不在项目范围，需适配。建议使用同一 conda Python 的
独立 target 目录 `/data/tether/python-deps`，避免影响其他使用 rocm-nightly 的项目：

```bash
python -m pip freeze > /data/tether/logs/rocm-nightly-before.txt
python -m pip install --dry-run 'lerobot==0.6.1' 'transformers==5.5.4'
```

根据 dry-run 输出制定安装约束；torch/torchvision 必须绑定现有 nightly，
不能接受解析器替换它们。具体安装依赖方案需在下一阶段落实并验证，
此文件不把未经解析验证的 pip 全量安装命令当作可直接执行方案。
实际已执行带 ROCm torch/torchvision 约束的 dry-run：失败，LeRobot 0.6.1
声明 `torch>=2.7,<2.12.0`，现有 nightly 是 2.15.0a0。
日志：`/data/tether/logs/dependency-dry-run.log`。
因此除了 Transformers 版本，还需要明确绕过 LeRobot 的 torch 版本上界并验证
运行时兼容性。可在独立 target 目录中以 `--no-deps` 安装 LeRobot，
再逐项解析和补齐非 torch 依赖；这属于下一阶段适配，不能把包安装成功
当作 ROCm nightly 训练兼容已成立。本轮未修改 conda 环境。
项目代码可先以 `PYTHONPATH="$PWD/src:${PYTHONPATH:-}"` 使用。
还需确认 processor 注册、datasets/parquet、AV1 解码器、num2words 等依赖。
默认视频 backend 若依赖不兼容的 torchcodec，评估 PyAV 路径并做真实视频读取。

## 实施与验证顺序

1. 完整下载输入，核对 manifest、文件数和大小；记录项目 commit 与环境包版本。
2. 适配依赖并验证 PI05Policy 和 SnapFlow 扩展导入。用本地 tokenizer 覆盖
   processor 的 tokenizer_name，避免运行时访问 gated Google 模型。
3. 确认 LeRobot 根据 HF_LEROBOT_HOME 读取本地目录；若 LeRobot 检查版本并
   触发远端访问，修正本地路径/版本绑定后再开启离线模式。现有 distill
   dataloader 未传入 cfg.dataset_root/revision，不能直接靠这两个配置固定输入。
4. preflight；真实读取一个 AV1 视频 batch，确认图像/state/action/语言与
   padding/mask，确认 action chunk=(B,50,7) 经处理补到模型所需维度。
5. 20-step、batch=1 训练 smoke：有限 loss/梯度、参数有更新、单步推理可运行，
   记录 RSS/GPU 峰值和每步时间。再以 batch=4 跑 100 steps。
6. 根据 100-step 实测估算 10k 耗时，稳定后正式跑 batch=4、10k。
7. 验证保存/重新加载学生与 1-NFE 推理；用同一 LIBERO 任务、初态和 seeds
   对比 teacher 10-NFE 与 student 1-NFE。成功率必须来自真实 rollout。
8. ONNX/部署另行验证，训练阶段关闭自动 ONNX export；不安装 CUDA ORT。

## 当前代码的训练命令

项目真实入口是 `tether distill --teacher-export`，不要照抄旧模型卡的
`reflex distill --teacher` 或 examples 的 `tether train distill --teacher`。
依赖验证完成后，普通 CLI 的 smoke 命令如下（不是已执行结果）：

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python -m tether.cli distill \
  --teacher-export /data/tether/checkpoints/pi05_libero_finetuned_v044 \
  --dataset lerobot/libero \
  --output /data/tether/runs/pi05-snapflow-smoke \
  --steps 20 --batch-size 1 --learning-rate 1e-4 \
  --precision bf16 --consistency-alpha 1.0 \
  --skip-export --skip-libero-gate --verbose
```

需先用 `python -m tether.cli --help` 验证入口；若模块无 main，安装项目
`python -m pip install --no-deps -e .` 后使用 `tether distill`。
`--skip-libero-gate` 表示该运行没有成功率评测，不能报告性能复现成功。
已提供 `scripts/run_pi05_distill_rocm.py`，设置 CLI 未暴露的
`checkpoint_every`、seed、local tokenizer override；复用 run_finetune，
不复制训练实现。本地教师目录通过软链接复用权重，只更改派生目录里的
processor tokenizer 路径，不改原始输入。普通 CLI 默认每 1000 steps 保存一次全模型。
派生 config 还禁用 compile_model（先验证 eager 路径，避免原教师开启的
max-autotune 编译干扰 SnapFlow 类替换及 ROCm smoke）和 push_to_hub。

推荐在环境适配完成后使用以下指令，依次执行，前一关通过再跑后一关：

```bash
# 不加载模型、不训练；只准备本地 tokenizer 配置和权重软链接。
python scripts/run_pi05_distill_rocm.py --prepare-only --output /data/tether/runs/prepare
# preflight 检查。
python scripts/run_pi05_distill_rocm.py --dry-run --output /data/tether/runs/preflight
# 真实 20-step forward/backward 验证。
python scripts/run_pi05_distill_rocm.py --steps 20 --batch-size 1 \
  --output /data/tether/runs/pi05-snapflow-smoke
# batch=4 短程性能与内存验证。
python scripts/run_pi05_distill_rocm.py --steps 100 --batch-size 4 \
  --output /data/tether/runs/pi05-snapflow-100
# 10k 正式蒸馏，仅末尾保存，使用独立输出目录。
python scripts/run_pi05_distill_rocm.py --steps 10000 --batch-size 4 \
  --checkpoint-every 10000 --learning-rate 1e-4 --seed 42 \
  --output /data/tether/runs/pi05-snapflow-10k \
  > /data/tether/logs/pi05-snapflow-10k.log 2>&1
```

runner 当前尚未经过训练验证；这组命令是后续执行计划。
runner 将 base 设为本地 teacher 路径，供通用 preflight 读取教师特征；
旧 distill CLI 的 base 为空，通用 preflight 可能无法解析教师 config。
所有训练命令跳过自动 export 和 LIBERO gate，评测须另行执行。

## 磁盘预算

| 项目 | 估算新增 GiB |
|---|---:|
| 教师 + LIBERO + tokenizer | 8.783 |
| 学生一个 checkpoint（当前 bf16 保存估算） | 约 7，保守按 14 |
| 默认每 1000 步保存，10k 总计 10 个 checkpoint | 约 70，保守按 140 |
| smoke/短程 checkpoint | 约 7–28，及时决定保留量 |
| 可选 ONNX + 临时导出文件 | 额外预留 20–40 |
| Python 依赖、日志、下载临时文件 | 额外预留 10–20 |

当前 backend 保存模型权重，不保存 AdamW optimizer state，未实现训练恢复；
不要把 checkpoint 当作可精确续训的完整状态。模型 bf16 总体积仍需实测，
部分参数可能保持 fp32。推荐只在末尾保存（checkpoint_every=10000），
或显式设置 5000 保存两个，训练日志持续写出。
不将 snapshot_download(local_dir=...) 的输入再复制一遍到 HF hub cache。
即使保守保留默认 10 个 checkpoint + smoke + 导出 + 依赖，预算约 237 GiB，
小于当前 339 GiB 可用空间；只存最终 checkpoint 的训练预算约 35–65 GiB。
正式训练前再次 df/du，保留至少 30 GiB 余量。

## 完成标准

- 准备阶段：输入完整落盘，manifest 可审计，磁盘空间充足，命令/适配事项清楚。
- 训练阶段：ROCm 真实 forward/backward、稳定短程训练、10k checkpoint 可加载。
- 结果阶段：同协议 LIBERO rollout 对比；不给未测得的成功率或 10×延迟承诺。
