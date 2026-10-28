# pi0.5 ROCm 蒸馏运行与观察

本实验使用 `rocm-nightly` 的 ROCm 10.1 PyTorch，通过
`/data/tether/rocm-venv` 隔离 LeRobot/Transformers 依赖，不替换原 conda GPU 栈。

```bash
cd /home/haojiema/code/amd_robot/models/tether
conda activate rocm-nightly
source scripts/pi05_distill_rocm_env.sh
python -c 'import torch, transformers; print(torch.__version__, transformers.__version__)'
```

输入：原始未蒸馏 `lerobot/pi05_libero_finetuned_v044`，LeRobot LIBERO v3 数据。
教师、数据、tokenizer 共 8.78 GiB，全部位于 `/data/tether`；版本见
`download_manifest.json`。Tokenizer 使用公开的逐文件哈希一致副本，原始
Google 版本对照证据位于 `tokenizer_source_identity.json`。

LIBERO-90 是额外评估集，未加入原来的 40 任务蒸馏训练。
官方原始数据：`yifengzhu-hf/LIBERO-datasets`，固定 revision
`f13aa24a3da8c43c7225569f28c562979fa0e35a`，90 个任务 HDF5，约62.08 GiB。
目录：`/data/tether/datasets/LIBERO-datasets/libero_90`。
下载与哈希/HDF5验证清单：`/data/tether/libero90_download_manifest.json`。
真实仿真评估使用已固定的 LIBERO 代码、BDDL 和初态文件，不直接读取演示HDF5。

## 查看当前阶段

```bash
cat /data/tether/delivery_pipeline.json
cat /data/tether/restart_status.json
tail -f /data/tether/logs/training_10k.log
tail -f /data/tether/runs/pi05-snapflow-10k/training_log.jsonl
tail -f /data/tether/runs/pi05-snapflow-10k/telemetry.jsonl
```

`delivery_pipeline.json` 标记当前阶段、子进程 PID、退出码和日志路径。
判断进程是否还在运行，请同时使用 `ps -p <pid> -o pid,etime,args`，
不要只看状态文件。正式训练开始前，训练日志路径尚不存在。

## 查看 loss 曲线

TensorBoard：<http://127.0.0.1:6006/>，选择 `pi05-snapflow-10k`。
若浏览器不在这台主机，通过 SSH 转发 6006 端口后打开本地地址。
曲线包含 loss、flow_matching、consistency、gradient_norm、步耗时和内存。

静态图每 100 步及保存 checkpoint 时更新：

- `/data/tether/runs/pi05-snapflow-10k/loss.png`
- `/data/tether/runs/pi05-snapflow-10k/loss.svg`
- `/data/tether/runs/pi05-snapflow-10k/training_summary.json`

随时手动刷新：

```bash
python scripts/plot_pi05_distill.py /data/tether/runs/pi05-snapflow-10k
```

## 训练与评估协议

10,000 steps，batch=4，bf16，LR=1e-4，seed=42，consistency_alpha=1。
保留 default SnapFlow pi0.5 架构：VLM prefix KV 被截断梯度，实际训练
action expert 和 time/action projections；显式冻结 VLM 以减少无效图构建。
使用 PyAV 读取真实 AV1 帧和 50-step action chunks。
在 5000 和 10000 步保存模型，不声称现有权重 checkpoint 能精确续训。

流程按序执行：student 单次 rollout smoke → 1/10-step latency smoke →
正式训练 → 同机 latency benchmark → teacher 和 student LIBERO rollout → 英文报告。
新增阶段在原四组评估之后：student 的完整 LIBERO-90 rollout → 英文报告。
这些阶段串行使用 GPU，减少相互干扰。任何阶段失败，流程停在该阶段，保留日志。

评估覆盖 spatial/object/goal/10 四个 suite 的全部 40 个任务，
每任务 5 个相同初态、相同随机 seed 的 episode，合计每模型 200 次。
测量实际成功率、按任务配对差值和置信区间；不把训练 loss 作为成功率。
这是教师训练域内的仿真评估，不声称覆盖新任务或实机。

另单独评估学生的 LIBERO-90：实际遍历90个任务，每任务5个初态，
共450 episodes，单独报告成功数、成功率、按任务统计及95%任务/episode
bootstrap区间。这450次不混入原40任务的200次学生评估。
本次没有LIBERO-90教师对照，不能据该单独分数归因蒸馏造成的泛化变化。

耗时对比包含真实输入、完整 VLM prefix、action expert；验证实际 denoiser
调用次数分别为 1 和 10。GPU 同步后测量 policy 以及预处理/推理/后处理路径，
不包含仿真渲染、视频文件解码和网络传输。正式测量：batch=1，10 次 warmup，
每模型 50 次样本，报告 mean/p50/p95；不使用训练同时进行的测量作为最终结果。

## 交付物

完成后：

- `/data/tether/runs/pi05-snapflow-10k/training/checkpoints/00010000/pretrained_model`
- `/data/tether/evaluation/latency/latency.json`
- `/data/tether/evaluation/teacher/` 与 `student/` 每个任务的原始结果
- `/data/tether/evaluation/student-libero90/`：90任务、450个episode的学生结果
- `/data/tether/evaluation/student-libero90/summary.json`：单独的LIBERO-90成功率及置信区间
- `/data/tether/delivery/report.md`：英文领导汇报
- `/data/tether/delivery/metrics.json`、`loss.png`、`environment.txt`
- `/data/tether/pi05-distillation-report.tar.gz`：报告及精简附件

报告生成器要求训练确实完成 10,000 步、两个模型都完成 200 次同协议评估。
还要求学生完成LIBERO-90全部90个任务/450 episodes，官方数据下载验证完成。
在这些证据齐全前，最终报告不会生成。当前阶段请以实际日志和进程为准。
