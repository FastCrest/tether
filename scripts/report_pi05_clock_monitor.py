"""Summarize steady inference frequency telemetry and render charts."""
import csv
import json
from pathlib import Path
import statistics
ROOT=Path('/data/tether/evaluation/pi05-fast-high-clock-monitor')
summary={};data={}
for arm in ['teacher','student']:
 rows=list(csv.DictReader((ROOT/f'{arm}-clocks.csv').open()));steady=[r for r in rows if r['phase']=='steady'];start=float(steady[0]['t']);rows=[r for r in steady if float(r['t'])>=start+2]
 data[arm]=(start,rows)
 stats={}
 for key in ['gfxclk','cpu_busy_weighted_mhz','cpu_max_mhz','gpu_busy_pct','gpu_temp_c','socket_power','gfx_power','proc_cpu_pct']:
  vals=[float(r[key]) for r in rows]
  stats[key]={'min':min(vals),'mean':statistics.mean(vals),'max':max(vals)}
 stats['ppt_85_plus_minus_1w_fraction']=sum(abs(float(r['socket_power'])-85)<1 for r in rows)/len(rows)
 d=json.loads((ROOT/f'{arm}.json').read_text());stats['latency']={k:d[k] for k in ['mean_ms','p50_ms','n']};stats['monitor_samples']=len(rows);stats['steady_seconds']=float(rows[-1]['t'])-float(rows[0]['t']);summary[arm]=stats
(ROOT/'clock_summary.json').write_text(json.dumps(summary,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(3,2,figsize=(12,8),sharex='col')
for col,arm in enumerate(['teacher','student']):
 start,rows=data[arm];t=[float(r['t'])-start for r in rows]
 axes[0,col].plot(t,[float(r['gfxclk'])/1000 for r in rows],label='GPU actual (SMU)',color='#2276bb')
 axes[0,col].plot(t,[float(r['cpu_busy_weighted_mhz'])/1000 for r in rows],label='CPU load-weighted snapshot',color='#d97829',alpha=.8)
 axes[0,col].axhline(2.9,color='gray',ls='--',lw=1,label='GPU HIGH DPM: 2.9 GHz')
 axes[0,col].set_title(f'{arm.capitalize()} | P50 {summary[arm]["latency"]["p50_ms"]:.2f} ms (n=400)')
 axes[0,col].set_ylabel('Clock (GHz)');axes[0,col].legend(fontsize=8)
 axes[1,col].plot(t,[float(r['socket_power']) for r in rows],color='#349966',label='Socket / PPT')
 axes[1,col].axhline(85,color='gray',ls='--',lw=1);axes[1,col].set_ylabel('Power (W)');axes[1,col].legend(fontsize=8)
 axes[2,col].plot(t,[float(r['gpu_temp_c']) for r in rows],color='#c75151',label='GPU edge temperature')
 axes[2,col].set_ylabel('Temperature (C)');axes[2,col].set_xlabel('Seconds since timed loop start');axes[2,col].legend(fontsize=8)
 for row in range(3):axes[row,col].grid(alpha=.2)
fig.suptitle('CPU performance / GPU HIGH: steady inference telemetry (first 2s excluded)')
fig.tight_layout();fig.savefig(ROOT/'inference-clocks.png',dpi=160);plt.close(fig)
lines=['# HIGH 档 Pi0.5 推理与频率监控','','GPU 始终 HIGH；所有 CPU governor 为 performance，系统电源策略 performance。每模型 400 次推理，每 200 ms 采样；剔除正式计时窗口开始后 2 秒。','', '| 模型 | Mean / P50 | GPU 实际频率平均（范围） | CPU 忙核加权平均 | PPT 平均 | GPU 温度平均 / 最高 |','|---|---:|---:|---:|---:|---:|']
for arm,s in summary.items():
 l=s['latency'];g=s['gfxclk'];c=s['cpu_busy_weighted_mhz'];p=s['socket_power'];temp=s['gpu_temp_c']
 lines.append(f'| {arm} | {l["mean_ms"]:.2f} / {l["p50_ms"]:.2f} ms | {g["mean"]:.0f} ({g["min"]:.0f}–{g["max"]:.0f}) MHz | {c["mean"]:.0f} MHz | {p["mean"]:.1f} W | {temp["mean"]:.1f} / {temp["max"]:.0f} °C |')
lines+=['','GPU DPM 档位显示 2900 MHz，但实际 SMU/ hwmon 时钟低于此值。CPU scaling_max_freq 仍为 5187.5 MHz，忙核快照大多显著低于上限。HIGH/performance 不等于锁住实际最高频率。', '', 'Teacher/Student 正式采样点分别 94.8% / 90.0% 落在 PPT 85±1 W；GPU busy 平均 96.5% / 96.9%。这更符合持续负载下的功耗预算约束，但未读取到 SMU 上限或明确 throttling 标志，不能排除温控及固件策略。','', '此前 README 的性能监控示例约 140 W socket、2900 MHz GPU，当前持续窗口约 85 W，运行条件明显不同。该历史示例不是已证实的 109 ms 当次硬件记录。','', 'HIGH 无监控短测 n=50：Teacher P50 188.90 ms、Student 117.65 ms。此次有监控长测 n=400：Teacher 190.69 ms、Student 125.12 ms。两者样本长度、温度和监控开销不同，不能仅凭差值定量归因于降频。', '', '配置为 fast 库 W4A4 prefix LM、FP16 prefix、BF16 vision/expert、fullgraph max-autotune-no-cudagraphs；语言200/有效52，空相机缓存关闭；Teacher10步、Student1步。未更改功耗上限、CPU频率上下限或GPU手动锁频。','', '采样限制：CPU 为 cpufreq 瞬时频率按每核利用率加权，不是硬件周期计数；GPU SMU 为滤波数据，200 ms 采样不能识别每个 kernel 的频率波动。', '', '![推理频率、功耗和温度](inference-clocks.png)','', '原始数据：teacher-clocks.csv、student-clocks.csv；汇总 clock_summary.json；延迟样本 teacher.json、student.json。', '', '复跑：`/data/tether/rocm-venv/bin/python scripts/run_pi05_clock_monitor_latency.py`。']
(ROOT/'report.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({a:s['latency'] for a,s in summary.items()},indent=2))
