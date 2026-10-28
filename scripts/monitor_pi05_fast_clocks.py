"""Extend pi05-fast's monitor with CPU clock samples and GPU temperature."""
import os
from pathlib import Path
import sys
FAST=Path('/home/haojiema/code/amd_robot/models/pi05-pytorch-fast')
sys.path.insert(0,str(FAST/'scripts'))
import monitor_util as monitor
import psutil
sys.stdout.reconfigure(line_buffering=True)
original=monitor._gpu_sample
psutil.cpu_percent(None,percpu=True)
def enriched(dev):
    result=original(dev)
    util=psutil.cpu_percent(None,percpu=True)
    frequencies=[]
    for i in range(len(util)):
        base=Path(f'/sys/devices/system/cpu/cpu{i}/cpufreq')
        try:freq=int((base/'scaling_cur_freq').read_text())/1000
        except (OSError,ValueError):freq=0
        result[f'cpu{i}_mhz']=freq
        result[f'cpu{i}_busy_pct']=util[i]
        frequencies.append(freq)
    busy=sum(util)
    result['cpu_busy_weighted_mhz']=sum(f*u for f,u in zip(frequencies,util))/busy if busy else 0
    result['cpu_max_mhz']=max(frequencies,default=0)
    result['cpu_min_mhz']=min(frequencies,default=0)
    for hm in (dev/'hwmon').glob('hwmon*'):
        for key,file,scale in [('gpu_temp_c','temp1_input',1000),('gpu_hwmon_mhz','freq1_input',1000000)]:
            try:result[key]=int((hm/file).read_text())/scale
            except (OSError,ValueError):pass
    result['gpu_high']=int((dev/'power_dpm_force_performance_level').read_text().strip()=='high')
    return result
monitor._gpu_sample=enriched
monitor.main()
