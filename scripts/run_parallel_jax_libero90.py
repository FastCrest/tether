"""Two resource-limited JAX workers with staggered initialization and aggregation."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/data/tether')
OUT=ROOT/'evaluation/libero90-jax'
REPO=Path(__file__).resolve().parents[1]
PYTHON='/home/haojiema/miniconda3/envs/rocm-nightly/bin/python'


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main():
    import psutil
    OUT.mkdir(parents=True,exist_ok=True)
    state={'status':'waiting_for_gpu_slot','pid':os.getpid(),'started':time.time(),
           'total_tasks':90,'total_episodes':450,'requested_workers':2,'workers':[]}
    processes=[]
    def save():
        state['system_available_gib']=psutil.virtual_memory().available/2**30
        state['episodes']=sum(w.get('progress',{}).get('episodes',0) for w in state['workers'])
        state['successes']=sum(w.get('progress',{}).get('successes',0) for w in state['workers'])
        (OUT/'status.json').write_text(json.dumps(state,indent=2))
    def start(index):
        cpu_list=sorted(os.sched_getaffinity(0))[index*8:(index+1)*8]
        directory=OUT/f'worker-{index}'
        directory.mkdir(exist_ok=True)
        env=os.environ.copy()
        env.update(TETHER_JAX_CPUS=','.join(map(str,cpu_list)),OMP_NUM_THREADS='4',
            OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            TF_NUM_INTRAOP_THREADS='4',TF_NUM_INTEROP_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
        command=[PYTHON,'-u',str(REPO/'scripts/evaluate_pi05_jax_rocm.py'),'--worker',
                 '--tasks',','.join(map(str,range(index,90,2))),'--output',str(directory)]
        log=ROOT/'logs'/f'libero90-jax-worker-{index}.log'
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        processes.append(process)
        state['workers'].append({'index':index,'pid':process.pid,'cpus':cpu_list,
                                 'tasks':45,'log':str(log),'directory':str(directory)})
        print(f'Worker {index} PID={process.pid}, CPUs={cpu_list}',flush=True)
        save()
    try:
        with (ROOT/'evaluation_gpu.lock').open('a') as lock:
            save();fcntl.flock(lock,fcntl.LOCK_EX)
            state['status']='initializing';start(0)
            while True:
                progress=read(OUT/'worker-0/status.json')
                state['workers'][0]['progress']=progress;save()
                if processes[0].poll() is not None:
                    if processes[0].returncode:raise RuntimeError('Worker 0 initialization failed')
                    break
                if progress.get('status')=='evaluating':break
                time.sleep(3)
            # Loading a second checkpoint temporarily requires host and device
            # copies. Retain at least 45 GiB headroom before starting it.
            while psutil.virtual_memory().available<45*2**30 and processes[0].poll() is None:
                state['status']='worker_1_waiting_for_memory'
                state['workers'][0]['progress']=read(OUT/'worker-0/status.json');save();time.sleep(5)
            start(1);state['status']='evaluating'
            while True:
                for worker,process in zip(state['workers'],processes):
                    worker['progress']=read(Path(worker['directory'])/'status.json')
                    try:worker['rss_gib']=psutil.Process(process.pid).memory_info().rss/2**30
                    except psutil.NoSuchProcess:pass
                    if process.poll() is not None and process.returncode:
                        raise RuntimeError(f'Worker {worker["index"]} failed: {process.returncode}')
                save()
                if all(p.poll() is not None for p in processes):break
                time.sleep(5)
            results=[]
            for task in range(90):
                path=OUT/f'worker-{task%2}'/f'libero_90-task-{task}.json'
                result=read(path)
                if result.get('task_idx')!=task or [ep['ep'] for ep in result.get('episodes',[])]!=list(range(5)):
                    raise RuntimeError(f'Missing or malformed evidence for task {task}')
                results.append(result);shutil.copy2(path,OUT/path.name)
            successes=sum(int(ep['success']) for r in results for ep in r['episodes'])
            state.update(status='complete',episodes=450,successes=successes,
                         success_rate_pct=successes/450*100,ended=time.time());save()
            (OUT/'summary.json').write_text(json.dumps(state,indent=2))
            (OUT/'report.md').write_text(f'# Native JAX LIBERO-90\n\n| Model | Success | Rate |\n|---|---:|---:|\n| JAX teacher | {successes}/450 | {successes/450*100:.1f}% |\n| PyTorch teacher | 343/450 | 76.2% |\n\nSame source checkpoint; seed 7001; 90 tasks × 5 initial states; 10 denoising steps. Two JAX workers, 8 CPU cores each, no GPU preallocation. This compares the local implementations; it does not reproduce the author’s seed-7 protocol.\n')
            print('COMPLETE',successes,'/450',flush=True)
    except BaseException as exc:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill()
        state.update(status='failed',error=str(exc));save();raise


if __name__=='__main__':main()
