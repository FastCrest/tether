"""Wait for successful JAX evaluation and free evaluation slot; test precision."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data/tether')
REPO = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evaluation/rocm-scripts-precision-latency'


def jax_running():
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if any(Path(x.decode(errors='replace')).name in
               {'run_parallel_jax_libero90.py', 'evaluate_pi05_jax_rocm.py'}
               for x in argv if x):
            return True
    return False


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    state = {'status': 'waiting_for_jax', 'pid': os.getpid(), 'started': time.time()}
    def save():
        (OUT / 'status.json').write_text(json.dumps(state, indent=2))
    save()
    try:
        while True:
            try:
                jax = json.loads((ROOT / 'evaluation/libero90-jax/status.json').read_text())
            except (OSError, ValueError):
                jax = {}
            state['jax_episodes'] = jax.get('episodes')
            save()
            if jax.get('status') == 'failed':
                raise RuntimeError('JAX failed; latency not started')
            if jax.get('status') == 'complete' and not jax_running():
                break
            time.sleep(15)
        with (ROOT / 'evaluation_gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if jax_running():
                raise RuntimeError('New JAX evaluation appeared; latency not started')
            results = []
            for name, arm, precision in [('teacher-native', 'teacher', 'native'),
                                         ('teacher-vision-bf16', 'teacher', 'bf16'),
                                         ('student', 'student', 'native')]:
                state.update(status='benchmarking', arm=name)
                save()
                cmd = [str(ROOT / 'rocm-venv/bin/python'),
                       str(REPO / 'scripts/benchmark_rocm_scripts_pi05.py'),
                       '--arm', arm, '--vision-precision', precision,
                       '--json', str(OUT / f'{name}.json')]
                with (OUT / f'{name}.log').open('w') as stream:
                    subprocess.run(cmd, cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT, check=True)
                result = json.loads((OUT / f'{name}.json').read_text())
                result['test_name'] = 'openpi_pi05_' + name.replace('-', '_')
                results.append(result)
            (OUT / 'benchmarks.json').write_text(json.dumps({'results': results}, indent=2))
            lines = ['# Teacher 视觉精度延迟对照', '',
                     '| 配置 | Mean ms | P50 ms | 样本数 |',
                     '|---|---:|---:|---:|']
            for result in results:
                lines.append(f'| {result["test_name"]} | {result["latency_mean_ms"]:.2f} | {result["latency_p50_ms"]:.2f} | {result["timed_iterations"]} |')
            native, vision, student = results
            lines += ['', f'仅修改 Teacher 视觉路径为 BF16，P50 减少 {native["latency_p50_ms"]-vision["latency_p50_ms"]:.2f} ms。',
                      '', '同一 Teacher 权重、输入与 10 步采样；语言模型和动作专家精度保持不变。视觉 BF16 对照不等同于完整复现 OpenPI 的精度规则。',
                      '', 'max-autotune / GPU Graph / batch 1 / 50×7 动作 / 每配置至少 50 样本。',
                      'JAX 成功完成并退出后启动；未暂停 JAX。延迟结论不替代 BF16 模型成功率验证。']
            (OUT / 'report.md').write_text('\n'.join(lines) + '\n')
            state.update(status='complete', ended=time.time())
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
