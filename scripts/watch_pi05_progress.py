"""Live training/evaluation progress; standard library only, read-only."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, size=256 * 1024):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            offset = max(0, stream.tell() - size)
            stream.seek(offset)
            data = stream.read().decode('utf-8', errors='replace')
            return data.split('\n', 1)[-1] if offset else data
    except OSError:
        return ''


def alive(pid):
    if not pid:
        return False
    try:
        stat = Path(f'/proc/{pid}/stat').read_text()
        return stat[stat.rfind(')') + 2:].split()[0] != 'Z'
    except OSError:
        return False


def duration(seconds):
    if seconds is None:
        return '待估算'
    minutes = max(0, round(seconds / 60))
    return f'{minutes // 60}h {minutes % 60:02d}m'


def evaluation_progress(directory, log):
    # JSON persists finished tasks; log includes episodes in the active task.
    # Merge by (suite, task, episode) so they cannot be counted twice.
    episodes = {}
    errors = 0
    for path in directory.glob('*-task-*.json'):
        raw = read_json(path)
        errors += len(raw.get('errors', []))
        for task in raw.get('per_task', []):
            for ep in task.get('episodes', []):
                episodes[(raw.get('suite'), task['task_idx'], ep['ep'])] = bool(ep['success'])
    suite = None
    task = None
    for line in log.splitlines():
        match = re.search(r'suite=(\w+)', line)
        if match:
            suite = match[1]
        match = re.search(r'\] TASK (\d+):', line)
        if match:
            task = int(match[1])
        match = re.search(r'ep (\d+): ([✓✗])', line)
        if match and suite is not None and task is not None:
            episodes[(suite, task, int(match[1]))] = match[2] == '✓'
    tasks = {}
    for key in episodes:
        tasks.setdefault(key[:2], set()).add(key[2])
    complete = sum(indices == set(range(5)) for indices in tasks.values())
    return len(episodes), sum(episodes.values()), complete, suite, task, errors


def render(root):
    delivery = read_json(root / 'delivery_pipeline.json')
    interim = read_json(root / 'evaluation/libero90-interim/status.json')
    training = next((p for p in delivery.get('phases', []) if p['name'] == 'training_10k'), {})
    log = tail(root / 'logs/training_10k.log')
    matches = list(re.finditer(r'step (\d+)/(\d+)\s+total=([\d.eE+-]+)', log))
    step, total, loss = (int(matches[-1][1]), int(matches[-1][2]), float(matches[-1][3])) if matches else (0, 10000, None)
    records = []
    for line in tail(root / 'runs/pi05-snapflow-10k/telemetry.jsonl').splitlines():
        try:
            record = json.loads(line)
            if 'step_time_s' in record:
                records.append(record)
        except ValueError:
            pass  # Writer may not have finished its last line yet.
    if records and records[-1]['step'] >= step:
        step, loss = records[-1]['step'], records[-1]['loss']
    speed = sum(r['step_time_s'] for r in records[-100:]) / len(records[-100:]) if records else None
    live = alive(training.get('pid'))
    state = '已完成' if step >= total else ('运行中' if live else '未运行/需检查')
    lines = [f'pi0.5 进度监控  {datetime.now():%Y-%m-%d %H:%M:%S}', f'数据目录：{root}', '',
             f'训练：{state}  {step:,}/{total:,} ({step / total:.1%})  PID={training.get("pid", "—")}',
             f'  loss={loss:.5f}' if loss is not None else '  loss=待记录']
    if speed:
        eta = duration((total - step) * speed) if live else '—'
        lines.append(f'  最近 {min(100, len(records))} 步平均 {speed:.2f}s/step；剩余约 {eta}')
        if step < 5000:
            lines.append(f'  第 5,000 步 checkpoint：约 {duration((5000 - step) * speed) if live else "—"}')
    telemetry = root / 'runs/pi05-snapflow-10k/telemetry.jsonl'
    if telemetry.exists():
        lines.append(f'  遥测更新时间：{time.time() - telemetry.stat().st_mtime:.0f} 秒前')
    lines += ['', f'中期流程：{interim.get("status", "尚未启动")}']
    jobs = [
        ('中期 Teacher LIBERO-90', 'libero90-interim/teacher', 'libero90-interim-teacher.log', 90, interim.get('teacher_pid')),
        ('中期 Student LIBERO-90', 'libero90-interim/student', 'libero90-interim-student.log', 90, interim.get('student_pid')),
        ('最终 Teacher 原始40任务', 'teacher', 'eval_teacher.log', 40, None),
        ('最终 Student 原始40任务', 'student', 'eval_student.log', 40, None),
        ('最终 Student LIBERO-90', 'student-libero90', 'eval_student_libero90.log', 90, None),
    ]
    for name, subdir, logfile, count, pid in jobs:
        phase = next((p for p in delivery.get('phases', []) if Path(p.get('command', [''])[-1]).name == Path(subdir).name and 'eval' in p.get('command', [])), {})
        path = Path(phase['log']) if phase.get('log') and not subdir.startswith('libero90-interim') else root / 'logs' / logfile
        pid = pid or phase.get('pid')
        done, success, tasks, suite, task, errors = evaluation_progress(root / 'evaluation' / subdir, tail(path))
        state = '已完成' if done == count * 5 else ('运行中' if alive(pid) else ('部分结果/未运行' if done else '等待启动'))
        phase_name={'teacher':'teacher_evaluation','student':'student_evaluation',
                    'student-libero90':'student_libero90_evaluation'}.get(subdir)
        if phase_name in delivery.get('cancelled_phases',[]) and done<count*5:
            state='已停止' if done else '已取消'
        lines.append(f'{name}：{state}  任务 {tasks}/{count}；episodes {done}/{count * 5} ({done / (count * 5):.1%})')
        if done:
            lines.append(f'  当前已完成样本成功 {success}/{done} ({success / done:.1%})' + ('【部分结果】' if done < count * 5 else ''))
        if task is not None:
            lines.append(f'  日志最新任务：{suite} #{task}（任务编号从 0 开始）')
        if errors:
            lines.append(f'  错误记录：{errors}')
    lines += ['', f'最终交付流程：{delivery.get("status", "未知")}']
    for phase in delivery.get('phases', []):
        if phase['name'] != 'training_10k':
            lines.append(f'  {phase["name"]}: {phase.get("status", "未知")}')
    report = root / 'evaluation/libero90-interim/report.md'
    lines.append(f'中期报告：{report if report.exists() else "尚未生成"}')
    jax=read_json(root/'evaluation/libero90-jax/status.json')
    if jax:
        done=jax.get('episodes',0);success=jax.get('successes',0)
        lines += ['',f'JAX Teacher LIBERO-90：{jax.get("status", "未知")}；episodes {done}/450；成功 {success}']
        if 'system_available_gib' in jax:
            lines.append(f'  系统可用共享内存：{jax["system_available_gib"]:.1f} GiB')
        for worker in jax.get('workers',[]):
            progress=worker.get('progress',{})
            lines.append(f'  worker {worker["index"]} PID={worker["pid"]}：{progress.get("status", "启动中")}；{progress.get("episodes",0)}/225 episodes；CPU 核心 {len(worker["cpus"])}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data/tether'))
    parser.add_argument('--interval', type=float, default=5, help='刷新间隔（秒），默认 5')
    parser.add_argument('--once', action='store_true', help='打印一次后退出')
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval 必须大于 0')
    try:
        while True:
            if sys.stdout.isatty() and not args.once:
                print('\033[2J\033[H', end='')
            print(render(args.root), flush=True)
            if args.once:
                return
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
