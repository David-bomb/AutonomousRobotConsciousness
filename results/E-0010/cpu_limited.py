"""Последовательные прогоны E-0010 под ограничением ЦП Windows."""
import argparse
from collections import Counter, deque
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil
import win32event
import win32job
import win32process


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MONITOR = HERE / "cpu"
JOB_RATE = 200  # 2% общего процессорного времени, единица Windows равна 0,01%.
INTERVAL = 0.1
PAUSE_AT = 58
RESUME_BELOW = 48


@contextmanager
def single_runner():
    MONITOR.mkdir(exist_ok=True)
    with (MONITOR / "runner.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise RuntimeError("Уже работает ограниченный исполнитель E-0010") from error
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def limited_job():
    job = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(job, win32job.JobObjectExtendedLimitInformation)
    basic = info["BasicLimitInformation"]
    basic["LimitFlags"] = (win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS | win32job.JOB_OBJECT_LIMIT_AFFINITY
                           | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)
    basic["ActiveProcessLimit"] = 1
    basic["Affinity"] = 1 << psutil.Process().cpu_affinity()[-1]
    win32job.SetInformationJobObject(job, win32job.JobObjectExtendedLimitInformation, info)
    # pywin32 не объявляет JobObjectCpuRateControlInformation (15).
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = wintypes.HANDLE(int(job))
    rate = (wintypes.DWORD * 2)(0x1 | 0x4, JOB_RATE)  # ENABLE | HARD_CAP
    if not kernel.SetInformationJobObject(handle, 15, ctypes.byref(rate), ctypes.sizeof(rate)):
        raise ctypes.WinError(ctypes.get_last_error())
    actual = (wintypes.DWORD * 2)()
    if not kernel.QueryInformationJobObject(handle, 15, ctypes.byref(actual), ctypes.sizeof(actual), None):
        raise ctypes.WinError(ctypes.get_last_error())
    if tuple(actual) != tuple(rate):
        raise RuntimeError("Windows не подтвердила жёсткую квоту ЦП")
    return job


def run_task(name, arguments, log, cool_until):
    env = os.environ.copy()
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    command = subprocess.list2cmdline([sys.executable, "-B", "-X", "utf8", *arguments])
    job = limited_job()
    startup = win32process.STARTUPINFO()
    with Path(log.name).with_suffix(".out.log").open("ab") as output, open(os.devnull, "rb") as stdin:
        for stream in (output, stdin):
            os.set_inheritable(stream.fileno(), True)
        startup.dwFlags = 0x100  # STARTF_USESTDHANDLES
        startup.hStdInput = msvcrt.get_osfhandle(stdin.fileno())
        startup.hStdOutput = startup.hStdError = msvcrt.get_osfhandle(output.fileno())
        process, thread, pid, _ = win32process.CreateProcess(
            None, command, None, None, True,
            win32process.CREATE_SUSPENDED | win32process.CREATE_NO_WINDOW | subprocess.IDLE_PRIORITY_CLASS,
            env, str(ROOT), startup)
    def record(**values):
        log.write(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), task=name, **values))+"\n")
        log.flush()
    try:
        win32job.AssignProcessToJobObject(job, process)
        if not win32job.IsProcessInJob(process, job):
            raise RuntimeError("Исполнитель не включён в группу с квотой ЦП")
        worker = psutil.Process(pid)
        record(event="start", pid=pid, cpu_quota=JOB_RATE/100, affinity=worker.cpu_affinity(), priority=worker.nice(),
               total_pause=PAUSE_AT, total_resume=RESUME_BELOW)
        psutil.cpu_percent()
        running = started = False
        history = deque(maxlen=15)
        def worker_seconds():
            cpu = worker.cpu_times()
            return cpu.user+cpu.system
        last_cpu = worker_seconds()
        sampled_at = time.monotonic()
        while win32event.WaitForSingleObject(process, round(INTERVAL*1000)) == win32event.WAIT_TIMEOUT:
            total = psutil.cpu_percent()
            was_running = running
            now = time.monotonic()
            try:
                cpu = worker_seconds()
            except psutil.NoSuchProcess:
                break
            contribution = 100*(cpu-last_cpu)/((now-sampled_at)*psutil.cpu_count())
            last_cpu,sampled_at=cpu,now
            history.append(total)
            if total >= 80:
                cool_until=now+30
            quiet=(len(history)==history.maxlen and max(history)<PAUSE_AT
                   and sum(history)/len(history)<=RESUME_BELOW and total<=RESUME_BELOW)
            try:
                if running and total >= PAUSE_AT:
                    worker.suspend()
                    running = False
                elif not running and quiet and now>=cool_until:
                    if started:
                        worker.resume()
                    else:
                        win32process.ResumeThread(thread)
                        started = True
                    running = True
            except psutil.NoSuchProcess:
                break
            record(total=total, worker_cpu=round(contribution,2), running_during_sample=was_running, running_next=running)
        code = win32process.GetExitCodeProcess(process)
        record(event="finish", code=code)
        if code:
            raise RuntimeError(f"Исполнитель {name} завершился с кодом {code}")
        return cool_until
    finally:
        if win32event.WaitForSingleObject(process, 0) == win32event.WAIT_TIMEOUT:
            win32process.TerminateProcess(process, 1)
        thread.Close()
        process.Close()
        job.Close()


def pending_commands(path):
    completed = set()
    while True:
        plan = json.loads(path.read_text(encoding="utf8"))
        if not isinstance(plan,list) or not plan or not all(
                isinstance(cmd,list) and cmd and all(isinstance(arg,str) for arg in cmd) for cmd in plan):
            raise ValueError("Очередь должна содержать непустые списки аргументов Python")
        occurrences = Counter()
        for command in plan:
            args = tuple(command)
            occurrences[args] += 1
            key = args, occurrences[args]
            if key not in completed:
                yield command
                completed.add(key)
                break
        else:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents", nargs="*")
    parser.add_argument("--run", nargs=argparse.REMAINDER, help="одна команда Python без дочерних процессов")
    parser.add_argument("--queue", type=Path, help="JSON со списком последовательных команд Python")
    args = parser.parse_args()
    if sum(map(bool, (args.agents, args.run, args.queue))) != 1:
        parser.error("Укажите имена автоматов, --run или --queue")
    with single_runner():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f")
        with (MONITOR/f"{stamp}.jsonl").open("x", encoding="utf8") as log:
            cool_until = time.monotonic()+30
            if args.run:
                run_task(Path(args.run[0]).name, args.run, log, cool_until)
            if args.queue:
                for cmd in pending_commands(args.queue):
                    cool_until = run_task(" ".join(cmd), cmd, log, cool_until)
            for name in args.agents:
                if not name.replace("-", "").isalnum():
                    parser.error(f"Некорректное имя автомата {name}")
                trace = HERE/f"{name}.jsonl"
                command = [str(HERE/"bench.py"), name, str(trace), "--workers", "1"]
                if trace.exists():
                    command.append("--resume")
                cool_until = run_task(name, command, log, cool_until)
                print(name, "серия завершена", flush=True)


if __name__ == "__main__":
    main()
