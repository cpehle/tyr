#!/usr/bin/env python3
"""Launch native Lean analytical TP autograd tests; Python performs no model math."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ports():
    for _ in range(100):
        with socket.socket() as first, socket.socket() as second:
            first.bind(('127.0.0.1', 0))
            port = first.getsockname()[1]
            if port >= 65535:
                continue
            try:
                second.bind(('127.0.0.1', port + 1))
            except OSError:
                continue
            return port
    raise RuntimeError('cannot reserve consecutive rendezvous ports')


def gpu_processes():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True, timeout=10)
    return [{'pid': int(row[0]), 'gpu_uuid': row[1], 'memory_mib': row[2]}
            for line in result.stdout.splitlines() if line.strip()
            for row in [[field.strip() for field in line.split(',')]]]


def run_pair(args, root, gpus):
    root.mkdir()
    port = ports()
    processes, logs, commands = [], [], []
    observed = {}
    report = {'status': 'failed', 'backend': args.backend, 'gpus': gpus,
              'rendezvous_ports': [port, port + 1]}
    try:
        if gpus:
            report['gpu_processes_before'] = gpu_processes()
        for rank in range(2):
            output = root / f'rank-{rank}.json'
            command = [str(args.binary), args.backend, '127.0.0.1', str(port), str(rank), str(output)]
            commands.append(command)
            env = dict(os.environ, OMP_NUM_THREADS='1', CUDA_VISIBLE_DEVICES=gpus[rank] if gpus else '',
                       TYR_TP_TEST_GATE=str(root))
            log = (root / f'rank-{rank}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
        deadline = time.monotonic() + args.timeout
        released = False
        while any(p.poll() is None for p in processes):
            failed = [p.returncode for p in processes if p.poll() is not None and p.returncode != 0]
            assert not failed, f'native rank failed: {failed}'
            assert time.monotonic() < deadline, 'native rank deadline'
            if gpus:
                live = {}
                for row in gpu_processes():
                    live.setdefault(row['pid'], []).append(row)
                for rank, process in enumerate(processes):
                    if process.pid in live:
                        assert {row['gpu_uuid'] for row in live[process.pid]} == {gpus[rank]}, 'rank used wrong GPU'
                        observed[rank] = live[process.pid][0]
            if not released and all((root / f'rank-{rank}.ready.json').exists() for rank in range(2)):
                for rank in range(2):
                    ready = json.loads((root / f'rank-{rank}.ready.json').read_text())
                    assert ready == {'rank': rank, 'ready': True}
                if not gpus or set(observed) == {0, 1}:
                    (root / 'release').write_text('both native ranks ready; GPU attribution verified when required\n')
                    released = True
            time.sleep(.025)
        assert all(p.returncode == 0 for p in processes), 'native rank failure'
        assert released, 'native ranks exited without readiness gate'
        results = [json.loads((root / f'rank-{rank}.json').read_text()) for rank in range(2)]
        for rank, result in enumerate(results):
            assert result['status'] == 'passed' and result['rank'] == rank and result['backend'] == args.backend
            assert len(result['checks']) == 12
        if gpus:
            assert set(observed) == {0, 1}, 'missing actual rank GPU attribution'
        report.update(status='passed', results=results, observed_rank_gpus=observed)
    except Exception as error:
        report['failure'] = repr(error)
        raise
    finally:
        errors = []
        for process in processes:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=10)
            except Exception as error:
                errors.append(repr(error))
        for log in logs:
            log.close()
        report.update(commands=commands, exit_codes=[p.returncode for p in processes],
                      process_ids=[p.pid for p in processes], cleanup_errors=errors)
        if errors:
            report['status'] = 'failed'
        (root / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
        if errors:
            raise RuntimeError('native fixture cleanup failed: ' + repr(errors))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, default=Path('.lake/build/bin/tp_autograd_tests'))
    parser.add_argument('--backend', choices=['gloo', 'nccl'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+')
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.output = args.output.resolve()
    assert args.timeout > 0
    if args.backend == 'nccl':
        assert args.gpus and len(args.gpus) in (2, 4) and len(set(args.gpus)) == len(args.gpus)
        pairs = [args.gpus[i:i+2] for i in range(0, len(args.gpus), 2)]
    else:
        assert not args.gpus, 'Gloo test uses CPU tensors'
        pairs = [None]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = ['Tyr/Distributed.lean', 'cc/src/tyr_distributed.cpp',
               'Tests/TensorParallelAutograd.lean', 'scripts/test-tp-autograd.py', 'lakefile.lean']
    report = {'status': 'failed', 'backend': args.backend, 'pairs': [],
              'binary': str(args.binary), 'binary_sha256': sha(args.binary),
              'source_sha256': {name: sha(name) for name in sources}}
    try:
        for index, gpus in enumerate(pairs):
            report['pairs'].append(run_pair(args, args.output / f'pair-{index}', gpus))
            print(f'pair {index}: native TP autograd analytical tests PASS', flush=True)
        report['status'] = 'passed'
    except Exception as error:
        report['failure'] = repr(error)
        raise
    finally:
        (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
