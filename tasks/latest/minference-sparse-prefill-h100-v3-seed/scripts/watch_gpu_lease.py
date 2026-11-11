#!/usr/bin/env python3
"""Host-side sampled GPU ownership monitor; stops only its named container."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--container', required=True)
p.add_argument('--uuid', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.parent.mkdir(parents=True, exist_ok=True)
with a.output.open('a', buffering=1) as log:
    while True:
        try:
            top = subprocess.check_output(['docker', 'top', a.container, '-eo', 'pid'], text=True)
            owned = {int(x.strip()) for x in top.splitlines()[1:] if x.strip()}
            query = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader'], text=True)
            visible = {int(line.split(',')[1]) for line in query.splitlines()
                       if line.split(',')[0].strip().lower() == a.uuid.lower()}
            foreign = visible - owned
            log.write(json.dumps({'utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                  'uuid': a.uuid, 'own_pids': sorted(owned),
                                  'compute_pids': sorted(visible), 'foreign_pids': sorted(foreign),
                                  'sampled_guard': True}) + '\n')
            if foreign:
                print('Foreign compute client detected; stopping our validation container.', flush=True)
                subprocess.run(['docker', 'stop', '--time', '1', a.container], check=True)
                raise SystemExit(2)
        except subprocess.CalledProcessError:
            break
        time.sleep(2)
