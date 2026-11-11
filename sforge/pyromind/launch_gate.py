"""Do not exec a GPU-allocating Controller until its PID is durably recorded."""
import json
import os
from pathlib import Path
import sys
import time

from sforge.harness.agent_config import read_private_file


def main():
    config=json.loads(read_private_file(Path(sys.argv[1])))
    ack=Path(config['ack_path'])
    while time.time()<config['expires_at']:
        if ack.exists():
            document=json.loads(read_private_file(ack))
            if document != {'nonce':config['nonce']}:
                raise ValueError('Launch acknowledgement does not match its recorded nonce')
            os.execv(config['command'][0],config['command'])
        time.sleep(.1)
    raise SystemExit('Launch gate expired before durable admission; no Controller was started')


if __name__=='__main__':
    main()
