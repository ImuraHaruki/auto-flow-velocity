"""Request a graceful pause, or inspect the durable run state."""
import argparse
import json
from datetime import datetime
from run_all_q075_q025 import OUT


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['pause','status'])
    args=parser.parse_args()
    if args.action=='pause':
        OUT.mkdir(exist_ok=True)
        target=OUT/'control.json'; temporary=OUT/'control.tmp'
        temporary.write_text(json.dumps(dict(action='pause',requested_at=datetime.now().astimezone().isoformat()),ensure_ascii=False),encoding='utf-8')
        temporary.replace(target)
        print('Pause requested. Wait until progress.json says paused before disconnecting storage.')
    progress=OUT/'progress.json'
    if progress.exists(): print(progress.read_text(encoding='utf-8'))


if __name__=='__main__': main()
