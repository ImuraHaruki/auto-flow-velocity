"""Verify drain/save/resume without touching the real job or external storage."""
import json
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch
import run_all_q075_q025 as runner


def fake_process(task):
    time.sleep(.04)
    return task[0],task[1],dict(saved=True),[]


def main():
    runner.test_selector()
    with tempfile.TemporaryDirectory(prefix='pause_verification_',dir=runner.ROOT) as directory:
        folder=Path(directory)
        inventory={'test':dict(files=[str(i) for i in range(21)])}
        def pause_later():
            time.sleep(.09)
            (folder/'control.json').write_text(json.dumps(dict(action='pause')),encoding='utf-8')
        with patch.object(runner,'OUT',folder),patch.object(runner,'ProcessPoolExecutor',ThreadPoolExecutor),patch.object(runner,'init_worker',lambda:None),patch.object(runner,'process',fake_process):
            stopper=threading.Thread(target=pause_later); stopper.start()
            complete=runner.estimate(inventory,[('test',i) for i in range(20)],0,2,None)
            stopper.join()
            assert complete is False
            assert json.loads((folder/'progress.json').read_text())['status']=='paused'
            db=runner.connect(); keys=set(db.execute('SELECT dataset,start FROM results')); db.close()
            assert 0<len(keys)<20
            (folder/'control.json').write_text(json.dumps(dict(action='run')),encoding='utf-8')
            remaining=[('test',i) for i in range(20) if ('test',i) not in keys]
            assert runner.estimate(inventory,remaining,len(keys),2,None) is True
            db=runner.connect()
            assert db.execute('SELECT COUNT(*) FROM results').fetchone()[0]==20
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            db.close()
    print('PASS: graceful pause drained in-flight tasks, preserved results, and resumed without duplicates.')


if __name__=='__main__': main()
