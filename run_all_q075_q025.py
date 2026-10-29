"""Resumable, isolated dual-policy estimation. Never changes historical results."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import csv
import hashlib
import json
import shutil
import sqlite3
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from auto_pair_benchmark import match_candidates, estimate_from_matches
from benchmark_tracking import grayscale, list_images
from run_all_unlabeled_sequences import discover_unlabeled, calibration_for

ROOT=Path(__file__).resolve().parent
OLD=ROOT/'all_patterns_step1_temporal_v2_20260831'
OUT=ROOT/'all_patterns_q075_q025_step1_20261027'
FRAMES=Path(r'E:\2607波崎\260722波崎\箱カメラ\時刻入りコピー\frames')
PARAMS=dict(fps=240.,step=1,max_gap=4,radius=32,candidates=80,low_speed_pixels=4.,
            minimum_confidence=.85,new_positive_quantile=.75,old_positive_quantile=.85,
            negative_quantile=.25,temporal=dict(window=30,zero_abs=3.,ref_motion=20.,rescue_conf=.75,
                rescue_abs=10.,min_count=25,dominance=.75))
WORK=None
DB=None
PAIR_MEMORY={}


class PauseRequested(Exception):
    pass


def pause_requested():
    path=OUT/'control.json'
    return path.exists() and json.loads(path.read_text(encoding='utf-8')).get('action')=='pause'


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def state(status,**extra):
    data=dict(status=status,updated_at=datetime.now().astimezone().isoformat(timespec='seconds'),**extra)
    target=OUT/'progress.json'
    temporary=OUT/'progress.tmp'
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(target)
    print(json.dumps(data,ensure_ascii=False),flush=True)


def connect(readonly=False):
    path=OUT/'checkpoint.sqlite'
    if readonly:
        return sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True,timeout=60)
    db=sqlite3.connect(path,timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=NORMAL')
    db.execute('CREATE TABLE IF NOT EXISTS results (dataset TEXT, start INTEGER, record TEXT NOT NULL, PRIMARY KEY(dataset,start))')
    db.execute('CREATE TABLE IF NOT EXISTS pairs (dataset TEXT,before_file TEXT,after_file TEXT,payload BLOB NOT NULL,PRIMARY KEY(dataset,before_file,after_file))')
    return db


def prepare():
    OUT.mkdir(exist_ok=True)
    snapshots=OUT/'baseline_archive'; snapshots.mkdir(exist_ok=True)
    files=['all_patterns_velocity_step1_temporal_v2.csv','sequence_summary_step1_temporal_v2.csv','run_metadata.json','calibration_map.json']
    for name in files:
        source=OLD/name; target=snapshots/name
        if not target.exists(): shutil.copy2(source,target)
        assert digest(source)==digest(target),('Historical source changed',name)
    manual=ROOT/'manual_series_combined_20260831'/'manual_measurements_combined.csv'
    if not (snapshots/'manual_measurements_combined.csv').exists(): shutil.copy2(manual,snapshots/manual.name)
    sources=[Path(__file__),ROOT/'auto_pair_benchmark.py',ROOT/'benchmark_tracking.py',
        ROOT/'manual_series_combined_20260831'/'evaluate_temporal_filter.py']
    signature=dict(parameters=PARAMS,source_sha256={f.name:digest(f) for f in sources})
    config=OUT/'configuration.json'
    if config.exists(): assert json.loads(config.read_text(encoding='utf-8'))==signature,'Code or parameters changed: choose a new run directory.'
    else: config.write_text(json.dumps(signature,ensure_ascii=False,indent=2),encoding='utf-8')
    calibration=json.loads((snapshots/'calibration_map.json').read_text(encoding='utf-8'))
    discovered=discover_unlabeled(FRAMES,include_manual=True)
    inventory={}
    for folder,images in discovered:
        if folder.name in calibration:
            scale=calibration[folder.name]['cm_per_pixel']; group=calibration[folder.name]['calibration_group']
        else:
            scale,group=calibration_for(folder.name)
            group='user_approved_late_group_for_GX010214_to_GX010222'
        inventory[folder.name]=dict(folder=str(folder),files=[p.name for p in images],cm_per_pixel=scale,calibration_group=group)
    manifest=OUT/'input_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text(encoding='utf-8'))==inventory,'Input file inventory changed since this run started.'
    else: manifest.write_text(json.dumps(inventory,ensure_ascii=False),encoding='utf-8')
    pd.DataFrame([dict(dataset=d,frames=len(v['files']),evaluation_times=len(v['files'])-1,
        cm_per_pixel=v['cm_per_pixel'],calibration_group=v['calibration_group']) for d,v in inventory.items()]).to_csv(OUT/'inventory.csv',index=False,encoding='utf-8-sig')
    db=connect()
    if db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==0:
        seeds=[ROOT/'all_manual_quantile_sensitivity_20261027'/'matches_manual_pairs.jsonl',
               ROOT/'GX010202_quantile_sensitivity_20261008'/'matches_manual_pairs.jsonl']
        for path in seeds:
            for line in path.read_text(encoding='utf-8').splitlines():
                item=json.loads(line); dataset=item.get('dataset','GX010202_timestamp_JST')
                blob=zlib.compress(json.dumps(item['matches'],separators=(',',':')).encode(),level=1)
                db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?)',(dataset,item['before_file'],item['after_file'],blob))
        db.commit()
    completed=set(db.execute('SELECT dataset,start FROM results'))
    db.close()
    tasks=[(d,i) for d,v in inventory.items() for i in range(len(v['files'])-1) if (d,i) not in completed]
    return inventory,tasks,len(completed)


def init_worker():
    global WORK,DB
    WORK=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    DB=connect(readonly=True)


def matches_for(dataset,index,gap):
    info=WORK[dataset]
    before,after=info['files'][index],info['files'][index+gap]
    key=(dataset,before,after)
    if key in PAIR_MEMORY: return PAIR_MEMORY[key],None
    found=DB.execute('SELECT payload FROM pairs WHERE dataset=? AND before_file=? AND after_file=?',key).fetchone()
    blob=None
    if found:
        matches=json.loads(zlib.decompress(found[0]))
    else:
        # An I/O failure is not interpreted as a physical zero or a low-confidence measurement.
        for attempt in range(3):
            try:
                a=grayscale(Path(info['folder'])/before)
                b=grayscale(Path(info['folder'])/after)
                matches=match_candidates(a,b,radius=PARAMS['radius'],candidate_count=PARAMS['candidates'])
                break
            except OSError:
                grayscale.cache_clear()
                if attempt==2: raise
                time.sleep(.5*(attempt+1))
        blob=zlib.compress(json.dumps(matches,separators=(',',':')).encode(),level=1)
    PAIR_MEMORY[key]=matches
    while len(PAIR_MEMORY)>12: PAIR_MEMORY.pop(next(iter(PAIR_MEMORY)))
    return matches,None if blob is None else (*key,blob)


def needs_more(result):
    return result is None or abs(result['dx'])<4 or result['confidence']<.65


def select(candidates):
    first=candidates[0][1]
    direction=None if first is None or abs(first['dx'])<.5 else (1 if first['dx']>0 else -1)
    consistent=[(gap,r) for gap,r in candidates if r is not None and
                (direction is None or abs(r['dx'])<.5 or (1 if r['dx']>0 else -1)==direction)]
    valid=[(gap,r) for gap,r in consistent if r['confidence']>=.85]
    pool=valid or consistent or [(gap,r) for gap,r in candidates if r is not None]
    if not pool: return 1,None,'unmeasurable'
    def score(item):
        gap,r=item
        return r['confidence']+.15*min(abs(r['dx'])/4,1)+(.02*gap if abs(r['dx'])<4 else 0)
    gap,result=max(pool,key=score)
    return gap,result,'measured' if result['confidence']>=.85 else 'unmeasurable'


def record_for(dataset,index,choice,positive):
    info=WORK[dataset]; gap,result,status=choice
    velocity=None if result is None else result['dx']*info['cm_per_pixel']*240/gap
    return dict(dataset=dataset,before_frame=index+1,after_frame=index+gap+1,
        before_file=info['files'][index],after_file=info['files'][index+gap],
        time_s=index/240,delta_time_s=gap/240,sample_step_frames=1,frame_gap=gap,
        cm_per_pixel=info['cm_per_pixel'],calibration_group=info['calibration_group'],
        displacement_px=None if result is None else result['dx'],
        velocity_cm_s=velocity if status=='measured' else None,raw_velocity_cm_s=velocity,status=status,
        support=0 if result is None else result['support'],trusted_matches=0 if result is None else result['trusted_matches'],
        support_fraction=0 if result is None else result['support_fraction'],
        direction_consistency=0 if result is None else result['direction_consistency'],
        spread_px=None if result is None else result['spread_px'],confidence=0 if result is None else result['confidence'],
        confidence_label='unmeasurable' if result is None else result['confidence_label'],
        positive_quantile=positive,negative_quantile=.25)


def process(task):
    dataset,index=task
    pairs=[]
    matches,cache=matches_for(dataset,index,1)
    if cache: pairs.append(cache)
    candidates={q:[(1,estimate_from_matches(matches,positive_quantile=q,negative_quantile=.25))] for q in (.75,.85)}
    longer={q:needs_more(items[0][1]) for q,items in candidates.items()}
    if any(longer.values()):
        available=min(4,len(WORK[dataset]['files'])-index-1)
        for gap in range(2,available+1):
            matches,cache=matches_for(dataset,index,gap)
            if cache: pairs.append(cache)
            for q in candidates:
                if longer[q]: candidates[q].append((gap,estimate_from_matches(matches,positive_quantile=q,negative_quantile=.25)))
    result=dict(new=record_for(dataset,index,select(candidates[.75]),.75),
                old_reproduced=record_for(dataset,index,select(candidates[.85]),.85))
    return dataset,index,result,pairs


def estimate(inventory,tasks,completed,workers,limit):
    total=sum(len(v['files'])-1 for v in inventory.values())
    if limit:
        # Stratify a timing pilot across videos and early/central/late portions.
        desired=[]
        for fraction in (.05,.5,.9):
            for d,v in inventory.items(): desired.append((d,int((len(v['files'])-2)*fraction)))
        available=set(tasks)
        desired=[t for t in desired if t in available]
        tasks=desired[:limit]
    if not tasks: return True
    db=connect()
    start=time.monotonic(); done=0
    state('running',completed=completed,total=total,workers=workers,pilot=bool(limit))
    pool=ProcessPoolExecutor(max_workers=workers,initializer=init_worker)
    pending={}; it=iter(tasks); pausing=False
    def submit():
        task=next(it,None)
        if task is not None: pending[pool.submit(process,task)]=task
    try:
        for _ in range(workers*2): submit()
        while pending:
            if pause_requested() and not pausing:
                pausing=True
                state('pausing',completed=completed+done,total=total,pending=len(pending),
                    message='No more submissions; finishing and saving in-flight tasks.')
            finished,_=wait(pending,return_when=FIRST_COMPLETED,timeout=30)
            if not finished:
                state('pausing' if pausing else 'running',completed=completed+done,total=total,elapsed_s=round(time.monotonic()-start,1),workers=workers)
            for future in finished:
                task=pending.pop(future)
                dataset,index,result,pairs=future.result()
                for cache in pairs: db.execute('INSERT OR IGNORE INTO pairs VALUES (?,?,?,?)',cache)
                db.execute('INSERT INTO results VALUES (?,?,?)',(dataset,index,json.dumps(result,separators=(',',':'))))
                db.commit()
                done+=1
                if done==1 or done%25==0 or done==len(tasks):
                    elapsed=time.monotonic()-start
                    state('pausing' if pausing else 'running',completed=completed+done,total=total,run_completed=done,
                        elapsed_s=round(elapsed,1),pairs_per_second=round(done/elapsed,4),
                        eta_hours=round((total-completed-done)*elapsed/done/3600,2),workers=workers,
                        last_dataset=dataset)
                if pause_requested(): pausing=True
                if not pausing: submit()
    except BaseException as error:
        state('failed',completed=completed+done,total=total,error=repr(error),last_task=pending.get(future) if 'future' in locals() else None)
        for future in pending: future.cancel()
        pool.shutdown(wait=True,cancel_futures=True)
        raise
    else: pool.shutdown(wait=True)
    finally: db.close()
    elapsed=time.monotonic()-start
    if pausing:
        state('paused',completed=completed+done,total=total,elapsed_s=round(elapsed,1),
            message='All in-flight results saved; resume explicitly to continue.')
        return False
    if limit:
        state('pilot_complete',completed=completed+done,total=total,pilot_pairs=done,elapsed_s=round(elapsed,1),
            pairs_per_second=round(done/elapsed,4),eta_hours=round((total-completed-done)*elapsed/done/3600,2))
    return True


def finalize():
    from compare_all_q075_q025 import main as compare
    compare()


def test_selector():
    from unittest.mock import patch
    from run_auto_sequence import choose_adaptive_result
    def sample(dx,c): return dict(dx=dx,confidence=c)
    examples=[[sample(20,.9)],[sample(-20,.9)],[sample(0,.95),sample(2,.9),sample(5,.9),sample(7,.87)],
              [None,None,None,None],[sample(-.1,.6),sample(-3,.9),sample(10,.91),sample(-8,.91)],
              [sample(5,.6),sample(-7,.9),sample(10,.8),None]]
    for candidates in examples:
        with patch('run_auto_sequence.grayscale',lambda p:p),patch('run_auto_sequence.estimate_pair',side_effect=candidates):
            expected=choose_adaptive_result(list(range(5)),0,4,32,4,.85)
        actual=select(list(enumerate(candidates,1)))
        assert actual==expected,(actual,expected)
    print('Adaptive selection parity tests passed.',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--finalize-only',action='store_true')
    parser.add_argument('--test',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Explicitly clear a requested pause and continue saved work.')
    args=parser.parse_args()
    if args.test: test_selector(); return
    if args.resume:
        OUT.mkdir(exist_ok=True)
        (OUT/'control.json').write_text(json.dumps(dict(action='run',requested_at=datetime.now().astimezone().isoformat())),encoding='utf-8')
    elif pause_requested():
        state('paused',message='Pause request retained. Use --resume only after explicit user instruction.'); return
    if args.finalize_only: finalize(); return
    inventory,tasks,completed=prepare()
    print(f'Datasets={len(inventory)} total={sum(len(v["files"])-1 for v in inventory.values())} completed={completed} pending={len(tasks)}',flush=True)
    if args.prepare_only: return
    complete=estimate(inventory,tasks,completed,args.workers,args.limit)
    if complete and not args.limit:
        if pause_requested(): state('paused',stage='before_finalization')
        else: finalize()


if __name__=='__main__': main()
