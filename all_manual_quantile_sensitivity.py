"""Sensitivity of the rightward weighted quantile on exact manual image pairs."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auto_pair_benchmark import estimate_from_matches, match_candidates
from benchmark_tracking import DATASETS, grayscale, list_images, image_resolver, trailing_number
from gx010202_quantile_fine_sensitivity import metrics, QS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'all_manual_quantile_sensitivity_20261027'
PIPE=ROOT/'all_patterns_step1_temporal_v2_20260831'


def prepare():
    calibration=json.loads((PIPE/'calibration_map.json').read_text(encoding='utf-8'))
    records=[]
    for dataset in DATASETS:
        if dataset.name=='B':
            continue
        labels=pd.read_csv(dataset.labels)
        images=list_images(dataset.frames)
        resolve=image_resolver(images)
        positions={p.name:i for i,p in enumerate(images)}
        scale=float(calibration[dataset.frames.name]['cm_per_pixel'])
        for index,row in labels.iterrows():
            before=resolve(str(row.before_file),str(row.before_frame))
            after=resolve(str(row.after_file),str(row.after_frame))
            gap=positions[after.name]-positions[before.name]
            assert gap>0 and np.isclose(gap,float(row.delta_time_s)*240), (dataset.name,index,'frame interval mismatch')
            def resolution(raw,resolved):
                if Path(str(raw)).name.casefold()==resolved.name.casefold(): return 'exact_name'
                a,b=trailing_number(str(raw)),trailing_number(resolved.name)
                if a is not None and b is not None and (a==b or a%10000==b%10000): return 'unique_numeric_name'
                return 'frame_index_fallback'
            record=dict(dataset=dataset.frames.name,row=int(index),manual=float(row.velocity),
                before_file=before.name,after_file=after.name,before_path=str(before),after_path=str(after),
                manual_before_file=str(row.before_file),manual_after_file=str(row.after_file),
                before_resolution=resolution(row.before_file,before),after_resolution=resolution(row.after_file,after),
                source_before_frame=positions[before.name]+1,source_after_frame=positions[after.name]+1,
                time_s=positions[before.name]/240,gap=gap,cm_per_pixel=scale,
                manual_csv_path=str(dataset.labels))
            # Do not quietly substitute an unrelated image when names cannot be resolved.
            assert 'frame_index_fallback' not in (record['before_resolution'],record['after_resolution']), record
            records.append(record)
        print(f'PREPARED {dataset.frames.name}: {len(labels)} pairs',flush=True)
    frame=pd.DataFrame(records)
    assert not frame.duplicated(['dataset','row']).any()
    frame.to_csv(OUT/'pair_manifest.csv',index=False,encoding='utf-8-sig')
    return records


def load_cache():
    cache={}
    path=OUT/'matches_manual_pairs.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            key=(item['dataset'],item['row'])
            assert key not in cache
            cache[key]=item
    return cache


def extract(records,limit=None):
    cache=load_cache()
    for r in records:
        c=cache.get((r['dataset'],r['row']))
        if c:
            for field in ('before_file','after_file','manual','gap','cm_per_pixel'):
                assert c[field]==r[field], (r['dataset'],r['row'],field,'cache mismatch')
    todo=[r for r in records if (r['dataset'],r['row']) not in cache]
    if limit is not None: todo=todo[:limit]
    start=time.monotonic()
    print(f'EXTRACT total={len(records)} cached={len(cache)} pending_this_run={len(todo)}',flush=True)
    def work(r):
        a,b=grayscale(Path(r['before_path'])),grayscale(Path(r['after_path']))
        matches=match_candidates(a,b,radius=32,candidate_count=80)
        return r|{'matches':[[float(dx),float(w)] for dx,w in matches]}
    with (OUT/'matches_manual_pairs.jsonl').open('a',encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=4) as pool:
        futures={pool.submit(work,r):r for r in todo}
        for n,future in enumerate(as_completed(futures),1):
            item=future.result()
            cache[(item['dataset'],item['row'])]=item
            handle.write(json.dumps(item,ensure_ascii=False)+'\n'); handle.flush()
            if n==1 or n%10==0 or n==len(todo):
                elapsed=time.monotonic()-start
                eta=elapsed/n*(len(todo)-n)
                state=dict(completed=len(cache),total=len(records),elapsed_s=round(elapsed,1),eta_s=round(eta,1),last_dataset=item['dataset'])
                (OUT/'progress.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
                print(json.dumps(state,ensure_ascii=False),flush=True)
    return cache


def predictions(records,cache):
    output=[]
    for r in records:
        item=cache[(r['dataset'],r['row'])]
        for q in QS:
            result=estimate_from_matches(item['matches'],positive_quantile=q,negative_quantile=.25)
            output.append({k:v for k,v in r.items() if k not in ('before_path','after_path')}|dict(
                quantile=q,predicted_cm_s=np.nan if result is None else result['dx']*r['cm_per_pixel']*240/r['gap'],
                confidence=np.nan if result is None else result['confidence'],trusted_matches=len(item['matches'])))
    additional=pd.DataFrame(output)
    old=pd.read_csv(ROOT/'GX010202_quantile_sensitivity_20261027'/'predictions.csv')
    old['dataset']='GX010202_timestamp_JST'
    pred=pd.concat([additional,old],ignore_index=True)
    assert not pred.duplicated(['dataset','row','quantile']).any()
    assert pred.groupby(['dataset','row']).size().eq(7).all()
    pred['direction']=np.select([pred.manual>10,pred.manual < -10],['right','left'],default='slow')
    pred['accepted_conf085']=pred.confidence>=.85
    common=pred.groupby(['dataset','row']).accepted_conf085.all()
    pred['common_conf085']=[common.loc[(d,r)] for d,r in zip(pred.dataset,pred.row)]
    pred.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    return pred


def summarize(pred):
    rows=[]
    for dataset,group in pred.groupby('dataset'):
        for q,g in group.groupby('quantile'):
            for direction in ['all','right','left','slow']:
                sub=g if direction=='all' else g.loc[g.direction.eq(direction)]
                for mode,selected in [('all_estimates',sub),('confidence_ge_085',sub.loc[sub.accepted_conf085]),('common_confidence_ge_085',sub.loc[sub.common_conf085])]:
                    rows.append(dict(dataset=dataset,quantile=q,direction=direction,evaluation=mode,**metrics(selected,len(sub))))
    summary=pd.DataFrame(rows)
    summary.to_csv(OUT/'per_video_summary.csv',index=False,encoding='utf-8-sig')
    aggregate=[]
    sets={'additional5':[d for d in pred.dataset.unique() if not d.startswith('GX010202')], 'all6':list(pred.dataset.unique())}
    for scope,datasets in sets.items():
        for direction in ['all','right','left','slow']:
            for mode in ['all_estimates','confidence_ge_085','common_confidence_ge_085']:
                for q in QS:
                    s=summary.loc[summary.dataset.isin(datasets)&summary.direction.eq(direction)&summary.evaluation.eq(mode)&summary['quantile'].eq(q)]
                    valid=s.loc[s.n>0]
                    aggregate.append(dict(scope=scope,direction=direction,evaluation=mode,quantile=q,
                        videos=len(valid),n=int(valid.n.sum()),total=int(s.total.sum()),
                        coverage=valid.n.sum()/s.total.sum() if s.total.sum() else np.nan,
                        pooled_mae=(valid.mae*valid.n).sum()/valid.n.sum() if valid.n.sum() else np.nan,
                        pooled_rmse=np.sqrt((valid.rmse**2*valid.n).sum()/valid.n.sum()) if valid.n.sum() else np.nan,
                        macro_mae=valid.mae.mean(),macro_rmse=valid.rmse.mean(),macro_coverage=s.loc[s.total>0,'coverage'].mean()))
    agg=pd.DataFrame(aggregate)
    agg.to_csv(OUT/'aggregate_summary.csv',index=False,encoding='utf-8-sig')
    # Leave one entire video out. The training score gives each video equal weight.
    cv=[]
    for direction in ['right','all']:
        s=summary.loc[summary.direction.eq(direction)&summary.evaluation.eq('all_estimates')]
        for held in sorted(s.dataset.unique()):
            training=s.loc[s.dataset.ne(held)].groupby('quantile').mae.mean()
            chosen=float(training.idxmin())
            test=s.loc[s.dataset.eq(held)]
            cv.append(dict(direction=direction,held_out=held,selected_quantile=chosen,
                test_mae=float(test.loc[test['quantile'].eq(chosen),'mae'].iloc[0]),
                baseline75_mae=float(test.loc[test['quantile'].eq(.75),'mae'].iloc[0]),
                n=int(test.loc[test['quantile'].eq(chosen),'n'].iloc[0])))
    pd.DataFrame(cv).to_csv(OUT/'leave_one_video_out.csv',index=False,encoding='utf-8-sig')
    return summary,agg


def plots(pred,summary,agg):
    plt.rcParams['font.family']='Meiryo'; plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    for ax,(dataset,g) in zip(axes.flat,summary.groupby('dataset')):
        for direction,label in [('right','右向き'),('all','全方向')]:
            s=g.loc[g.direction.eq(direction)&g.evaluation.eq('all_estimates')]
            ax.plot(s['quantile']*100,s.mae,'o-',label=f'{label}（n={int(s.n.iloc[0])}）')
        ax.set_title(dataset.replace('_timestamp_JST','').replace('_12s','（12秒区間）'))
        ax.set_xticks(np.array(QS)*100); ax.set_xlabel('右向き分位点（％）'); ax.set_ylabel('MAE（cm/s）')
        ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.suptitle('手動測定付き6動画：分位点70～85％の感度分析（選別前）',fontsize=18,weight='bold')
    fig.savefig(OUT/'per_video_sensitivity.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for col,scope in enumerate(['additional5','all6']):
        for mode,label in [('all_estimates','全推定点'),('confidence_ge_085','信頼度0.85以上'),('common_confidence_ge_085','全設定の共通採用点')]:
            s=agg.loc[agg.scope.eq(scope)&agg.direction.eq('right')&agg.evaluation.eq(mode)]
            axes[0,col].plot(s['quantile']*100,s.macro_mae,'o-',label=label)
            axes[1,col].plot(s['quantile']*100,s.coverage*100,'o-',label=label)
        axes[0,col].set_title(('追加5動画' if scope=='additional5' else 'GX010202を含む全6動画')+'：右向きMAE（動画等重み）')
        axes[1,col].set_title('右向き区間の採用率（測定点数で集計）')
    for ax in axes.flat:
        ax.set_xticks(np.array(QS)*100); ax.set_xlabel('右向き分位点（％）'); ax.grid(alpha=.25); ax.legend(fontsize=9)
    for ax in axes[0,:]: ax.set_ylabel('動画別MAEの平均（cm/s）')
    for ax in axes[1,:]: ax.set_ylabel('採用率（％）')
    fig.suptitle('右向き分位点の感度：誤差と採用率',fontsize=18,weight='bold')
    fig.savefig(OUT/'aggregate_sensitivity.png',dpi=170); plt.close(fig)
    plotdir=OUT/'manual_comparison'; plotdir.mkdir(exist_ok=True)
    colors={.70:'#718096',.75:'#2474D8',.80:'#279446',.85:'#9C42A0'}
    for dataset,g in pred.groupby('dataset'):
        fig,ax=plt.subplots(figsize=(14,6),constrained_layout=True)
        m=g.loc[g['quantile'].eq(.75)].sort_values('time_s')
        def gap_broken(t,v):
            x,y=[],[]
            last=None
            for ti,vi in zip(t,v):
                if last is not None and ti-last>.05:
                    x.append(np.nan); y.append(np.nan)
                x.append(ti); y.append(vi); last=ti
            return x,y
        ax.plot(*gap_broken(m.time_s,m.manual),'o-',color='#D52D2D',lw=1,ms=3,label='手動計測')
        for q,color in colors.items():
            p=g.loc[g['quantile'].eq(q)].sort_values('time_s')
            ax.plot(*gap_broken(p.time_s,p.predicted_cm_s),'.-',lw=.8,ms=3,alpha=.75,color=color,label=f'自動 {q*100:g}％（選別前）')
        name=dataset.replace('_timestamp_JST','')
        ax.set_title(name+'：手動と自動推定の比較',fontsize=16,weight='bold')
        ax.set_xlabel('系列開始からの時刻（s）［0.05秒超の空白は結線なし］'); ax.set_ylabel('流速（cm/s）'); ax.grid(alpha=.25)
        ax.axhline(0,color='gray',lw=.8); ax.legend(ncol=3,fontsize=9)
        fig.savefig(plotdir/(name+'_manual_auto.png'),dpi=170); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.analyze_only:
        records=pd.read_csv(OUT/'pair_manifest.csv').to_dict('records')
        cache=load_cache()
    else:
        records=prepare()
        if args.prepare_only: return
        cache=extract(records,args.limit)
    if len(cache)!=len(records):
        print(f'CHECKPOINT {len(cache)}/{len(records)}',flush=True); return
    pred=predictions(records,cache)
    summary,agg=summarize(pred)
    plots(pred,summary,agg)
    print(agg.loc[agg.direction.eq('right')&agg.evaluation.eq('all_estimates')].to_string(index=False),flush=True)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__': main()
