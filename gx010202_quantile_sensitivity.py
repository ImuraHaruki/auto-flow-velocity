"""Paired quantile sensitivity on every manually labelled GX010202 image pair."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auto_pair_benchmark import match_candidates, estimate_from_matches
from benchmark_tracking import DATASETS, list_images, image_resolver, grayscale
from plot_tracking_candidates_full_frame import set_font

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'GX010202_quantile_sensitivity_20261008'
QS = [.50, .65, .75, .85, .95]
CACHE = OUT / 'matches_manual_pairs.jsonl'


def main():
    OUT.mkdir(exist_ok=True)
    dataset = next(d for d in DATASETS if d.name == 'B')
    labels = pd.read_csv(dataset.labels)
    images = list_images(dataset.frames)
    resolve = image_resolver(images)
    image_indices = {p.name:i for i,p in enumerate(images)}
    scale = float(pd.read_csv(ROOT / 'all_patterns_step1_temporal_v2_20260831' / 'per_sequence' / 'GX010202_timestamp_JST_velocity_step1_temporal_v2.csv').cm_per_pixel.iloc[0])
    records = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding='utf-8').splitlines():
            if line.strip():
                item = json.loads(line)
                records[item['row']] = item

    def extract(index, row):
        before = resolve(str(row.before_file),str(row.before_frame))
        after = resolve(str(row.after_file),str(row.after_frame))
        # Resolve by the same historical name/number/index logic, and retain
        # the exact filenames in the audit output.
        matches = match_candidates(grayscale(before),grayscale(after),radius=32,candidate_count=80)
        gap = int(row.after_frame)-int(row.before_frame)
        if gap <= 0:
            raise ValueError('Nonpositive manual frame interval')
        return dict(row=int(index),before_frame=int(row.before_frame),after_frame=int(row.after_frame),
                    before_file=before.name,after_file=after.name,manual_before_file=str(row.before_file),
                    manual_after_file=str(row.after_file),time_s=(int(row.before_frame)-1)/240,
                    gap=gap,manual=float(row.velocity),matches=matches)

    start = time.monotonic()
    todo = [(i,row) for i,row in labels.iterrows() if i not in records]
    print(f'manual_rows={len(labels)} cached={len(records)} pending={len(todo)} scale={scale}',flush=True)
    with CACHE.open('a',encoding='utf-8') as handle, ThreadPoolExecutor(max_workers=4) as pool:
        jobs = {pool.submit(extract,i,row):i for i,row in todo}
        for n,future in enumerate(as_completed(jobs),1):
            item = future.result()
            records[item['row']] = item
            handle.write(json.dumps(item)+'\n')
            handle.flush()
            if n==1 or n%25==0 or n==len(todo):
                elapsed = time.monotonic()-start
                print(f'completed={n}/{len(todo)} elapsed_s={elapsed:.1f} eta_s={elapsed/n*(len(todo)-n):.1f}',flush=True)

    rows = []
    for item in sorted(records.values(),key=lambda r:r['row']):
        # CSV frame numbering is relative to the GUI's imported subset, while
        # filenames identify the original full-sequence images.
        item['source_before_frame'] = image_indices[item['before_file']] + 1
        item['source_after_frame'] = image_indices[item['after_file']] + 1
        item['time_s'] = image_indices[item['before_file']] / 240
        item['gap'] = image_indices[item['after_file']] - image_indices[item['before_file']]
        for q in QS:
            result = estimate_from_matches(item['matches'],positive_quantile=q,negative_quantile=.25)
            velocity = np.nan if result is None else result['dx']*scale*240/item['gap']
            rows.append({k:v for k,v in item.items() if k!='matches'} | dict(
                quantile=q,predicted_cm_s=velocity,
                confidence=np.nan if result is None else result['confidence'],
                trusted_matches=len(item['matches']),
                support=0 if result is None else result['support']))
    predictions = pd.DataFrame(rows)
    predictions['direction'] = np.select([predictions.manual>10,predictions.manual < -10],['right','left'],default='slow')
    predictions['accepted_conf085'] = predictions.confidence>=.85
    common_rows = predictions.groupby('row').accepted_conf085.all()
    predictions['common_accepted_conf085'] = predictions.row.map(common_rows)
    predictions.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    summaries=[]
    for (q,),group in predictions.groupby(['quantile']):
        for name,subset in [('all',group),*[(d,group[group.direction.eq(d)]) for d in ['right','left','slow']]]:
            for mode,data in [('all_estimates',subset),('confidence_ge_085',subset[subset.accepted_conf085]),('common_confidence_ge_085',subset[subset.common_accepted_conf085])]:
                valid=data[data.predicted_cm_s.notna()]
                error=valid.predicted_cm_s-valid.manual
                summaries.append(dict(quantile=q,direction=name,evaluation=mode,total=len(subset),n=len(valid),
                    coverage=len(valid)/max(len(subset),1),mae=error.abs().mean(),rmse=np.sqrt((error**2).mean()),
                    bias=error.mean(),median_ae=error.abs().median(),p90_ae=error.abs().quantile(.9),
                    pearson_r=valid[['predicted_cm_s','manual']].corr().iloc[0,1],
                    sign_accuracy=(np.sign(valid.predicted_cm_s)==np.sign(valid.manual)).mean()))
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    # Five contiguous chronological blocks: descriptive stability, not independent video validation.
    block_rows=[]
    right=predictions[predictions.direction.eq('right')].copy()
    ordered=sorted(right.row.unique(),key=lambda i: records[i]['time_s'])
    block_map={i:b+1 for b,indices in enumerate(np.array_split(ordered,5)) for i in indices}
    right['block']=right.row.map(block_map)
    for (block,q),g in right.groupby(['block','quantile']):
        e=g.predicted_cm_s-g.manual
        block_rows.append(dict(block=block,quantile=q,n=e.notna().sum(),mae=e.abs().mean(),rmse=np.sqrt((e**2).mean()),time_min=g.time_s.min(),time_max=g.time_s.max()))
    blocks=pd.DataFrame(block_rows)
    blocks.to_csv(OUT/'rightward_time_blocks.csv',index=False,encoding='utf-8-sig')
    set_font()
    colors=['#8172B2','#55A868','#E69F00','#4C72B0','#C44E52']
    fig,axes=plt.subplots(2,2,figsize=(16,10),constrained_layout=True)
    fig.suptitle('GX010202：右向き代表変位の分位点感度分析\nSensitivity to the rightward weighted quantile',fontsize=18,weight='bold')
    for ax,d,title in [(axes[0,0],'right','右向き / Rightward (>10 cm/s)'),(axes[0,1],'all','全方向 / All directions')]:
        s=summary[(summary.direction==d)&(summary.evaluation=='all_estimates')].sort_values('quantile')
        ax.plot(s['quantile']*100,s.mae,'o-',label='MAE')
        ax.plot(s['quantile']*100,s.rmse,'s--',label='RMSE')
        ax.set_title(title)
        ax.set_xlabel('右向き分位点 / Rightward quantile (%)')
        ax.set_ylabel('誤差 / Error (cm/s)')
        ax.set_xticks([50,65,75,85,95]); ax.legend()
    for q,color in zip(QS,colors):
        g=right[right['quantile'].eq(q)].sort_values('time_s')
        axes[1,0].plot(g.time_s,g.predicted_cm_s,'.',ms=3,alpha=.6,color=color,label=f'{q:.0%}')
    manual=right[right['quantile'].eq(.85)].sort_values('time_s')
    axes[1,0].plot(manual.time_s,manual.manual,'k.',ms=3,label='手動 / Manual')
    axes[1,0].set_title('右向き手動測定との比較 / Rightward manual comparison')
    axes[1,0].set_xlabel('時刻 / Time (s)'); axes[1,0].set_ylabel('流速 / Velocity (cm/s)'); axes[1,0].legend(ncol=3)
    for block,g in blocks.groupby('block'):
        axes[1,1].plot(g['quantile']*100,g.mae,'o-',label=f'区間 / Block {block}')
    axes[1,1].set_title('時系列5区間での安定性 / Five chronological blocks')
    axes[1,1].set_xlabel('右向き分位点 / Rightward quantile (%)')
    axes[1,1].set_ylabel('MAE (cm/s)'); axes[1,1].set_xticks([50,65,75,85,95]); axes[1,1].legend()
    for ax in axes.flat: ax.grid(alpha=.25)
    fig.savefig(OUT/'sensitivity_comparison.png',dpi=180)
    plt.close(fig)
    # Reference-style comparison plots. The median is a display aid only;
    # all accuracy statistics use the original unsmoothed measurements.
    for q in QS:
        g=predictions[predictions['quantile'].eq(q)].sort_values('time_s')
        valid=g[g.predicted_cm_s.notna()]
        error=valid.predicted_cm_s-valid.manual
        fig,(ax,bx)=plt.subplots(2,1,figsize=(15,11),gridspec_kw={'height_ratios':[1.6,1]},constrained_layout=True)
        ax.scatter(g.time_s,g.predicted_cm_s,s=9,color='#2563EB',alpha=.45,label=f'自動推定・選別前 / Automatic before filtering (n={len(valid)})')
        indexed=g.set_index('source_before_frame').predicted_cm_s
        smooth=indexed.reindex(range(indexed.index.min(),indexed.index.max()+1)).rolling(13,center=True,min_periods=3).median()
        ax.plot((smooth.index-1)/240,smooth,color='#0F3D91',lw=2,label='13フレーム中央値 / 13-frame median')
        ax.plot(g.time_s,g.manual,color='#DC2626',lw=1,alpha=.75,label=f'手動計測 / Manual (n={len(g)})')
        ax.set_title(f'GX010202：右向き分位点 {q:.0%}\nManual and automatic time series — rightward quantile {q:.0%}',fontsize=16,weight='bold')
        ax.set_xlabel('時刻 / Time (s)'); ax.set_ylabel('流速 / Velocity (cm/s)'); ax.axhline(0,color='gray',lw=.8); ax.legend(fontsize=10)
        bx.scatter(valid.manual,valid.predicted_cm_s,s=11,color='#2563EB',alpha=.45)
        lo=min(valid.manual.min(),valid.predicted_cm_s.min())-10
        hi=max(valid.manual.max(),valid.predicted_cm_s.max())+10
        bx.plot([lo,hi],[lo,hi],'k--',lw=1,label='1:1')
        bx.set_xlim(lo,hi); bx.set_ylim(lo,hi); bx.set_aspect('equal',adjustable='box')
        bx.set_title('同一画像ペアでの比較 / Matched original image pairs')
        bx.set_xlabel('手動流速 / Manual velocity (cm/s)'); bx.set_ylabel('自動流速 / Automatic velocity (cm/s)')
        corr=valid[['manual','predicted_cm_s']].corr().iloc[0,1]
        bx.text(.02,.98,f'n={len(valid)}\nMAE={error.abs().mean():.2f} cm/s\nRMSE={np.sqrt((error**2).mean()):.2f} cm/s\nPearson r={corr:.3f}',transform=bx.transAxes,va='top',bbox=dict(facecolor='white',alpha=.9,edgecolor='gray'))
        bx.legend(loc='lower right')
        for axis in (ax,bx): axis.grid(alpha=.25)
        fig.savefig(OUT/f'manual_vs_auto_q{int(q*100):02d}.png',dpi=180)
        plt.close(fig)
    print(summary[summary.evaluation.eq('all_estimates')].to_string(index=False),flush=True)
    print(f'OUTPUT={OUT}',flush=True)


if __name__=='__main__':
    main()
