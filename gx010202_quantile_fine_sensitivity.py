"""Reproducible paired sensitivity; reuse exact image matches, never refit tracking."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from auto_pair_benchmark import estimate_from_matches

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'GX010202_quantile_sensitivity_20261008'
OUT = ROOT / 'GX010202_quantile_sensitivity_20261027'
QS = [x / 1000 for x in range(700, 851, 25)]


def metrics(g, total):
    g = g.loc[g.predicted_cm_s.notna()]
    e = g.predicted_cm_s - g.manual
    return dict(n=len(g), total=total, coverage=len(g)/total if total else np.nan,
                mae=e.abs().mean(), rmse=np.sqrt((e**2).mean()), bias=e.mean(),
                median_ae=e.abs().median(), p90_ae=e.abs().quantile(.9))


def main():
    OUT.mkdir(exist_ok=True)
    old = pd.read_csv(SOURCE / 'predictions.csv')
    base = old.loc[old['quantile'].eq(.75)].sort_values('row').copy()
    assert len(base) == 1273 and base.row.is_unique
    cache = {}
    for line in (SOURCE/'matches_manual_pairs.jsonl').read_text(encoding='utf-8').splitlines():
        item = json.loads(line)
        assert item['row'] not in cache
        cache[item['row']] = item
    assert set(cache) == set(base.row)
    scale = float(pd.read_csv(ROOT/'all_patterns_step1_temporal_v2_20260831'/'per_sequence'/'GX010202_timestamp_JST_velocity_step1_temporal_v2.csv').cm_per_pixel.iloc[0])
    records = []
    for r in base.itertuples(index=False):
        item = cache[r.row]
        assert (item['before_file'], item['after_file']) == (r.before_file, r.after_file)
        assert np.isclose(item['manual'], r.manual)
        for q in QS:
            result = estimate_from_matches(item['matches'], positive_quantile=q, negative_quantile=.25)
            records.append(dict(row=r.row, source_before_frame=r.source_before_frame,
                source_after_frame=r.source_after_frame, before_file=r.before_file,
                after_file=r.after_file, time_s=r.time_s, gap=r.gap,
                manual=r.manual, direction=r.direction, quantile=q,
                predicted_cm_s=np.nan if result is None else result['dx']*scale*240/r.gap,
                confidence=np.nan if result is None else result['confidence']))
    pred = pd.DataFrame(records)
    assert len(pred) == 1273*7 and not pred.duplicated(['row','quantile']).any()
    assert pred.predicted_cm_s.notna().all()
    # Reproduce the old shared settings before interpreting new ones.
    for q in [.75,.85]:
        check = pred.loc[pred['quantile'].eq(q)].merge(old.loc[old['quantile'].eq(q)], on='row',suffixes=('_new','_old'),validate='one_to_one')
        assert np.allclose(check.predicted_cm_s_new,check.predicted_cm_s_old,atol=1e-10)
        assert np.allclose(check.confidence_new,check.confidence_old,atol=1e-10)
    pred['accepted_conf085'] = pred.confidence >= .85
    common = pred.groupby('row').accepted_conf085.all()
    pred['common_conf085'] = pred.row.map(common)
    summaries = []
    for q,g in pred.groupby('quantile'):
        for direction in ['right','left','slow','all']:
            sub = g if direction == 'all' else g.loc[g.direction.eq(direction)]
            for mode,mask in [('all_estimates',np.ones(len(sub),dtype=bool)),
                              ('confidence_ge_085',sub.accepted_conf085),
                              ('common_confidence_ge_085',sub.common_conf085)]:
                summaries.append(dict(quantile=q,direction=direction,evaluation=mode,**metrics(sub.loc[mask],len(sub))))
    summary = pd.DataFrame(summaries)
    blocks = []
    right = pred.loc[pred.direction.eq('right')].copy()
    ids = base.loc[base.direction.eq('right')].sort_values('time_s').row.to_numpy()
    blockmap = {int(i):b+1 for b,chunk in enumerate(np.array_split(ids,5)) for i in chunk}
    right['block'] = right.row.map(blockmap)
    for (block,q),g in right.groupby(['block','quantile']):
        blocks.append(dict(block=block,quantile=q,time_min=g.time_s.min(),time_max=g.time_s.max(),**metrics(g,len(g))))
    blockdf = pd.DataFrame(blocks)
    # Hold out contiguous chronological blocks. These are correlated portions
    # of ONE video, not independent video-level validation.
    cv=[]
    for block in range(1,6):
        train = right.loc[right.block.ne(block)].groupby('quantile').apply(lambda g:(g.predicted_cm_s-g.manual).abs().mean(),include_groups=False)
        chosen=float(train.idxmin())
        test=right.loc[right.block.eq(block)&right['quantile'].eq(chosen)]
        baseline=right.loc[right.block.eq(block)&right['quantile'].eq(.75)]
        cv.append(dict(held_out_block=block,selected_quantile=chosen,test_mae=metrics(test,len(test))['mae'],baseline75_mae=metrics(baseline,len(baseline))['mae'],n=len(test)))
    cvdf=pd.DataFrame(cv)
    pred.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(OUT/'summary.csv',index=False,encoding='utf-8-sig')
    blockdf.to_csv(OUT/'rightward_time_blocks.csv',index=False,encoding='utf-8-sig')
    cvdf.to_csv(OUT/'chronological_holdout.csv',index=False,encoding='utf-8-sig')
    plt.rcParams['font.family']='Meiryo'
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    fig.suptitle('GX010202：累積重み70～85％の感度分析（2.5％刻み）',fontsize=18,weight='bold')
    modes=[('all_estimates','全推定点'),('confidence_ge_085','信頼度0.85以上'),('common_confidence_ge_085','全7設定の共通採用点')]
    for mode,label in modes:
        s=summary.loc[summary.direction.eq('right')&summary.evaluation.eq(mode)]
        axes[0,0].plot(s['quantile']*100,s.mae,'o-',label=f'{label}（n={s.n.min()}～{s.n.max()}）')
    axes[0,0].set_title('右向き区間：平均絶対誤差（MAE）')
    axes[0,0].legend(fontsize=9)
    s=summary.loc[summary.direction.eq('right')&summary.evaluation.eq('all_estimates')]
    axes[0,1].plot(s['quantile']*100,s.rmse,'s-',label='RMSE')
    axes[0,1].plot(s['quantile']*100,s.bias,'o-',label='平均誤差（自動－手動）')
    axes[0,1].axhline(0,color='gray',lw=.8)
    axes[0,1].set_title('右向き区間：二乗平均平方根誤差と偏り')
    axes[0,1].legend()
    s=summary.loc[summary.direction.eq('right')&summary.evaluation.eq('confidence_ge_085')]
    axes[1,0].plot(s['quantile']*100,s.coverage*100,'o-',color='#278148')
    axes[1,0].set_title('右向き区間：信頼度0.85以上の採用率')
    for block,g in blockdf.groupby('block'):
        axes[1,1].plot(g['quantile']*100,g.mae,'o-',label=f'区間{block}（{g.time_min.iloc[0]:.2f}～{g.time_max.iloc[0]:.2f}秒）')
    axes[1,1].set_title('右向き区間：時系列5分割でのMAE')
    axes[1,1].legend(fontsize=9)
    for ax in axes.flat:
        ax.set_xlabel('右向き代表変位の分位点（％）')
        ax.set_xticks(np.array(QS)*100)
        ax.grid(alpha=.25)
        ax.set_ylabel('誤差（cm/s）')
    axes[1,0].set_ylabel('採用率（％）')
    fig.savefig(OUT/'sensitivity_comparison.png',dpi=170)
    plt.close(fig)
    best=float(summarize_best(summary,'right','all_estimates'))
    fig,axes=plt.subplots(2,1,figsize=(15,10),constrained_layout=True)
    for ax,limits,title in [(axes[0],None,'全区間（左向き・右向き・低速を含む）'),(axes[1],(3.2,7.0),'右向き区間の拡大')]:
        ax.plot(base.time_s,base.manual,color='#D52D2D',lw=.9,alpha=.75,label='手動計測')
        colors={.70:'#718096',.75:'#2474D8',.80:'#279446',.85:'#9C42A0'}
        for q in sorted(set([.70,.75,best,.85])):
            g=pred.loc[pred['quantile'].eq(q)].sort_values('time_s')
            ax.plot(g.time_s,g.predicted_cm_s,lw=.75,alpha=.7,color=colors[q],label=f'自動 {q*100:g}％（選別前）')
        if limits: ax.set_xlim(*limits); ax.set_ylim(-15,135)
        ax.set_title(title); ax.set_xlabel('系列開始からの時刻（s）'); ax.set_ylabel('流速（cm/s）')
        ax.axhline(0,color='gray',lw=.7); ax.grid(alpha=.25); ax.legend(ncol=3,fontsize=9)
    fig.suptitle('GX010202：手動計測と分位点設定の比較',fontsize=18,weight='bold')
    fig.savefig(OUT/'manual_auto_comparison.png',dpi=170)
    plt.close(fig)
    print(summary.loc[summary.direction.eq('right')].to_string(index=False))
    print('\nALL DIRECTIONS\n',summary.loc[summary.direction.eq('all')&summary.evaluation.eq('all_estimates')].to_string(index=False))
    print('\nBLOCK HOLDOUT\n',cvdf.to_string(index=False))
    print('OUTPUT',OUT)


def summarize_best(summary,direction,mode):
    g=summary.loc[summary.direction.eq(direction)&summary.evaluation.eq(mode)]
    return g.loc[g.mae.idxmin(),'quantile']


if __name__=='__main__':
    main()
