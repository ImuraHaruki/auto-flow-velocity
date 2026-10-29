"""Leftward weighted-quantile sensitivity on all cached exact manual pairs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from auto_pair_benchmark import estimate_from_matches
from gx010202_quantile_fine_sensitivity import metrics
from report_all_manual_sensitivity import mdtable

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'all_manual_quantile_sensitivity_20261027'
OUT=ROOT/'all_manual_left_quantile_sensitivity_20261027'
QS=[x/1000 for x in range(150,301,25)]


def calculate():
    base=pd.read_csv(SOURCE/'predictions.csv')
    base=base.loc[base['quantile'].eq(.75)].copy()
    assert len(base)==2245 and not base.duplicated(['dataset','row']).any()
    calibration=json.loads((ROOT/'all_patterns_step1_temporal_v2_20260831'/'calibration_map.json').read_text(encoding='utf-8'))
    cache={}
    paths=[SOURCE/'matches_manual_pairs.jsonl',ROOT/'GX010202_quantile_sensitivity_20261008'/'matches_manual_pairs.jsonl']
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            item=json.loads(line)
            dataset=item.get('dataset','GX010202_timestamp_JST')
            key=(dataset,item['row'])
            assert key not in cache
            cache[key]=item
    assert len(cache)==2245
    rows=[]
    for r in base.itertuples(index=False):
        item=cache[(r.dataset,r.row)]
        assert item['before_file']==r.before_file and item['after_file']==r.after_file
        assert np.isclose(item['manual'],r.manual)
        scale=calibration[r.dataset]['cm_per_pixel']
        for q in QS:
            result=estimate_from_matches(item['matches'],positive_quantile=.75,negative_quantile=q)
            rows.append(dict(dataset=r.dataset,row=r.row,before_file=r.before_file,after_file=r.after_file,
                source_before_frame=r.source_before_frame,source_after_frame=r.source_after_frame,
                time_s=r.time_s,gap=r.gap,manual=r.manual,direction=r.direction,cm_per_pixel=scale,
                quantile=q,positive_quantile=.75,negative_quantile=q,
                predicted_cm_s=np.nan if result is None else result['dx']*scale*240/r.gap,
                confidence=np.nan if result is None else result['confidence'],trusted_matches=len(item['matches'])))
    p=pd.DataFrame(rows)
    assert len(p)==2245*len(QS) and not p.duplicated(['dataset','row','quantile']).any()
    check=p.loc[p['quantile'].eq(.25)].merge(base,on=['dataset','row'],validate='one_to_one',suffixes=('_new','_old'))
    assert np.allclose(check.predicted_cm_s_new,check.predicted_cm_s_old,atol=1e-9)
    assert np.allclose(check.confidence_new,check.confidence_old,atol=1e-10)
    p['accepted_conf085']=p.confidence>=.85
    common=p.groupby(['dataset','row']).accepted_conf085.all()
    p['common_conf085']=[common.loc[(d,r)] for d,r in zip(p.dataset,p.row)]
    p.to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    return p


def summarize(p):
    rows=[]
    for (dataset,q),g in p.groupby(['dataset','quantile']):
        for direction in ['all','left','right','slow']:
            sub=g if direction=='all' else g.loc[g.direction.eq(direction)]
            for mode,data in [('all_estimates',sub),('confidence_ge_085',sub.loc[sub.accepted_conf085]),('common_confidence_ge_085',sub.loc[sub.common_conf085])]:
                rows.append(dict(dataset=dataset,quantile=q,direction=direction,evaluation=mode,**metrics(data,len(sub))))
    s=pd.DataFrame(rows)
    s.to_csv(OUT/'per_video_summary.csv',index=False,encoding='utf-8-sig')
    aggregate=[]
    for scope in ['all6','without_GX010202']:
        sub=s if scope=='all6' else s.loc[~s.dataset.str.startswith('GX010202')]
        for (q,direction,mode),g in sub.groupby(['quantile','direction','evaluation']):
            valid=g.loc[g.n>0]
            aggregate.append(dict(scope=scope,quantile=q,direction=direction,evaluation=mode,
                videos=len(valid),n=int(valid.n.sum()),total=int(g.total.sum()),
                coverage=valid.n.sum()/g.total.sum() if g.total.sum() else np.nan,
                macro_mae=valid.mae.mean(),macro_rmse=valid.rmse.mean(),
                pooled_mae=(valid.mae*valid.n).sum()/valid.n.sum() if valid.n.sum() else np.nan,
                pooled_rmse=np.sqrt((valid.rmse**2*valid.n).sum()/valid.n.sum()) if valid.n.sum() else np.nan,
                pooled_bias=(valid.bias*valid.n).sum()/valid.n.sum() if valid.n.sum() else np.nan))
    a=pd.DataFrame(aggregate)
    a.to_csv(OUT/'aggregate_summary.csv',index=False,encoding='utf-8-sig')
    cv=[]
    left=s.loc[s.direction.eq('left')&s.evaluation.eq('all_estimates')]
    bests=[]
    for dataset,g in left.groupby('dataset'):
        best=g.loc[g.mae.idxmin()]
        b=g.loc[g['quantile'].eq(.25)].iloc[0]
        bests.append(dict(dataset=dataset,n=int(b.n),best_quantile=best['quantile'],best_mae=best.mae,
            mae25=b.mae,rmse25=b.rmse,best_rmse=best.rmse))
        train=left.loc[left.dataset.ne(dataset)].groupby('quantile').mae.mean()
        selected=float(train.idxmin())
        test=g.loc[g['quantile'].eq(selected)].iloc[0]
        cv.append(dict(held_out=dataset,selected_quantile=selected,n=int(test.n),test_mae=test.mae,baseline25_mae=b.mae))
    b=pd.DataFrame(bests); c=pd.DataFrame(cv)
    b.to_csv(OUT/'best_by_video.csv',index=False,encoding='utf-8-sig')
    c.to_csv(OUT/'leave_one_video_out.csv',index=False,encoding='utf-8-sig')
    # Repeated observations are retained above; inspect an equal-image-pair alternative.
    keys=['dataset','before_file','after_file','quantile']
    spread=p.groupby(keys).predicted_cm_s.agg(lambda g:g.max()-g.min())
    assert spread.dropna().le(1e-9).all()
    u=p.groupby(keys,as_index=False).agg(manual=('manual','median'),predicted_cm_s=('predicted_cm_s','first'))
    ur=[]
    for (dataset,q),g in u.groupby(['dataset','quantile']):
        sub=g.loc[g.manual < -10]
        ur.append(dict(dataset=dataset,quantile=q,**metrics(sub,len(sub))))
    unique=pd.DataFrame(ur)
    unique.to_csv(OUT/'unique_pair_summary.csv',index=False,encoding='utf-8-sig')
    return s,a,b,c,unique


def plots(p,s,a,b):
    plt.rcParams['font.family']='Meiryo'; plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(2,3,figsize=(17,10),constrained_layout=True)
    for ax,(dataset,g) in zip(axes.flat,s.groupby('dataset')):
        for direction,label in [('left','左向き'),('all','全方向')]:
            v=g.loc[g.direction.eq(direction)&g.evaluation.eq('all_estimates')]
            ax.plot(v['quantile']*100,v.mae,'o-',label=f'{label}（n={int(v.n.iloc[0])}）')
        ax.set_title(dataset.replace('_timestamp_JST','').replace('_12s','（12秒区間）'))
        ax.set_xlabel('左向き分位点（％）'); ax.set_ylabel('MAE（cm/s）')
        ax.set_xticks(np.array(QS)*100); ax.grid(alpha=.25); ax.legend(fontsize=9)
    fig.suptitle('手動測定付き6動画：左向き分位点の感度分析（右向き75％固定・選別前）',fontsize=17,weight='bold')
    fig.savefig(OUT/'per_video_sensitivity.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,10),constrained_layout=True)
    for mode,label in [('all_estimates','全推定点'),('confidence_ge_085','信頼度0.85以上'),('common_confidence_ge_085','全設定の共通採用点')]:
        v=a.loc[a.scope.eq('all6')&a.direction.eq('left')&a.evaluation.eq(mode)]
        axes[0,0].plot(v['quantile']*100,v.macro_mae,'o-',label=label)
        axes[1,0].plot(v['quantile']*100,v.coverage*100,'o-',label=label)
    v=a.loc[a.scope.eq('all6')&a.direction.eq('left')&a.evaluation.eq('all_estimates')]
    axes[0,1].plot(v['quantile']*100,v.pooled_mae,'o-',label='全測定点MAE')
    axes[0,1].plot(v['quantile']*100,v.pooled_rmse,'s-',label='全測定点RMSE')
    for scope,label in [('all6','全6動画'),('without_GX010202','GX010202を除く5動画')]:
        v=a.loc[a.scope.eq(scope)&a.direction.eq('left')&a.evaluation.eq('all_estimates')]
        axes[1,1].plot(v['quantile']*100,v.macro_mae,'o-',label=label)
    axes[0,0].set_title('左向きMAE：動画を等重みで平均')
    axes[0,1].set_title('左向き誤差：全測定点をまとめて評価')
    axes[1,0].set_title('左向き区間の採用率')
    axes[1,1].set_title('GX010202の有無による比較（動画等重み）')
    for ax in axes.flat:
        ax.set_xlabel('左向き分位点（％）'); ax.set_xticks(np.array(QS)*100)
        ax.set_ylabel('誤差（cm/s）'); ax.grid(alpha=.25); ax.legend(fontsize=9)
    axes[1,0].set_ylabel('採用率（％）')
    fig.suptitle('左向き分位点15～30％：誤差と採用率',fontsize=18,weight='bold')
    fig.savefig(OUT/'aggregate_sensitivity.png',dpi=170); plt.close(fig)
    plotdir=OUT/'manual_comparison'; plotdir.mkdir(exist_ok=True)
    for dataset,g in p.groupby('dataset'):
        best=float(b.loc[b.dataset.eq(dataset),'best_quantile'].iloc[0])
        fig,ax=plt.subplots(figsize=(14,6),constrained_layout=True)
        def broken(times,values):
            xs,ys=[],[]; prev=None
            for t,v in zip(times,values):
                if prev is not None and t-prev>.05: xs.append(np.nan); ys.append(np.nan)
                xs.append(t); ys.append(v); prev=t
            return xs,ys
        m=g.loc[g['quantile'].eq(.25)].sort_values('time_s')
        ax.plot(*broken(m.time_s,m.manual),'o-',color='#D52D2D',lw=.9,ms=3,label='手動計測')
        palette=['#718096','#2474D8','#279446','#9C42A0']
        for q,color in zip(sorted(set([.15,.25,.30,best])),palette):
            v=g.loc[g['quantile'].eq(q)].sort_values('time_s')
            ax.plot(*broken(v.time_s,v.predicted_cm_s),'.-',color=color,lw=.8,ms=3,label=f'自動 左向き{q*100:g}％（選別前）')
        name=dataset.replace('_timestamp_JST','')
        ax.set_title(name+'：左向き分位点と手動計測の比較',fontsize=16,weight='bold')
        ax.set_xlabel('系列開始からの時刻（s）［0.05秒超の空白は結線なし］')
        ax.set_ylabel('流速（cm/s）'); ax.grid(alpha=.25); ax.axhline(0,color='gray',lw=.7)
        ax.legend(fontsize=9,ncol=3)
        fig.savefig(plotdir/(name+'_manual_auto.png'),dpi=170); plt.close(fig)


def report(p,s,a,b,c,u):
    left=a.loc[a.scope.eq('all6')&a.direction.eq('left')&a.evaluation.eq('all_estimates')]
    best=left.loc[left.macro_mae.idxmin()]
    baseline=left.loc[left['quantile'].eq(.25)].iloc[0]
    pooled=left.loc[left.pooled_mae.idxmin()]
    rmse=left.loc[left.pooled_rmse.idxmin()]
    uq=u.groupby('quantile',as_index=False).agg(macro_mae=('mae','mean'),n=('n','sum'))
    uqbest=uq.loc[uq.macro_mae.idxmin()]
    common=a.loc[a.scope.eq('all6')&a.direction.eq('left')&a.evaluation.eq('common_confidence_ge_085')]
    commonbest=common.loc[common.macro_mae.idxmin()]
    text=f'''# 左向き流れ：重み付き分位点15～30％の感度分析

実施日：2026-10-27

## 要約

- 左向き分位点15、17.5、20、22.5、25、27.5、30％を比較。右向きは75％固定。
- 6動画・全2245件の手動測定を使用し、左向き（手動速度が-10 cm/s未満）は{int(best.n)}件。
- 左向きの動画等重みMAE最小は{best['quantile']*100:g}％（{best.macro_mae:.3f} cm/s）。現行25％の{baseline.macro_mae:.3f} cm/sから{(baseline.macro_mae-best.macro_mae)/baseline.macro_mae*100:.2f}％改善。
- 左向き全測定点をまとめたMAE最小は{pooled['quantile']*100:g}％、RMSE最小は{rmse['quantile']*100:g}％。
- 同じ画像ペアの重複をまとめた場合、動画等重みMAE最小は{uqbest['quantile']*100:g}％。
- 最小値は今回の範囲内のもの。未知動画や範囲外の最適性は保証しない。

## 最良値の解釈

主基準を「動画を等重みで平均した、選別前の左向きMAE」とするなら、範囲内の最良値は20％。GX010202を除いても、重複をまとめても、この主基準の最小値は20％だった。

一方、全測定点をまとめたMAEと共通採用点の動画等重みMAEでは22.5％が最小。実運用候補として、20％は選別前の平均的な精度を重視する設定、22.5％は採用率とのバランスを取りやすい設定と考えられる。信頼度0.85以上の採用率は20％で65.66％、22.5％で67.86％、現行25％で68.41％。

ただし、動画を1本ずつ保留した補助評価では分位点選択方式が25％固定を上回らなかった。したがって未知動画にも通用する優位性が確立したとは言えず、設定変更は実施していない。時間フィルタ込みの最終採用率・誤差は別途検証が必要。

## 分位点の意味と固定条件

符号付き変位を小さい順（負の大きな変位から）に並べて重みを累積する。左向きでは15％は速い側、30％は遅い側。15～30％は、前回の右向き70～85％に方向を反転して対応させた範囲である。これは照合重みそのものや信頼度の係数を変更する実験ではない。

各画像ペアの局所変位・重みは既存キャッシュから再利用。画像の再照合はしていない。全動画240 fpsで、校正値・画像間隔は前回と同じ。現行の右75％・左25％の速度と信頼度が全2245件で前回の結果を再現することを確認した。

方向区分は手動速度に基づく。右向きは10 cm/s超、左向きは-10 cm/s未満、低速はその間。誤差は自動－手動、MAEは絶対誤差の平均、RMSEは二乗誤差平均の平方根。動画等重みMAEは動画別MAEの単純平均で、測定点数の多いGX010202だけに引っ張られない評価である。

## 動画別の左向き区間（選別前）

'''
    text+=mdtable(b,[('dataset','動画'),('n','左向き点数'),('mae25','現行25％のMAE'),('best_quantile','範囲内最良分位点'),('best_mae','最小MAE')])
    text+='\n\n## 全6動画の左向き集計（選別前）\n\n'
    text+=mdtable(left,[('quantile','左向き分位点'),('n','点数'),('macro_mae','動画等重みMAE'),('pooled_mae','全測定点MAE'),('pooled_rmse','全測定点RMSE'),('pooled_bias','平均誤差')])
    for mode,label in [('confidence_ge_085','信頼度0.85以上'),('common_confidence_ge_085','全7設定の共通採用点')]:
        text+=f'\n\n## 左向き：{label}\n\n'
        v=a.loc[a.scope.eq('all6')&a.direction.eq('left')&a.evaluation.eq(mode)]
        text+=mdtable(v,[('quantile','分位点'),('n','採用点数'),('coverage','採用率'),('macro_mae','動画等重みMAE'),('pooled_mae','全測定点MAE')])
    text+='\n\n## 全方向・低速も含む評価（選別前）\n\n'
    v=a.loc[a.scope.eq('all6')&a.direction.eq('all')&a.evaluation.eq('all_estimates')]
    text+=mdtable(v,[('quantile','左向き分位点'),('n','点数'),('macro_mae','動画等重みMAE'),('pooled_mae','全測定点MAE'),('pooled_rmse','全測定点RMSE')])
    text+='\n\n## 動画を1本ずつ保留した補助評価\n\n残り5動画の左向き・動画等重みMAEから分位点を選び、保留動画で評価した。同じ撮影条件の動画であり、独立した外部検証ではない。\n\n'
    text+=mdtable(c,[('held_out','保留動画'),('selected_quantile','他5動画で選択'),('n','点数'),('test_mae','保留動画MAE'),('baseline25_mae','25％固定MAE')])
    text+=f'\n\n保留動画MAEの動画等重み平均：選択方式{c.test_mae.mean():.3f} cm/s、25％固定{c.baseline25_mae.mean():.3f} cm/s。\n'
    text+='\n## 重複をまとめた補助評価\n\n主解析では元の手動測定行を削除していない。同じ画像ペアの手動速度を中央値にまとめ、画像ペアを1回ずつ数えた補助結果は以下。\n\n'
    text+=mdtable(uq,[('quantile','左向き分位点'),('n','左向きユニークペア数'),('macro_mae','動画等重みMAE')])
    text+='''

## 限界・出力

- 時間整合性フィルタ適用前の、手動と同じ画像ペアでの比較。時間平滑化・欠測補間・適応フレーム間隔の選び直しは実施していない。
- 信頼度による選別では採用点の母集団が変わるため、全点と共通採用点も併記した。
- 動画によって測定数が異なり、特に少数の左向き点しかない動画の最良値は不安定。手動測定を誤差のない真値とは見なさない。有意差検定は実施していない。
- 最良値が範囲端にある動画では、範囲外でさらに改善する可能性がある。
- 現行ソフトや既存の全動画推定結果は変更していない。
- 対象6動画はGX010101・105・164の12秒区間・165・202・213。別フォルダのGX010164（12秒切り出しでない手動14件）は含まない。
- `predictions.csv` の `quantile` は今回の左向き分位点。明示列 `positive_quantile=0.75`、`negative_quantile` も収録。
- `per_video_summary.csv`：動画別・方向別・選別別の指標。
- `aggregate_summary.csv`：全6動画、およびGX010202を除く5動画の集計。
- `best_by_video.csv`：動画別の最良値と現行25％との比較。
- `leave_one_video_out.csv`：動画単位の保留評価。
- `unique_pair_summary.csv`：重複をまとめた補助評価。
- `per_video_sensitivity.png` と `aggregate_sensitivity.png`：感度分析図。
- `manual_comparison/`：手動と自動推定の時系列比較図。0.05秒超の空白では線を切っている。
'''
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8')
    sources=[Path(__file__).resolve(),ROOT/'auto_pair_benchmark.py',SOURCE/'predictions.csv',SOURCE/'matches_manual_pairs.jsonl',ROOT/'GX010202_quantile_sensitivity_20261008'/'matches_manual_pairs.jsonl',ROOT/'all_patterns_step1_temporal_v2_20260831'/'calibration_map.json']
    metadata=dict(date='2026-10-27',fps=240,positive_quantile=.75,negative_quantiles=QS,
        manual_rows=2245,estimates=len(p),left_manual_rows=int(best.n),missing_estimates=int(p.predicted_cm_s.isna().sum()),
        baseline25_verified=True,source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('VIDEO BEST\n',b.to_string(index=False),flush=True)
    print('LEFT ALL\n',left.to_string(index=False),flush=True)
    print('ALL DIRECTIONS\n',a.loc[a.scope.eq('all6')&a.direction.eq('all')&a.evaluation.eq('all_estimates')].to_string(index=False),flush=True)
    print('LOVO\n',c.to_string(index=False),flush=True)
    print('UNIQUE\n',uq.to_string(index=False),flush=True)


def main():
    OUT.mkdir(exist_ok=True)
    p=calculate()
    s,a,b,c,u=summarize(p)
    plots(p,s,a,b)
    report(p,s,a,b,c,u)
    print('COMPLETE',OUT,flush=True)


if __name__=='__main__': main()
