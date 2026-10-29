"""Write an auditable narrative after the full multi-video batch completes."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from gx010202_quantile_fine_sensitivity import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'all_manual_quantile_sensitivity_20261027'
PIPE=ROOT/'all_patterns_step1_temporal_v2_20260831'


def mdtable(frame,columns):
    lines=['| '+' | '.join(label for _,label in columns)+' |',
           '| '+' | '.join('---' for _ in columns)+' |']
    for _,r in frame.iterrows():
        values=[]
        for field,label in columns:
            v=r[field]
            if pd.isna(v): text='—'
            elif field=='quantile' or field.startswith('best_') or field=='selected_quantile': text=f'{float(v)*100:g}％'
            elif field in ('coverage','macro_coverage'): text=f'{float(v)*100:.2f}％'
            elif field in ('n','total','n_right','n_all','videos'): text=str(int(v))
            elif isinstance(v,(float,np.floating)): text=f'{v:.3f}'
            else: text=str(v)
            values.append(text)
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def main():
    p=pd.read_csv(OUT/'predictions.csv')
    s=pd.read_csv(OUT/'per_video_summary.csv')
    a=pd.read_csv(OUT/'aggregate_summary.csv')
    cv=pd.read_csv(OUT/'leave_one_video_out.csv')
    manifest=pd.read_csv(OUT/'pair_manifest.csv')
    assert len(manifest)==972
    assert p.groupby(['dataset','row']).size().eq(7).all()
    assert len(p)==(972+1273)*7
    assert not p.duplicated(['dataset','row','quantile']).any()
    # Keep all supplied measurements in the main analysis, and separately
    # check whether repeated measurements of the same pair alter the ranking.
    keys=['dataset','before_file','after_file','quantile']
    pairspread=p.groupby(keys).predicted_cm_s.agg(lambda v: v.max()-v.min())
    assert pairspread.dropna().le(1e-9).all()
    unique=p.groupby(keys,as_index=False).agg(manual=('manual','median'),predicted_cm_s=('predicted_cm_s','first'))
    unique['direction']=np.select([unique.manual>10,unique.manual < -10],['right','left'],default='slow')
    unique_rows=[]
    for (dataset,q),g in unique.groupby(['dataset','quantile']):
        for direction in ['right','all']:
            sub=g if direction=='all' else g.loc[g.direction.eq(direction)]
            unique_rows.append(dict(dataset=dataset,quantile=q,direction=direction,**metrics(sub,len(sub))))
    unique_summary=pd.DataFrame(unique_rows)
    unique_summary.to_csv(OUT/'unique_pair_summary.csv',index=False,encoding='utf-8-sig')
    audit=[]
    for dataset,g in p.groupby('dataset'):
        base=g.loc[g['quantile'].eq(.75)]
        for direction in ('all','right'):
            sub=s.loc[s.dataset.eq(dataset)&s.direction.eq(direction)&s.evaluation.eq('all_estimates')]
            assert len(sub)==7
            if sub.n.gt(0).any():
                best=sub.loc[sub.mae.idxmin()]
                baseline=sub.loc[sub['quantile'].eq(.75)].iloc[0]
                at80=sub.loc[sub['quantile'].eq(.80)].iloc[0]
                audit.append(dict(dataset=dataset,direction=direction,n=int(baseline.n),total=int(baseline.total),
                    best_quantile=best['quantile'],best_mae=best.mae,mae75=baseline.mae,mae80=at80.mae,
                    improvement_vs75=baseline.mae-best.mae))
        # Strictly verify the cached GX010202 results are unchanged.
        if dataset.startswith('GX010202'):
            prior=pd.read_csv(ROOT/'GX010202_quantile_sensitivity_20261027'/'predictions.csv')
            joined=g.merge(prior,on=['row','quantile'],suffixes=('_new','_prior'),validate='one_to_one')
            assert np.allclose(joined.predicted_cm_s_new,joined.predicted_cm_s_prior)
    bests=pd.DataFrame(audit)
    bests.to_csv(OUT/'best_by_video.csv',index=False,encoding='utf-8-sig')
    mainagg=a.loc[a.direction.eq('right')&a.evaluation.eq('all_estimates')]
    winner5=mainagg.loc[mainagg.scope.eq('additional5')].sort_values(['macro_mae','quantile']).iloc[0]
    winner6=mainagg.loc[mainagg.scope.eq('all6')].sort_values(['macro_mae','quantile']).iloc[0]
    prior75=mainagg.loc[mainagg.scope.eq('all6')&mainagg['quantile'].eq(.75)].iloc[0]
    relative=(prior75.macro_mae-winner6.macro_mae)/prior75.macro_mae*100
    text=f'''# 手動測定付き動画：累積重み分位点の感度分析

実施日：2026-10-27

## 結論の要約

- 追加5動画の右向き区間を動画等重みで評価したMAE最小は{winner5['quantile']*100:g}％（{winner5.macro_mae:.3f} cm/s）。
- GX010202を含む全6動画の同じ指標では{winner6['quantile']*100:g}％（{winner6.macro_mae:.3f} cm/s）。75％の{prior75.macro_mae:.3f} cm/sに対して{relative:.2f}％改善。
- これは今回検討した70～85％の範囲内、画像ペア単位・時間フィルタ適用前の最小値。範囲外や未知動画での最適性を保証しない。
- 右向き精度・低速精度・採用率では最良設定が異なりうる。下記の共通採用点と全方向評価を併せて判断する。
- ソフト・既存全動画結果の設定やファイルは変更していない。

### 実用上の推奨

共通設定は当面75％を維持することを推奨する。右向きの動画等重みMAEでは80％がわずかに良いものの、全測定点で集計した右向きMAE・RMSEは75％が最小であり、左向き・低速を含む全方向でも75％が良かった。80％への一律変更を裏付ける安定した優位性は確認できない。

GX010101・GX010105・GX010213では85％が候補になる一方、GX010164の12秒区間・GX010165では高い分位点が悪化する。これらの動画名に応じて後付けで設定を変えた最小誤差を、そのまま未知動画での性能と見なさないこと。

## データと方法

追加対象：GX010101、GX010105、GX010164の12秒区間、GX010165、GX010213。
追加5動画の全972手動画像ペアを再照合。GX010202の1273ペアは前回の結果を再利用し、全2245組×7設定＝15715推定を集計した。全数を対象とし、古い少数サンプルの候補キャッシュだけで評価していない。

- 分位点は70、72.5、75、77.5、80、82.5、85％。右向き代表変位を選ぶ重み付き分位点のみ変更、左向きは25％固定。
- 重みの式、照合条件、信頼度の係数は固定。各ペアの変位と重みを一度抽出し、同じデータに7設定を適用。
- 全動画240 fps。各動画の既存校正値を使用し、変位×cm/pixel×240÷画像間隔でcm/sに換算。
- 元CSVの画像名を実画像名へ解決し、実画像のインデックス差がCSVの時間間隔と一致することを検証。曖昧なフレーム番号だけの代用は許さず、全ペアの対応を `pair_manifest.csv` に記録。
- 方向区分は手動速度に基づく。右向きは10 cm/s超、左向きは-10 cm/s未満、低速はその間。
- 誤差は自動－手動。MAEは絶対誤差の平均、RMSEは二乗誤差の平均の平方根。
- `macro_mae` は動画別MAEの単純平均（動画等重み）。`pooled_mae` は全測定点のMAE（測定点等重み）。測定数の多い動画に偏らないよう前者を主に評価。
- 信頼度0.85以上の採用点と、全7設定で共通採用の点も別集計。採用点のみの比較では母集団が変わることに注意。
- 周辺時刻フィルタ、欠測の補間、適応フレーム間隔の再選択は実施していない。

## 動画別：右向き区間（選別前）

'''
    text+=mdtable(bests.loc[bests.direction.eq('right')],[('dataset','動画'),('n','点数'),('mae75','75％のMAE'),('mae80','80％のMAE'),('best_quantile','最小分位点'),('best_mae','最小MAE')])
    for scope,title in [('additional5','追加5動画'),('all6','全6動画')]:
        text+=f'\n\n## {title}：右向き区間（選別前）\n\n'
        g=a.loc[a.scope.eq(scope)&a.direction.eq('right')&a.evaluation.eq('all_estimates')]
        text+=mdtable(g,[('quantile','分位点'),('n','点数'),('macro_mae','動画等重みMAE'),('pooled_mae','全点MAE'),('pooled_rmse','全点RMSE')])
    text+='\n\n## 全6動画：右向き採用点の比較\n\n'
    for mode,title in [('confidence_ge_085','信頼度0.85以上'),('common_confidence_ge_085','全7設定で共通して信頼度0.85以上')]:
        text+=f'\n### {title}\n\n'
        g=a.loc[a.scope.eq('all6')&a.direction.eq('right')&a.evaluation.eq(mode)]
        text+=mdtable(g,[('quantile','分位点'),('n','採用点数'),('coverage','採用率'),('macro_mae','動画等重みMAE'),('pooled_mae','全点MAE')])+'\n'
    text+='\n## 全6動画：左向き・低速も含む評価（選別前）\n\n'
    g=a.loc[a.scope.eq('all6')&a.direction.eq('all')&a.evaluation.eq('all_estimates')]
    text+=mdtable(g,[('quantile','分位点'),('n','点数'),('macro_mae','動画等重みMAE'),('pooled_mae','全点MAE'),('pooled_rmse','全点RMSE')])
    text+='\n\n## 未使用動画での補助評価\n\n各動画を1本ずつ保留し、残り5動画の動画等重みMAEで分位点を選択して、保留動画で比較した。学習対象が動画ごとに変わるため、選ばれる分位点が一定である必要はない。同一撮影条件の6動画であり、完全に独立した外部検証ではない。\n\n'
    text+=mdtable(cv.loc[cv.direction.eq('right')],[('held_out','保留動画'),('selected_quantile','他5動画で選択'),('n','点数'),('test_mae','保留動画MAE'),('baseline75_mae','75％固定MAE')])
    rightcv=cv.loc[cv.direction.eq('right')]
    text+=f'\n\n保留動画のMAEを動画等重みで平均すると、選択方式は{rightcv.test_mae.mean():.3f} cm/s、75％固定は{rightcv.baseline75_mae.mean():.3f} cm/s。少数動画からの補助評価であり、有意差検定は行っていない。\n'
    text+='\n## 同じ画像ペアの重複に対する確認\n\n元CSVの行は削除せず主解析に含めた。ただし追加5動画には同じ画像ペアの追加行が50行ある（GX010165で49行、GX010164の12秒区間で1行）。これらを独立した画像ペアとは扱えないため、同じ画像ペアの手動速度を中央値にまとめた補助評価も実施した。\n\n'
    us=unique_summary.loc[unique_summary.direction.eq('right')].groupby('quantile',as_index=False).agg(macro_mae=('mae','mean'),n=('n','sum'))
    text+=mdtable(us,[('quantile','分位点'),('n','右向きユニークペア数'),('macro_mae','動画等重みMAE')])
    text+=f'\n\n重複をまとめた右向きの動画等重みMAE最小は{us.loc[us.macro_mae.idxmin(),"quantile"]*100:g}％。\n'
    text+='''
## 限界

- 手動結果は比較対象であり誤差のない真値とは限らない。
- 右向き分位点だけの感度分析であり、左向きの最適化を行ったわけではない。ただし手動では左向き・低速でも自動が右向きと判断した点には分位点変更が影響する。
- 測定点の少ない動画では最小値が不安定。時間的に近い測定点は独立ではない。
- 動画ごとに最小設定を後から選んだ場合の誤差は楽観的であり、汎用設定の評価とは分ける。
- 時系列比較図の折れ線は手動測定の存在する時刻だけを結び、0.05秒超の空白では線を切っている。間の時刻を新たに測定・補間したものではない。
- 追加対象とは別のGX010164（12秒切り出しでないフォルダ）の手動14点は、今回指定した5動画には含めていない。

## ファイル

- `per_video_sensitivity.png`：6動画の感度分析。
- `aggregate_sensitivity.png`：追加5動画と全6動画の誤差・採用率。
- `manual_comparison/`：各動画の手動と自動の時系列図。
- `per_video_summary.csv`：動画別・方向別・選別別の全指標。
- `aggregate_summary.csv`：追加5動画・全6動画の集計。
- `best_by_video.csv`：動画ごとの75・80％と最小値。
- `leave_one_video_out.csv`：動画を1本ずつ保留した評価。
- `predictions.csv`：全15715推定。
- `pair_manifest.csv`：追加972組の画像対応・校正値。
- `unique_pair_summary.csv`：同じ画像ペアを中央値にまとめた補助評価。
- `matches_manual_pairs.jsonl`：局所変位・重みの再利用用キャッシュ。
- `run_metadata.json`：件数と元ファイルのハッシュ。
'''
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8')
    sources=[ROOT/'all_manual_quantile_sensitivity.py',ROOT/'auto_pair_benchmark.py',PIPE/'calibration_map.json',OUT/'pair_manifest.csv',ROOT/'GX010202_quantile_sensitivity_20261027'/'predictions.csv']
    metadata=dict(date='2026-10-27',fps=240,quantiles=[.7,.725,.75,.775,.8,.825,.85],negative_quantile=.25,
        manual_pairs=2245,additional_pairs=972,predictions=len(p),missing_predictions=int(p.predicted_cm_s.isna().sum()),
        image_resolution_modes=manifest.before_resolution.value_counts().to_dict(),
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    (OUT/'run_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(bests.to_string(index=False))
    print('\nAGGREGATE\n',mainagg.to_string(index=False))
    print('\nLOVO\n',cv.to_string(index=False))
    print('REPORT',OUT/'RESULTS.md')


if __name__=='__main__': main()
