"""Finalize both policies, preserve archive, and compare on identical start frames."""
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_all_q075_q025 import OUT,OLD,ROOT,PARAMS,state,digest,pause_requested,PauseRequested
sys.path.insert(0,str(ROOT/'manual_series_combined_20260831'))
from evaluate_temporal_filter import temporal_filter_cluster
from benchmark_tracking import image_resolver,trailing_number


def save(frame,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    frame.to_csv(tmp,index=False,encoding='utf-8-sig'); tmp.replace(path)


def load_raw():
    manifest=json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    expected={(d,i) for d,v in manifest.items() for i in range(len(v['files'])-1)}
    db=sqlite3.connect(OUT/'checkpoint.sqlite')
    keys=set(db.execute('SELECT dataset,start FROM results'))
    assert keys==expected, f'Incomplete: {len(keys)}/{len(expected)}; do not finalize partial results.'
    new,old=[],[]
    for (payload,) in db.execute('SELECT record FROM results ORDER BY dataset,start'):
        r=json.loads(payload); new.append(r['new']); old.append(r['old_reproduced'])
    db.close()
    return pd.DataFrame(new),pd.DataFrame(old),manifest


def temporal(frame):
    groups=[]
    for d,g in frame.groupby('dataset',sort=True):
        r=temporal_filter_cluster(g,**PARAMS['temporal'])
        r['dataset']=d; groups.append(r)
    result=pd.concat(groups,ignore_index=True)
    result['velocity_final_cm_s']=result.velocity_temporal_cm_s
    result['final_status']=np.where(result.velocity_final_cm_s.notna(),'measured','unmeasurable')
    result['fps']=240.; result['confidence_threshold']=.85
    return result.sort_values(['dataset','before_frame']).reset_index(drop=True)


def manual_alignment(manifest):
    manual=pd.read_csv(OUT/'baseline_archive'/'manual_measurements_combined.csv')
    records=[]
    for d,g in manual.groupby('dataset'):
        if d not in manifest: continue
        info=manifest[d]
        files=[Path(info['folder'])/name for name in info['files']]
        resolver=image_resolver(files); index={p.name:i for i,p in enumerate(files)}
        for rownum,r in g.iterrows():
            before=resolver(str(r.before_file),str(r.before_frame))
            after=resolver(str(r.after_file),str(r.after_frame))
            raw=Path(str(r.before_file)).name
            mode='exact_name' if raw.casefold()==before.name.casefold() else 'numeric_name'
            if mode!='exact_name':
                a,b=trailing_number(raw),trailing_number(before.name)
                assert a is not None and b is not None and a%10000==b%10000,('Ambiguous manual image mapping',d,rownum)
            assert index[after.name]>index[before.name]
            records.append(dict(manual_row=int(rownum),dataset=d,before_frame=index[before.name]+1,
                before_file=before.name,manual_after_file=after.name,manual_velocity_cm_s=float(r.velocity),
                manual_gap=index[after.name]-index[before.name],time_s=index[before.name]/240,
                mapping=mode))
    frame=pd.DataFrame(records)
    frame['direction']=np.select([frame.manual_velocity_cm_s>10,frame.manual_velocity_cm_s < -10],['right','left'],default='slow')
    save(frame,OUT/'comparison'/'manual_alignment.csv')
    return frame


def paired(new,old,archive):
    cols=['dataset','before_frame','before_file','after_frame','frame_gap','raw_velocity_cm_s',
          'velocity_final_cm_s','confidence','final_status','temporal_status']
    common=new[cols].merge(old[cols],on=['dataset','before_frame','before_file'],suffixes=('_new','_old_reproduced'),validate='one_to_one')
    common['raw_delta_cm_s']=common.raw_velocity_cm_s_new-common.raw_velocity_cm_s_old_reproduced
    common['final_delta_cm_s']=common.velocity_final_cm_s_new-common.velocity_final_cm_s_old_reproduced
    common['new_accepted']=common.velocity_final_cm_s_new.notna()
    common['old_accepted']=common.velocity_final_cm_s_old_reproduced.notna()
    common['gap_changed']=common.frame_gap_new.ne(common.frame_gap_old_reproduced)
    save(common,OUT/'comparison'/'new_vs_old_reproduced.csv')
    historical=new[cols].merge(archive[cols],on=['dataset','before_frame','before_file'],suffixes=('_new','_archived'),validate='one_to_one')
    assert len(historical)==len(archive),'Historical start frames no longer align with the input manifest.'
    historical['raw_delta_cm_s']=historical.raw_velocity_cm_s_new-historical.raw_velocity_cm_s_archived
    historical['final_delta_cm_s']=historical.velocity_final_cm_s_new-historical.velocity_final_cm_s_archived
    save(historical,OUT/'comparison'/'new_vs_archived.csv')
    reproduced=old[cols].merge(archive[cols],on=['dataset','before_frame','before_file'],suffixes=('_reproduced','_archived'),validate='one_to_one')
    reproduced['raw_abs_difference_cm_s']=(reproduced.raw_velocity_cm_s_reproduced-reproduced.raw_velocity_cm_s_archived).abs()
    reproduced['raw_mismatch']=~np.isclose(reproduced.raw_velocity_cm_s_reproduced,reproduced.raw_velocity_cm_s_archived,atol=1e-6,rtol=1e-9,equal_nan=True)
    reproduced['final_mismatch']=~np.isclose(reproduced.velocity_final_cm_s_reproduced,reproduced.velocity_final_cm_s_archived,atol=1e-6,rtol=1e-9,equal_nan=True)
    save(reproduced,OUT/'comparison'/'old_reproduction_check.csv')
    summaries=[]
    for d,g in common.groupby('dataset'):
        both=g.new_accepted & g.old_accepted
        summaries.append(dict(dataset=d,evaluated=len(g),new_accepted=int(g.new_accepted.sum()),
            old_reproduced_accepted=int(g.old_accepted.sum()),new_acceptance_rate=g.new_accepted.mean(),
            old_acceptance_rate=g.old_accepted.mean(),gained=int((g.new_accepted&~g.old_accepted).sum()),
            lost=int((~g.new_accepted&g.old_accepted).sum()),common_accepted=int(both.sum()),
            mean_abs_raw_change_cm_s=g.raw_delta_cm_s.abs().mean(),
            mean_abs_common_final_change_cm_s=g.loc[both,'final_delta_cm_s'].abs().mean(),
            frame_gap_changes=int(g.gap_changed.sum()),
            archived_data_available=d in set(archive.dataset)))
    summary=pd.DataFrame(summaries)
    save(summary,OUT/'comparison'/'per_sequence_comparison.csv')
    return common,summary,reproduced


def accuracy(manual,new,old,archive):
    aligned=manual.copy()
    for name,frame in [('new075',new),('old085_reproduced',old),('old_archived',archive)]:
        cols=['dataset','before_frame','raw_velocity_cm_s','velocity_final_cm_s','frame_gap']
        f=frame[cols].rename(columns={c:name+'_'+c for c in cols if c not in ('dataset','before_frame')})
        aligned=aligned.merge(f,on=['dataset','before_frame'],how='left',validate='many_to_one')
    aligned['common_accepted']=aligned.new075_velocity_final_cm_s.notna() & aligned.old085_reproduced_velocity_final_cm_s.notna()
    save(aligned,OUT/'comparison'/'manual_comparison.csv')
    rows=[]
    for name in ['new075','old085_reproduced','old_archived']:
        for d in ['ALL',*sorted(aligned.dataset.unique())]:
            group=aligned if d=='ALL' else aligned.loc[aligned.dataset.eq(d)]
            for direction in ['all','right','left','slow']:
                sub=group if direction=='all' else group.loc[group.direction.eq(direction)]
                for mode,field in [('raw','raw_velocity_cm_s'),('final','velocity_final_cm_s'),('common_final','velocity_final_cm_s')]:
                    valid=sub.loc[sub[name+'_'+field].notna()]
                    if mode=='common_final': valid=valid.loc[valid.common_accepted]
                    e=valid[name+'_'+field]-valid.manual_velocity_cm_s
                    rows.append(dict(policy=name,dataset=d,direction=direction,evaluation=mode,total=len(sub),n=len(valid),
                        coverage=len(valid)/len(sub) if len(sub) else np.nan,mae=e.abs().mean(),
                        rmse=np.sqrt((e**2).mean()),bias=e.mean(),median_ae=e.abs().median()))
    metrics=pd.DataFrame(rows)
    save(metrics,OUT/'comparison'/'manual_accuracy_summary.csv')
    return aligned,metrics


def plots(new,old,archive,manual,summary,metrics):
    plt.rcParams['font.family']='Meiryo'; plt.rcParams['axes.unicode_minus']=False
    out=OUT/'comparison'/'graphs'; out.mkdir(parents=True,exist_ok=True)
    for n,(d,g) in enumerate(new.groupby('dataset'),1):
        if pause_requested(): raise PauseRequested('Paused between comparison graphs.')
        o=old.loc[old.dataset.eq(d)]
        previous=archive.loc[archive.dataset.eq(d)]
        fig,(ax,bx)=plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':[2,1]},constrained_layout=True)
        if len(previous): ax.plot(previous.time_s,previous.velocity_final_cm_s,color='#8792A2',lw=.8,alpha=.5,label='過去の保存結果')
        ax.plot(o.time_s,o.velocity_final_cm_s,color='#3478BA',lw=1,label='従来基準：右85％・左25％（再計算）')
        ax.plot(g.time_s,g.velocity_final_cm_s,color='#E07A21',lw=1,label='新基準：右75％・左25％')
        m=manual.loc[manual.dataset.eq(d)]
        if len(m): ax.scatter(m.time_s,m.manual_velocity_cm_s,s=7,color='#C32630',alpha=.7,label='手動測定')
        ax.set_title(d.replace('_timestamp_JST','')+'：新旧基準の比較',fontsize=16,weight='bold')
        ax.set_ylabel('最終採用流速（cm/s）'); ax.legend(fontsize=9)
        pair=g[['before_frame','time_s','raw_velocity_cm_s','velocity_final_cm_s']].merge(
            o[['before_frame','raw_velocity_cm_s','velocity_final_cm_s']],on='before_frame',suffixes=('_new','_old'))
        bx.scatter(pair.time_s,pair.raw_velocity_cm_s_new-pair.raw_velocity_cm_s_old,s=2,alpha=.25,color='#8792A2',label='選別前の差')
        bx.scatter(pair.time_s,pair.velocity_final_cm_s_new-pair.velocity_final_cm_s_old,s=3,alpha=.6,color='#E07A21',label='両方で採用された点の差')
        bx.set_ylabel('新基準－従来基準（cm/s）'); bx.set_xlabel('系列開始からの時刻（s）'); bx.legend(fontsize=9)
        for axis in (ax,bx): axis.axhline(0,color='gray',lw=.7); axis.grid(alpha=.25)
        fig.savefig(out/(d+'_comparison.png'),dpi=150); plt.close(fig)
        if n==1 or n%10==0 or n==new.dataset.nunique(): state('plotting',graphs_completed=n,graphs_total=int(new.dataset.nunique()))
    fig,axes=plt.subplots(2,1,figsize=(18,10),constrained_layout=True)
    x=np.arange(len(summary)); labels=summary.dataset.str.replace('_timestamp_JST','',regex=False)
    axes[0].bar(x-.2,summary.old_acceptance_rate*100,width=.4,label='従来85％（再計算）',color='#3478BA')
    axes[0].bar(x+.2,summary.new_acceptance_rate*100,width=.4,label='新75％',color='#E07A21')
    axes[0].set_ylabel('採用率（％）'); axes[0].legend()
    axes[1].bar(x,summary.mean_abs_common_final_change_cm_s,color='#E07A21')
    axes[1].set_ylabel('共通採用点の平均絶対変化（cm/s）')
    for axis in axes:
        axis.set_xticks(x,labels,rotation=90,fontsize=8); axis.grid(axis='y',alpha=.25)
    fig.suptitle('全動画の新旧比較（左向き25％・時間フィルタ条件は共通）',fontsize=17,weight='bold')
    fig.savefig(OUT/'comparison'/'overview_comparison.png',dpi=170); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),constrained_layout=True)
    for mode,label in [('raw','選別前'),('final','最終採用'),('common_final','共通採用点')]:
        vals=[]
        for policy in ['old_archived','old085_reproduced','new075']:
            row=metrics.loc[metrics.dataset.eq('ALL')&metrics.direction.eq('all')&metrics.evaluation.eq(mode)&metrics.policy.eq(policy)].iloc[0]
            vals.append(row.mae)
        axes[0].plot(['過去の保存結果','従来85％再計算','新75％'],vals,'o-',label=label)
    axes[0].set_ylabel('手動測定とのMAE（cm/s）'); axes[0].legend(); axes[0].grid(alpha=.25)
    for policy,label in [('old085_reproduced','従来85％再計算'),('new075','新75％')]:
        rows=metrics.loc[metrics.dataset.eq('ALL')&metrics.evaluation.eq('final')&metrics.policy.eq(policy)].set_index('direction')
        axes[1].plot(['右向き','左向き','低速'],[rows.loc[d,'mae'] for d in ['right','left','slow']],'o-',label=label)
    axes[1].set_ylabel('最終採用点のMAE（cm/s）'); axes[1].legend(); axes[1].grid(alpha=.25)
    fig.suptitle('手動測定を基準とした新旧比較',fontsize=17,weight='bold')
    fig.savefig(OUT/'comparison'/'manual_accuracy_comparison.png',dpi=170); plt.close(fig)


def main():
    try:
        state('finalizing')
        new_raw,old_raw,manifest=load_raw()
        archive=pd.read_csv(OUT/'baseline_archive'/'all_patterns_velocity_step1_temporal_v2.csv')
        save(new_raw,OUT/'raw'/'new_q075_q025_raw.csv')
        save(old_raw,OUT/'raw'/'old_q085_q025_reproduced_raw.csv')
        new,old=temporal(new_raw),temporal(old_raw)
        for policy,frame in [('new_q075_q025',new),('old_q085_q025_reproduced',old)]:
            save(frame,OUT/(policy+'_velocity_step1_temporal_v2.csv'))
            for d,g in frame.groupby('dataset'): save(g,OUT/policy/'per_sequence'/(d+'_velocity.csv'))
        common,summary,reproduced=paired(new,old,archive)
        manual=manual_alignment(manifest)
        aligned,metrics=accuracy(manual,new,old,archive)
        plots(new,old,archive,manual,summary,metrics)
        write_report(new,old,archive,summary,reproduced,metrics,manifest)
        for name in ['all_patterns_velocity_step1_temporal_v2.csv','sequence_summary_step1_temporal_v2.csv','run_metadata.json','calibration_map.json']:
            assert digest(OLD/name)==digest(OUT/'baseline_archive'/name),('Historical file modified during this job',name)
        state('complete',datasets=len(manifest),rows=len(new),new_accepted=int(new.velocity_final_cm_s.notna().sum()),
              old_reproduced_accepted=int(old.velocity_final_cm_s.notna().sum()),
              historical_raw_mismatches=int(reproduced.raw_mismatch.sum()),
              new_csv=str(OUT/'new_q075_q025_velocity_step1_temporal_v2.csv'),report=str(OUT/'comparison'/'RESULTS.md'))
    except PauseRequested:
        state('paused',stage='finalization',message='Saved data retained; resume to regenerate comparison outputs.')
    except BaseException as error:
        state('finalization_failed',error=repr(error)); raise


def write_report(new,old,archive,summary,reproduced,metrics,manifest):
    from datetime import datetime
    rows=['# 全動画：右75％・左25％への変更結果',f'\n作成：{datetime.now().astimezone().isoformat(timespec="seconds")}',
          '\n## 対象・保全',
          f'\n{len(manifest)}動画、{len(new):,}測定開始点。240 fps、1フレーム刻み、最大4フレームの適応間隔。',
          '\n新基準は右75％・左25％。比較用に右85％・左25％も同じ画像照合候補から再計算した。時間整合性フィルタの条件と校正値は共通。',
          '\n過去の50動画の保存結果は一切上書きせず、baseline_archiveにも複製した。追加9動画（GX010214～222）はユーザー承認により後半グループの校正値約0.0107992 cm/pxを使用。追加9動画の従来基準は今回初めて計算した比較用の値である。',
          '\n## 全体の採用率',
          '\n| 結果 | 対象点数 | 採用点数 | 採用率 |','| --- | --- | --- | --- |']
    for label,frame in [('過去の保存結果（50動画）',archive),('従来基準の再計算（59動画）',old),('新基準（59動画）',new)]:
        n=int(frame.velocity_final_cm_s.notna().sum())
        rows.append(f'| {label} | {len(frame):,} | {n:,} | {n/len(frame)*100:.2f}％ |')
    rows+=['\n50動画と59動画の採用率を直接比較して改善を判断しないこと。同じ対象での比較はcomparison/new_vs_archived.csv、およびnew_vs_old_reproduced.csvを用いる。',
           '\n## 過去結果の再現性',
           f'\n過去結果と同じ{len(reproduced):,}開始点について、従来基準の選別前速度の差が許容誤差を超えた点は{int(reproduced.raw_mismatch.sum()):,}点。最終採用値・採否の差は{int(reproduced.final_mismatch.sum()):,}点。',
           '\n差がある場合は分位点変更だけの効果と混同せず、同じ画像照合からの従来基準再計算と新基準の比較を重視する。詳細はold_reproduction_check.csv。',
           '\n## 手動測定との比較',
           '\n元CSVの画像名を実際の画像名に対応させ、同じ測定開始フレームに結合した。適応フレーム間隔は基準ごとに選び直しているため、手動と常に同じ終了画像とは限らない。手動値を真値と断定せず、方向別・共通採用点も確認する。',
           '\n| 結果 | 評価 | 点数 | MAE (cm/s) | RMSE (cm/s) |','| --- | --- | --- | --- | --- |']
    for policy,label in [('old_archived','過去保存'),('old085_reproduced','従来85％再計算'),('new075','新75％')]:
        for mode,title in [('raw','選別前'),('final','最終採用'),('common_final','新旧共通採用')]:
            r=metrics.loc[metrics.policy.eq(policy)&metrics.dataset.eq('ALL')&metrics.direction.eq('all')&metrics.evaluation.eq(mode)].iloc[0]
            rows.append(f'| {label} | {title} | {int(r.n):,} | {r.mae:.3f} | {r.rmse:.3f} |')
    rows+=['\n## 出力',
           '\n- ../new_q075_q025_velocity_step1_temporal_v2.csv：新基準の全動画結果。',
           '- ../new_q075_q025/per_sequence/：新基準の動画別CSV。',
           '- ../old_q085_q025_reproduced_velocity_step1_temporal_v2.csv：比較用の従来基準再計算。',
           '- ../baseline_archive/：過去の保存結果の保全用複製。',
           '- new_vs_old_reproduced.csv：59動画の同じ開始点での新旧比較。',
           '- new_vs_archived.csv：過去50動画の保存結果との比較。',
           '- per_sequence_comparison.csv：動画別の採用率・変化量。',
           '- manual_accuracy_summary.csv：手動測定との方向別・動画別誤差。',
           '- manual_alignment.csv：手動画像名の対応記録。',
           '- graphs/：全59動画の時系列比較図。',
           '- overview_comparison.png：全体の比較図。',
           '- manual_accuracy_comparison.png：手動測定との比較図。',
           '\n計算途中で画像を読めない場合はゼロや低信頼度に置き換えず停止する。checkpoint.sqliteは測定開始点ごとの結果と画像ペアごとの局所変位・重みを保存しており、再開時に混在を防ぐためコード・パラメータ・画像一覧を検証する。']
    (OUT/'comparison'/'RESULTS.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')


if __name__=='__main__': main()
