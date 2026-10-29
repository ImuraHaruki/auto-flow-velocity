import math
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from auto_pair_benchmark import candidate_centers, estimate_from_matches, match_candidates
from benchmark_tracking import grayscale, local_match
from plot_tracking_candidates_full_frame import BEFORE_PATH, AFTER_PATH, OUTPUT_DIR, RESULT_DIR, set_font

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'manual_series_combined_20260831'))
from evaluate_temporal_filter import temporal_filter_cluster
QUANTILE = .75


def main():
    set_font()
    a, b = grayscale(BEFORE_PATH), grayscale(AFTER_PATH)
    matches, centers = [], []
    for x, y, activity in candidate_centers(a, b, 32, 80):
        try:
            dx, dy, corr, margin = local_match(a, b, (x, y), 32, search_x=150, search_y=55)
            rx, ry, _, _ = local_match(b, a, (x+dx, y+dy), 32, search_x=150, search_y=55)
        except ValueError:
            continue
        if math.hypot(dx+rx, dy+ry) > 3 or margin < .015:
            continue
        matches.append((dx, max(margin,.001)*max(corr+1,.05)*max(activity,1e-5)))
        centers.append((x,y))
    result = estimate_from_matches(matches, positive_quantile=QUANTILE, negative_quantile=.25)
    dxs = np.array([m[0] for m in matches])
    df = pd.read_csv(RESULT_DIR / 'per_sequence' / 'GX010202_timestamp_JST_velocity_step1_temporal_v2.csv')
    sample = df.loc[df.before_frame.eq(1083)].iloc[0]
    t = sample.time_s
    # Recalculate every displayed measurement plus a 30-frame margin with
    # q=.75. Preserve the original image intervals, not its q=.85 velocities.
    padded = df.loc[df.time_s.between(t-.63,t+.63)].copy()
    cache = {}
    cache_path = ROOT / 'GX010202_quantile_sensitivity_20261008' / 'matches_manual_pairs.jsonl'
    if cache_path.exists():
        for line in cache_path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            cache[(item['before_file'], item['after_file'])] = item['matches']
    print(f'Recalculating {len(padded)} surrounding pairs at q={QUANTILE}',flush=True)
    def recalculate(row):
        key = (row.before_file,row.after_file)
        local_matches = cache.get(key)
        if local_matches is None:
            local_matches = match_candidates(grayscale(BEFORE_PATH.parent/key[0]),grayscale(BEFORE_PATH.parent/key[1]),radius=32,candidate_count=80)
        estimate = estimate_from_matches(local_matches,positive_quantile=QUANTILE,negative_quantile=.25)
        velocity = np.nan if estimate is None else estimate['dx']*row.cm_per_pixel/row.delta_time_s
        return dict(before_frame=row.before_frame,after_frame=row.after_frame,before_file=row.before_file,after_file=row.after_file,
                    time_s=row.time_s,frame_gap=row.frame_gap,raw_velocity_cm_s=velocity,
                    confidence=np.nan if estimate is None else estimate['confidence'])
    with ThreadPoolExecutor(max_workers=4) as pool:
        recalculated = pd.DataFrame(pool.map(recalculate, padded.itertuples(index=False)))
    filtered = temporal_filter_cluster(recalculated,window=30,zero_abs=3.,ref_motion=20.,rescue_conf=.75,rescue_abs=10.,min_count=25,dominance=.75)
    filtered['velocity_final_cm_s'] = filtered.velocity_temporal_cm_s
    filtered['final_status'] = np.where(filtered.velocity_final_cm_s.notna(),'measured','unmeasurable')
    near = filtered.loc[filtered.time_s.between(t-.5,t+.5)].copy()
    sample = near.loc[near.before_frame.eq(1083)].iloc[0]
    assert np.isclose(result['dx']*float(padded.cm_per_pixel.iloc[0])*240,sample.raw_velocity_cm_s,atol=1e-9)
    filtered.to_csv(OUTPUT_DIR/'GX010202_sample_q75_temporal.csv',index=False,encoding='utf-8-sig')
    print(sample[['time_s','raw_velocity_cm_s','velocity_final_cm_s','confidence','temporal_reference_cm_s','temporal_status']].to_string())
    print('representative_dx',result['dx'])
    print('positive',sum(dxs>=2),'negative',sum(dxs<=-2),'near_zero',sum(abs(dxs)<2))
    fig, (ax, bx) = plt.subplots(2,1,figsize=(16,10),constrained_layout=True)
    fig.suptitle('GX010202：局所変位から代表流速を求める実測例（右向き75％）',fontsize=18,weight='bold')
    weights = np.array([m[1] for m in matches])
    order = np.argsort(dxs, kind='stable')
    dxs, weights = dxs[order], weights[order]
    inds = np.arange(1,len(dxs)+1)
    pos, neg, zero = dxs>=2, dxs<=-2, abs(dxs)<2
    for mask,color,label in [(pos,'#3979F6','右向き（%d領域）'%sum(pos)),(neg,'#F19A38','左向き（%d領域）'%sum(neg)),(zero,'#A8AFBA','微小変位（%d領域）'%sum(zero))]:
        ax.scatter(inds[mask],dxs[mask],s=35,color=color,alpha=.8,edgecolors='none',label=label)
    ax.axhline(result['dx'],color='#E62727',lw=2,label=f"代表変位 Δx = {result['dx']:.2f} px")
    ax.axhline(0,color='#727B88',lw=.8)
    ax.set_title('④ 照合を通過した51領域の変位から、代表変位を選ぶ',fontsize=15)
    ax.set_xlabel('変位の小さい順に並べた照合通過領域の番号')
    ax.set_ylabel('水平方向の変位 Δx（px）')
    ax.set_ylim(bottom=-20)
    direction_legend = ax.legend(fontsize=9,loc='lower left',ncol=2,scatterpoints=1,markerscale=1)
    ax.add_artist(direction_legend)
    # Only the rightward population contributes to its weighted quantile.
    # Keep other populations visible, but do not include them in the denominator.
    cumulative = np.cumsum(np.where(pos, weights, 0.0)) / weights[pos].sum() * 100
    cx = ax.twinx()
    cx.plot(inds,cumulative,color='#8E44AD',lw=2,marker='.',markersize=4,label='右向き領域の累積重み')
    cx.axhline(QUANTILE*100,color='#8E44AD',lw=1,ls='--',label='累積75％')
    selected_index = int(np.searchsorted(cumulative, QUANTILE*100))
    ax.axvline(inds[selected_index],color='#E62727',lw=1,ls=':')
    # Match the zero height on both axes while leaving room above 100%.
    left_bottom, left_top = ax.get_ylim()
    cx.set_ylim(105 * left_bottom / left_top, 105)
    cx.set_yticks([0,25,50,75,100])
    cx.set_ylabel('右向き領域の累積重み（％）',color='#8E44AD')
    cx.tick_params(axis='y',colors='#8E44AD')
    cx.legend(loc='upper left',fontsize=9)
    ok = near.final_status.eq('measured')
    bx.scatter(near.loc[ok,'time_s'],near.loc[ok,'velocity_final_cm_s'],s=20,color='#3979F6',label='最終採用')
    bx.scatter(near.loc[~ok,'time_s'],near.loc[~ok,'raw_velocity_cm_s'],s=35,color='#E62727',marker='x',label='最終棄却（生値）')
    bx.plot(near.time_s,near.temporal_reference_cm_s,color='#27AE60',lw=2,label='周辺時刻の代表傾向')
    bx.axvspan(t-.125,t+.125,color='#27AE60',alpha=.08,label='対象時刻の前後0.125秒')
    bx.axvline(t,color='#E62727',lw=1,ls='--',label=f'対象ペア：{sample.velocity_final_cm_s:.2f} cm/s')
    bx.set_title('⑤ 周辺時刻の傾向と照合し、最終的な採否を確認する',fontsize=15)
    bx.set_xlabel('系列開始からの時刻（s）')
    bx.set_ylabel('流速（cm/s）')
    bx.legend(fontsize=9,loc='best',ncol=2)
    for axis in (ax,bx):
        axis.grid(alpha=.25)
        axis.set_axisbelow(True)
    out = OUTPUT_DIR / '02_GX010202_current_sample_consensus_temporal_q75_ja_cumulative.png'
    fig.savefig(out,dpi=180,facecolor='white')
    print(out)


if __name__=='__main__':
    main()
