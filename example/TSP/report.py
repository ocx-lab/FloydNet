import os
import sys
import json
from pathlib import Path

import pandas as pd

seed = 42

def calc_downsampled_oracle(df, downsample):
    """
    Downsample the results for each gid and calculate if any of the predictions is best.
    :param df: DataFrame containing the results
    :param downsample: int, downsampling factor
    :return: DataFrame with downsampled results aggregated by nodes
    """
    records = []
    for gid, group in df.groupby('gid'):
        n = min(len(group), downsample)
        sampled = group.sample(n=n, random_state=seed)
        any_best = bool(sampled['is_best'].any())
        any_valid = bool(sampled['is_valid'].any())
        nodes = int(group['nodes'].iloc[0])
        records.append({
            'gid': gid,
            'nodes': nodes,
            'is_best_oracle': int(any_best),
            'is_valid_oracle': int(any_valid)
        })

    oracle_df = pd.DataFrame(records)
    agg_df = oracle_df.groupby('nodes').agg(
        valid_ratio=('is_valid_oracle', 'mean'),
        best_ratio=('is_best_oracle', 'mean'),
    ).reset_index()
    return agg_df

def main():
    result_folder = Path(sys.argv[1])
    result_list = []
    for file in result_folder.rglob('*.json'):
        with open(file, 'r') as f:
            data = json.load(f)
            result_list.extend(data)
    
    # a list of dicts contains:
    # gid: str, graph id
    # nodes: int, number of nodes in graph
    # gt_len: int, ground truth length
    # pred_len: int, predicted length
    # is_valid: bool, whether the prediction is valid
    # is_best: bool, whether the prediction is the best
    # f1_score: float, F1 score of the prediction
    df = pd.DataFrame(result_list)
    df['valid'] = df['is_valid'].astype(int)
    df['best'] = df['is_best'].astype(int)

    # print number of rows for each gid
    print(f'Number of results: {len(df)}')
    print(f'Number of unique gids: {df["gid"].nunique()}')
    print(f'Average number of rows per gid: {df.groupby("gid").size().mean():.2f}')

    # for each gid, downsample results and get if any of the predictions is best
    for downsample in [1, 5, 10]:
        new_df = calc_downsampled_oracle(df, downsample)
        save_path = result_folder / f'result_downsample_{downsample}.csv'
        new_df.to_csv(save_path, index=False)
        print(f'Saved downsampled results to {save_path}')

if __name__ == '__main__':
    main()