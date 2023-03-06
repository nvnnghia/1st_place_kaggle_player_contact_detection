from tqdm import tqdm 
import pandas as pd 
import numpy as np 

## cache baseline helmet bboxes
df = pd.read_csv('data/train_baseline_helmets.csv')
df = df.fillna(99)

det_dict = {}
for i, row in tqdm(df.iterrows()):
    vid = row['game_play'] + '_' + row['view']
    frame = row['frame']
    p_id = row['nfl_player_id']
    if vid not in det_dict:
        det_dict[vid] = {}

    if frame not in det_dict[vid]:
        det_dict[vid][frame] = {}

    if p_id not in det_dict[vid][frame]:
        if 'H' in row['player_label']:
            t = 'home'
        else:
            t = 'v'

        det_dict[vid][frame][p_id] = {'box': [row['left'], row['top'], row['width'], row['height']], 'contact': [], 't': t}

np.save('data/det_dict.npy', det_dict)


## cache tracking position
trk_df = pd.read_csv('data/train_player_tracking.csv')
trk_df = trk_df[trk_df.step>-60]
print(trk_df.shape)
trk_dict = {}
for i, row in tqdm(trk_df.iterrows()):
    vid = row['game_play']
    p_id = row['nfl_player_id']
    step = row['step']
    idx = f'{vid}_{step}'
    if idx not in trk_dict:
        trk_dict[idx] = {}

    trk_dict[idx][p_id] = {'x': row['x_position'], 'y': row['y_position'], 't': row['team']}

np.save('data/trk_pos.npy', trk_dict)


## cache tracking metadata
trk_df = pd.read_csv('data/train_player_tracking.csv')
trk_df = trk_df[trk_df.step>-60]
print(trk_df.shape)
trk_dict = {}
for i, row in tqdm(trk_df.iterrows()):
    vid = row['game_play']
    idx = row['nfl_player_id']
    idx = f'{vid}_{idx}'
    step = row['step']
    if idx not in trk_dict:
        trk_dict[idx] = {}

    trk_dict[idx][step] = {'s': row['speed'], 'dis': row['distance'], 'dir': row['direction'], 'o': row['orientation'], 'a': row['acceleration'], 'sa': row['sa'], 'x': row['x_position'], 'y': row['y_position'], 't': row['team']}

np.save('data/trk_dict.npy',trk_dict)