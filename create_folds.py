import os, cv2
from tqdm import tqdm 
import pandas as pd 
from sklearn.model_selection import GroupKFold
import numpy as np 

def compute_distance(df, tr_tracking, merge_col="datetime", use_cols=["x_position", "y_position"]):
    """
    Merges tracking data on player1 and 2 and computes the distance.
    """
    df_combo = (
        df.astype({"nfl_player_id_1": "str"})
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id",] + use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_1"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .rename(columns={c: c+"_1" for c in use_cols})
        .drop("nfl_player_id", axis=1)
        .merge(
            tr_tracking.astype({"nfl_player_id": "str"})[
                ["game_play", merge_col, "nfl_player_id"] +  use_cols
            ],
            left_on=["game_play", merge_col, "nfl_player_id_2"],
            right_on=["game_play", merge_col, "nfl_player_id"],
            how="left",
        )
        .drop("nfl_player_id", axis=1)
        .rename(columns={c: c+"_2" for c in use_cols})
        .copy()
    )

    df_combo["distance"] = np.sqrt(
        np.square(df_combo["x_position_1"] - df_combo["x_position_2"])
        + np.square(df_combo["y_position_1"] - df_combo["y_position_2"])
    )
    return df_combo

use_cols = [
    'x_position', 'y_position', 'speed', 'distance',
    'direction', 'orientation', 'acceleration', 'sa'
]

df_l = pd.read_csv('data/train_labels.csv')
tr_tracking = pd.read_csv("data/train_player_tracking.csv")
df_l = compute_distance(df_l, tr_tracking, use_cols=use_cols)


df_m = pd.read_csv('data/train_video_metadata.csv')
df_m = df_m[df_m.view=='Endzone'][['game_play', 'start_time']]
df = df_l.merge(df_m, on=['game_play'])

df['datetime'] = pd.to_datetime(df["datetime"], utc=True)
df['start_time'] = pd.to_datetime(df["start_time"], utc=True)

df['frame'] = (df['datetime'] - df['start_time'] - pd.to_timedelta(50, "ms")).astype('timedelta64[ms]')*59.94/1000

print(df.head())
print(df.shape)
print(df.columns)

df['game_key'] = df['game_play'].apply(lambda x: x.split('_')[0])
# print(df.game_key.value_counts())

df['fold'] = -1
group_kfold = GroupKFold(n_splits=5)
for fold_id, (train_index, val_index) in enumerate(group_kfold.split(df, df, df.game_key.values)):
    df.iloc[val_index, -1] = fold_id

df = df[['contact_id', 'nfl_player_id_1',
       'nfl_player_id_2', 'x_position_1', 'y_position_1', 'speed_1',
       'distance_1', 'direction_1', 'orientation_1', 'acceleration_1', 'sa_1',
       'x_position_2', 'y_position_2', 'speed_2', 'distance_2', 'direction_2',
       'orientation_2', 'acceleration_2', 'sa_2', 'contact', 'frame', 'distance', 'fold']]

df.to_csv('data/train_folds.csv', index=False, float_format='%.3f')

for i in [0,1,2,3,4]:
    print(df[df.fold==i].contact.value_counts())
