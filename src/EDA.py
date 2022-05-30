import numpy as np
import pandas as pd
import os
import soundfile as sf

from pathlib import Path

def pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320):
    # Show more than 10 or 20 rows when a dataframe comes back.
    pd.set_option('display.max_rows', max_rows)
    # Columns displayed in debug view
    pd.set_option('display.max_columns', max_columns)
    pd.set_option('display.width', display_width)

pd_set_df_view_options(max_rows=1000, max_columns=350, display_width=320)

tra_metadata_more_generated = True
df_label_counts_generated = False

root_dir = Path.cwd().parent
CSV_TRA_META = str(root_dir.joinpath(f'./input/birdclef-2021/train_metadata.csv'))
CSV_TRA_SOUNDSCAPE = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscape_labels.csv'))
CSV_SAMPLE_SUB = str(root_dir.joinpath(f'./input/birdclef-2021/sample_submission.csv'))
CSV_TEST = str(root_dir.joinpath(f'./input/birdclef-2021/test.csv'))
DIR_TRA_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscapes/'))
DIR_TRA_SHORT_AUDIO = str(root_dir.joinpath(f'./input/birdclef-2021/train_short_audio/'))
DIR_TEST_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/test_soundscapes/'))


if not tra_metadata_more_generated:
    # Generate 'duration', 'sample_rate', 'channels' columns for train_metadata.csv
    df_tra_meta = pd.read_csv(CSV_TRA_META)
    ogg_files_num = df_tra_meta['filename'].nunique()  # 62874 ogg files

    df_ogg_info = pd.DataFrame(columns=['filename', 'duration', 'sample_rate', 'channels'])
    for dirname, _, filenames in os.walk(DIR_TRA_SHORT_AUDIO):
        for filename in filenames:
            ogg_info = sf.info(os.path.join(dirname, filename))
            df_ogg_info.loc[len(df_ogg_info)] = [filename, ogg_info.duration, ogg_info.samplerate, ogg_info.channels]

    df_tra_meta = df_tra_meta.merge(df_ogg_info, how='left', on='filename')
    df_tra_meta.to_csv('./train_metadata_more.csv', index=False)
else:
    train_metadata_more_path = str(root_dir.joinpath('./EDA/train_metadata_more.csv'))
    df_tra_meta = pd.read_csv(train_metadata_more_path)
    print(f"df_tra_meta has {len(df_tra_meta)} rows, {df_tra_meta['primary_label'].nunique()} primary_labels")
    pri_labels = df_tra_meta['primary_label'].unique()
    """
    print(df_tra_meta['rating'].value_counts())
        4.0    14393
        5.0    13410
        3.5    10660
        4.5    10423
        3.0     5009
        2.5     3484
        0.0     3334
        2.0     1121
        1.5      674
        1.0      212
        0.5      154
    print(df_tra_meta['duration'].describe())
        count    62874.000000
        mean        56.255305
        std         74.042363
        min          5.958125
        25%         18.378250
        50%         34.260688
        75%         66.205000
        max       2745.352937
    """

    df_less_than30s = df_tra_meta[df_tra_meta['duration'] <= 30.0]
    print(f"df_less_than30s has {len(df_less_than30s)} rows, {df_less_than30s['primary_label'].nunique()} primary_labels")

    if not df_label_counts_generated:
        # primary_label EDA, generate df_label_counts.csv
        df_label_counts = pd.DataFrame(df_tra_meta['primary_label'].value_counts())
        df_label_counts.columns = ['all_pri_counts']
        df_30s_counts = pd.DataFrame(df_less_than30s['primary_label'].value_counts())
        df_30s_counts.columns = ['30s_pri_counts']

        df_label_counts = df_label_counts.merge(df_30s_counts, left_index=True, right_index=True)
        df_label_counts['30s_pri_ratio'] = df_label_counts['30s_pri_counts']/df_label_counts['all_pri_counts']
        # print(f"df_label_counts['30s_pri_ratio'].describe() is:\n{df_label_counts['30s_pri_ratio'].describe()}")

        # secondary_labels EDA
        df_sec_labels = pd.DataFrame(columns=['sec_labels'])
        arr_sec_labels_all = df_tra_meta[df_tra_meta['secondary_labels'] != '[]']['secondary_labels'].to_numpy()
        for sec_labels in arr_sec_labels_all:
            sec_labels = eval(sec_labels)
            for s_label in sec_labels:
                s_label = 'rocpig' if s_label == 'rocpig1' else s_label  # sec label 'rocpig1' is actually 'rocpig'
                df_sec_labels.loc[len(df_sec_labels)] = s_label
        df_sec_label_counts_all = pd.DataFrame(df_sec_labels['sec_labels'].value_counts())

        df_label_counts['all_sec_counts'] = 0
        for idx, row in df_sec_label_counts_all.iterrows():
            df_label_counts.loc[idx, 'all_sec_counts'] = row[0]
        assert (df_label_counts['all_sec_counts'] > 0).sum() == len(df_sec_label_counts_all)  # 393 rows out of 397

        df_sec_labels = df_sec_labels.iloc[0:0]
        arr_sec_labels_30s = df_less_than30s[df_less_than30s['secondary_labels'] != '[]']['secondary_labels'].to_numpy()
        for sec_labels in arr_sec_labels_30s:
            sec_labels = eval(sec_labels)
            for s_label in sec_labels:
                s_label = 'rocpig' if s_label == 'rocpig1' else s_label  # sec label 'rocpig1' is actually 'rocpig'
                df_sec_labels.loc[len(df_sec_labels)] = s_label
        df_sec_label_counts_30s = pd.DataFrame(df_sec_labels['sec_labels'].value_counts())

        df_label_counts['30s_sec_counts'] = 0
        for idx, row in df_sec_label_counts_30s.iterrows():
            df_label_counts.loc[idx, '30s_sec_counts'] = row[0]
        assert (df_label_counts['30s_sec_counts'] > 0).sum() == len(df_sec_label_counts_30s)  # 379 rows out of 397

        df_label_counts.to_csv('./df_label_counts.csv')
    else:
        df_label_counts = pd.read_csv('../EDA/df_label_counts.csv')
        print(f"df_label_counts is:\n{df_label_counts.describe()}")

## EDA for train_soundscape_labels.csv
df_tra_soundscape = pd.read_csv(CSV_TRA_SOUNDSCAPE)
df_soundscape_labels = pd.DataFrame(columns=['birds'])
for birds in df_tra_soundscape['birds'].to_numpy():
    birds = birds.split()
    for bird in birds:
        df_soundscape_labels.loc[len(df_soundscape_labels)] = bird
df_soundscape_label_counts = pd.DataFrame(df_soundscape_labels['birds'].value_counts())
df_soundscape_label_counts.to_csv('./df_soundscape_label_counts.csv')
# print(f"df_soundscape_label_counts is {df_soundscape_label_counts}")
# Merge to df_label_counts.csv
df_label_counts['tra_soundscape_labels'] = 0
for idx, row in df_soundscape_label_counts.iterrows():
    df_label_counts.loc[idx, 'tra_soundscape_labels'] = row[0]
df_label_counts.to_csv('./df_label_counts.csv')
