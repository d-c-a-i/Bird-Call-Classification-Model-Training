import torch
import random
import numpy as np
import pandas as pd
import src.utils as utils
import src.datasets as datasets
import src.models as models
import src.tra_val_test as tra_val_test

from pathlib import Path

root_dir = Path.cwd().parent
config = utils.load_config(str(root_dir.joinpath(f'./configs/bootstrap_model_a.yml')))
SAMPLE_RATE = config['spectro']['sample_rate']
CLEAR_DURATION = config['globals']['clear_duration']
STEP_SECS = config['globals']['step_seconds']
EFFECTIVE_SECS = config['spectro']['period']
DIR_TRA_SHORT_AUDIO = str(root_dir.joinpath(f'./input/birdclef-2021/train_short_audio/'))
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

Infer_Model_Path = f"./output/bootstrap_model_a/BootstrapModelA_rexnet_100_fold0.bin"
# f'./output/bootstrap_model_a/bootstrap_model_a_fold0_34s.bin'

hard_labels_dir = Path("./hard_labels_generating/")
hard_labels_dir.mkdir(exist_ok=True, parents=True)
HARD_LABELS_DIR = str(Path.cwd().joinpath(hard_labels_dir))

df_gen_hard_labels_generated = True
df_need_gen_inferenced = True
df_need_gen_generated = True
df_final_gen_labels_generated = False

if not df_gen_hard_labels_generated:
    with utils.timer("Generating df_gen_hard_labels.csv"):
        df_metadata_labels = pd.read_csv('./df_metadata_labels.csv')
        df_gen_hard_labels = pd.DataFrame(columns=['ogg_crop_step', 'ogg_name', 'start_sample', 'rating', 'duration',
                                                   'ogg_pri_label', 'ogg_sec_labels', 'gen_confident_labels'])
        for idx, row in df_metadata_labels.iterrows():
            ogg_name, rating, duration = row['filename'], row['rating'], row['duration']
            ogg_pri_label, ogg_sec_labels = row['primary_label'], row['secondary_labels']
            start_sample, ogg_crop_step, gen_confident_labels = None, None, None

            if len(eval(ogg_sec_labels)) > 0:  # handle wrong 'rocpig1' in 'secondary_labels'
                ogg_sec_labels = np.unique(eval(ogg_sec_labels))
                ogg_sec_labels = np.where(ogg_sec_labels == 'rocpig1', 'rocpig', ogg_sec_labels)
                ogg_sec_labels = utils.labels_to_listlike_str(np.unique(ogg_sec_labels))

            if row['duration'] <= CLEAR_DURATION:
                start_sample = 0
                ogg_crop_step = f"{ogg_name}_0"
                gen_confident_labels = utils.labels_to_space_sep_str(np.unique([ogg_pri_label] + eval(ogg_sec_labels)))
                df_gen_hard_labels.loc[len(df_gen_hard_labels)] = [ogg_crop_step, ogg_name, start_sample, rating,
                                                                   duration, ogg_pri_label, ogg_sec_labels,
                                                                   gen_confident_labels]
            else:
                total_sec = int(np.floor(row['duration']))
                for start_sec in range(0, total_sec, STEP_SECS):
                    if start_sec + EFFECTIVE_SECS > total_sec + STEP_SECS:
                        break

                    ogg_crop_step = f"{ogg_name}_{start_sec}"
                    if start_sec == 0:
                        start_sample = 0
                    else:
                        start_sample = random.randint(start_sec*SAMPLE_RATE,
                                                      (start_sec + EFFECTIVE_SECS - STEP_SECS)*SAMPLE_RATE)
                    gen_confident_labels = None
                    df_gen_hard_labels.loc[len(df_gen_hard_labels)] = [ogg_crop_step, ogg_name, start_sample, rating,
                                                                       duration, ogg_pri_label, ogg_sec_labels,
                                                                       gen_confident_labels]

        df_gen_hard_labels.to_csv(f"{HARD_LABELS_DIR}/df_gen_hard_labels.csv", index=False)

elif not df_need_gen_inferenced:
    df_gen_hard_labels = pd.read_csv(f"{HARD_LABELS_DIR}/df_gen_hard_labels.csv")
    df_not_need_gen = df_gen_hard_labels[df_gen_hard_labels['gen_confident_labels'].notna()].reset_index(drop=True)
    assert(len(df_not_need_gen) == len(df_gen_hard_labels[df_gen_hard_labels['duration'] <= CLEAR_DURATION]))
    df_not_need_gen.to_csv(f'{HARD_LABELS_DIR}/df_not_need_gen.csv', index=False)

    df_need_gen = df_gen_hard_labels[df_gen_hard_labels['gen_confident_labels'].isna()].reset_index(drop=True)
    # df_need_gen = df_need_gen.loc[0:1000]  # For debug

    utils.set_seed(config['globals']['seed'])

    infer_hard_loader = datasets.get_loader('infer_hard', df_need_gen, DIR_TRA_SHORT_AUDIO, config)
    bootstrap_model_a = models.BootstrapModelA(config=config, base_model=config['base_models'][0])
    bootstrap_model_a.load_state_dict(torch.load(Infer_Model_Path, map_location=DEVICE))
    bootstrap_model_a = bootstrap_model_a.to(DEVICE)

    """
    # Generate hard-confident labels strategy:
    - Predicted probability could be rough, so set to hard label(0/1) with more confidence is better
    - Missing labels will be addressed by Pseudo labelling stage, so if found high probability missing label here, mask it is better.
    - If 'gen_confident_labels' of df_gen_hard_labels.csv has no 1(confident label), then throw this row.
    - How to set target value
    |----------------------------------------------------------------------------------------------------------------------------------|
    |                           |                           Predicted probability                                                      |
    |---------------------------|------------------------------------------------------------------------------------------------------|
    |    Labels IN ogg pri&sec  | <=0.5, set target to 0 | > 0.5, set target to 1                                                      |
    |---------------------------|------------------------------------------------------------------------------------------------------|
    | Labels NOT IN ogg pri&sec | <=0.5, set target to 0 | > 0.5, means probably it`s missing label here, set target to 0.5(as mask)   |
    |----------------------------------------------------------------------------------------------------------------------------------|
    """
    with utils.timer("Inference hard labels for df_need_gen.csv"):
        pred_pro, ogg_crop_steps = tra_val_test.infer_hard_labels(bootstrap_model_a, infer_hard_loader, config)
        assert(np.array_equal(df_need_gen['ogg_crop_step'].to_numpy(), ogg_crop_steps))
        df_need_gen[datasets.TARGET_COLS] = pred_pro
        df_need_gen.to_csv(f"{HARD_LABELS_DIR}/df_need_gen_with_{config['base_models'][0]}_pred_pro.csv", index=False)

elif not df_need_gen_generated:
    with utils.timer("Organize and merge predictions with original pri&sec labels"):
        df_need_gen = pd.read_csv(f"{HARD_LABELS_DIR}/df_need_gen_with_{config['base_models'][0]}_pred_pro.csv")
        pred_pro = df_need_gen[datasets.TARGET_COLS].to_numpy()
        df_need_gen.drop(columns=datasets.TARGET_COLS, inplace=True)

        # Convert predicted logits to binary and save predicted 1 to column 'predict_labels'
        binary_preds = np.array(pred_pro > config['infer_hard_threshold']).astype('int32')
        pos_preds_idx = binary_preds.nonzero()
        df_need_gen['predict_labels'] = ''
        for row_idx, col_idx in zip(pos_preds_idx[0], pos_preds_idx[1]):
            df_need_gen.loc[row_idx, 'predict_labels'] += f"{datasets.INV_BIRD_CODE[col_idx]} "

        # Remove rows with no predicted 1
        print(f"Remove {(df_need_gen['predict_labels'] == '').sum()} rows of no predicted 1 from df_need_gen")
        df_need_gen = df_need_gen[df_need_gen['predict_labels'] != ''].reset_index(drop=True)

        # Merge 'predict_labels' with 'ogg_pri_label' and 'ogg_sec_labels'
        df_need_gen['probably_missing_labels'], df_need_gen['gen_confident_labels'] = '', ''
        for idx, row in df_need_gen.iterrows():
            ogg_pri_label, ogg_sec_labels = row['ogg_pri_label'], row['ogg_sec_labels']
            pri_sec_labels = np.unique([ogg_pri_label] + eval(ogg_sec_labels))
            predict_labels = np.unique(row['predict_labels'].split())
            for predict_label in predict_labels:
                if predict_label in pri_sec_labels:
                    df_need_gen.loc[idx, 'gen_confident_labels'] += f"{predict_label} "
                else:
                    df_need_gen.loc[idx, 'probably_missing_labels'] += f"{predict_label} "

        # Remove rows with no predicted labels in original pri&sec labels
        print(f"Remove {(df_need_gen['gen_confident_labels'] == '').sum()} rows of "
              f"no gen_confident_labels from df_need_gen")
        df_need_gen = df_need_gen[df_need_gen['gen_confident_labels'] != ''].reset_index(drop=True)

        df_need_gen.to_csv(f"{HARD_LABELS_DIR}/df_need_gen_{config['infer_hard_threshold']}.csv", index=False)

    # merge df_not_need_gen and df_need_gen and EDA for labels' count
    df_not_need_gen = pd.read_csv(f"{HARD_LABELS_DIR}/df_not_need_gen.csv")
    df_not_need_and_need_gen_labels = df_not_need_gen.append(df_need_gen).reset_index(drop=True)
    assert(df_not_need_and_need_gen_labels['ogg_crop_step'].nunique() == len(df_not_need_and_need_gen_labels))
    assert((df_not_need_and_need_gen_labels['gen_confident_labels'] == '').sum() == 0)
    print(f"df_final_gen_labels_{config['infer_hard_threshold']} has total {len(df_not_need_and_need_gen_labels)} rows")
    # df_final_gen_labels.to_csv(f"{HARD_LABELS_DIR}/df_final_gen_labels_{config['infer_hard_threshold']}.csv",
    #                            index=False)

    # EDA for 'gen_confident_labels' of df_final_gen_labels.csv
    df_gen_label_counts = utils.get_labels_value_counts(df_not_need_and_need_gen_labels['gen_confident_labels'], sep='space')
    df_gen_label_counts.to_csv(f"{HARD_LABELS_DIR}/df_gen_label_counts_{config['infer_hard_threshold']}.csv")

    df_label_counts = pd.read_csv(f"{HARD_LABELS_DIR}/df_label_counts.csv", index_col=0)
    df_label_counts[f"{config['infer_hard_threshold']}_gen_label_counts"] = 0
    for idx, row in df_gen_label_counts.iterrows():
        df_label_counts.loc[idx, f"{config['infer_hard_threshold']}_gen_label_counts"] = row[0]
    assert (df_label_counts[f"{config['infer_hard_threshold']}_gen_label_counts"] > 0).sum() == len(df_gen_label_counts)
    df_label_counts.to_csv(f"{HARD_LABELS_DIR}/df_label_counts.csv")

elif not df_final_gen_labels_generated:
    """ Further sample for df_need_gen:
        - label of df_label_counts < 200 : keep all rows in df_need_gen
        - other labels: random sample one ogg file
    """
    df_not_need_gen = pd.read_csv(f"{HARD_LABELS_DIR}/df_not_need_gen.csv")
    df_need_gen = pd.read_csv(f"{HARD_LABELS_DIR}/df_need_gen_{config['infer_hard_threshold']}.csv")
    df_label_counts = pd.read_csv(f"{HARD_LABELS_DIR}/df_label_counts.csv", index_col=0)
    df_label_counts = df_label_counts[df_label_counts.index != 'nocall']

    print(f"Before sampling, df_need_gen_{config['infer_hard_threshold']}.csv has {len(df_need_gen)} rows")
    df_need_gen['keep'] = 0
    less_than400_labels = df_label_counts[df_label_counts[f"{config['infer_hard_threshold']}_gen_label_counts"] < 400].index
    for idx, row in df_need_gen.iterrows():
        gen_labels = row['gen_confident_labels'].split()
        for gen_label in gen_labels:
            if gen_label in less_than400_labels:
                df_need_gen.loc[idx, 'keep'] = 1
                break
    print(f"less_than400_labels of df_need_gen_{config['infer_hard_threshold']}.csv has {df_need_gen['keep'].sum()} rows")

    df_remain = df_need_gen[df_need_gen['keep'] == 0]
    print(f"df_remain has {df_remain['ogg_name'].nunique()} nunique ogg files")
    df_groupby_filename = df_remain.groupby[['ogg_name']]
    pass

