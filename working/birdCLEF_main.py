import numpy as np
import pandas as pd
import os
import torch
import config.globals as config_g
import config.params_baseline as config_params
import src.utils as utils
import src.tra_val_test as tra_val_test

from src.criterions import bce_loss_with_logits
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, precision_score, recall_score

df_tra_soundscape_generated = True

if __name__ == '__main__':
    params = config_params.Params()

    # Make log dir
    os.makedirs(config_g.DIR_OUTPUT, exist_ok=True)
    logger = utils.get_logger(f"{config_g.DIR_OUTPUT}/output.log")

    # Seed everything
    utils.set_seed(params.seed)

    # StratifiedKFold according to 'primary_label'
    df_tra_mel_meta = pd.read_csv(config_g.CSV_TRA_MEL_META)
    df_tra_mel_meta.fillna('', inplace=True)
    utils.logging_msg(f"Total {len(df_tra_mel_meta)} rows for train.", logger)

    df_tra_mel_meta['kfold'] = 0
    if params.nfolds > 1:
        skf = StratifiedKFold(n_splits=params.nfolds, shuffle=True, random_state=params.seed)
        for fold, (t_idx, v_idx) in enumerate(skf.split(df_tra_mel_meta, df_tra_mel_meta['pri_label'])):
            df_tra_mel_meta.loc[v_idx, 'kfold'] = int(fold)
        # print(f"SKF result: df_tra_mel_meta['kfold'].value_counts():\n{df_tra_mel_meta['kfold'].value_counts()}")
    df_tra_mel_meta.to_csv(f"{config_g.CSV_TRA_MEL_META}", index=False)

    # Prepare train_sounscapes as test set
    if not df_tra_soundscape_generated:
        df_tra_soundscape = pd.read_csv(config_g.CSV_TRA_SOUNDSCAPE)

        # Generate column 'filename'
        audio_ids = df_tra_soundscape['audio_id'].unique()
        audio_file_dict = {}
        for dirname, _, filenames in os.walk(config_g.DIR_TRA_SOUNDSCAPES):
            for filename in filenames:
                for audio_id in audio_ids:
                    if filename.startswith(str(audio_id)):
                        audio_file_dict[audio_id] = filename

        df_tra_soundscape['filename'] = ''
        for idx, row in df_tra_soundscape.iterrows():
            df_tra_soundscape.loc[idx, 'filename'] = audio_file_dict[row['audio_id']]

        # Generate targets columns
        df_tra_soundscape[config_g.TARGET_COLS_WITHOUT_NOCALL] = 0
        for idx, row in df_tra_soundscape.iterrows():
            birds = row['birds'].split()
            for bird in birds:
                if bird != 'nocall':
                    df_tra_soundscape.loc[idx, bird] = 1

        df_tra_soundscape.to_csv('./df_tra_soundscape.csv', index=False)
    else:
        df_tra_soundscape = pd.read_csv('./df_tra_soundscape.csv')

    # Start training
    for model in params.base_models:
        oof, pred_logits = tra_val_test.train_k_folds(config=params, base_model=model, tra_val_df=df_tra_mel_meta,
                                                test_df=df_tra_soundscape, logger=logger)

        """
        oof_bce_loss = np.inf
        if oof.sum():
            oof_bce_loss = bce_loss_with_logits(oof, df_less_than34s[datasets.TARGET_COLS].to_numpy())
        """

        # tra_ss_bce_loss = bce_loss_with_logits(preds, df_tra_soundscape[config_g.TARGET_COLS].to_numpy())

        if config_g.USE_NOCALL:
            binary_outputs = utils.convert_nocall_logits(pred_logits)
        else:
            binary_outputs = np.array((utils.sigmoid(pred_logits) > params.f1_threshold)).astype('int32')

        y_true = df_tra_soundscape[config_g.TARGET_COLS_WITHOUT_NOCALL].to_numpy().astype('int32')
        blending_f1_score = f1_score(y_true, binary_outputs, average='samples', zero_division=1)
        blending_prec_score = precision_score(y_true, binary_outputs, average='samples', zero_division=1)
        blending_recall_score = recall_score(y_true, binary_outputs, average='samples', zero_division=1)
        utils.logging_msg(f"Finished training {model} with {params.nfolds} folds - oof_bce_loss: "
                          f"{np.inf}, blending_f1_score: {blending_f1_score:.4f}, blending_prec_score: "
                          f"{blending_prec_score:.4f}, blending_recall_score: {blending_recall_score:.4f}, "
                          f"tra_ss_bce_loss: {np.inf:.4f} of train soundscapes", logger)


