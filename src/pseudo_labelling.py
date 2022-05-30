import pickle
import pandas as pd
import numpy as np
import librosa
import os
import torch
import torch.utils.data as data
import config.globals as config_g
import config.params_baseline as config_p
import src.utils as utils
import src.models as models
import multiprocessing as mp
import math

from tqdm import tqdm

pseudo_loader = {
    'batch_size': 400,
    'shuffle': False,
    'num_workers': 0,
    'drop_last': False,
    'pin_memory': True
}


def get_pseudo_loader(mel_filename, config_p, t_probs):
    dataset = AudioFileDataset(mel_filename, config_p, t_probs)
    loader = data.DataLoader(dataset, **pseudo_loader)
    return loader


class PredBuffer(object):
    def __init__(self):
        self.y_logits = []
        self.y_probs = []

    def update(self, pred_logits):
        pred_logits = pred_logits.detach().cpu()
        self.y_logits.extend(pred_logits.numpy().tolist())
        self.y_probs.extend(torch.sigmoid(pred_logits).numpy().tolist())

    @property
    def pred_probs(self):
        return np.array(self.y_probs).astype(np.float32)

    @property
    def pred_logits(self):
        return np.array(self.y_logits)


class AudioFileDataset(data.Dataset):
    def __init__(self, mel_filename, config_p, t_probs):
        self.mel_filename = mel_filename
        self.config = config_p
        self.mel_spec = torch.load(f"{config_g.DIR_MEL}/{mel_filename}").numpy().astype(np.float32)
        self.t_probs = t_probs

    def __len__(self):
        return len(self.t_probs[f"{self.mel_filename}"])

    def __getitem__(self, idx):
        start_frame = round(idx * self.config.frames_per_1s)
        end_frame = start_frame + self.config.frames_per_period
        images = self.mel_spec[:, start_frame:end_frame].astype(np.float32)

        images = librosa.power_to_db(images, ref=np.max)  # convert S**2 to db[-80, 0]
        images = utils.mono_to_color(images)  # Convert to 3 channels RGB array range in [0, 255], shape [H x W x 3]
        images = utils.normalize(images)  # Normalizes RGB array to range [0, 1], with shape [3 x H x W]

        return {'image': images}  # , 'start_sec': idx}


def inference_pseudo_labels(df, model_path, config_p):
    with open(f"{config_g.DIR_ENERGY_PROBS}/{config_p.energy_probs_pkl}", 'rb') as file:
        t_probs = pickle.load(file)

    bird_model = models.BirdCLEFModel(config_p, base_model=config_p.base_models[0], pretrain=False)
    bird_model.load_state_dict(torch.load(model_path, map_location=config_g.DEVICE))
    bird_model = bird_model.to(config_g.DEVICE)
    bird_model.eval()

    probs_dict = {}
    audio_files = df['mel_filename'].to_numpy()
    for filename in tqdm(audio_files):
        data_loader = get_pseudo_loader(filename, config_p, t_probs)
        with torch.no_grad():
            pred_buffer = PredBuffer()

            for sample in data_loader:
                inputs = sample['image'].to(config_g.DEVICE)
                pred_logits, _ = bird_model(inputs)
                pred_buffer.update(pred_logits)

        probs_dict[f"{filename}"] = pred_buffer.pred_probs

    return probs_dict


if __name__ == '__main__':
    generating_pseudo_from_stage1 = True
    generating_pseudo_from_stage2 = False

    if generating_pseudo_from_stage1:
        Trained_Models = [f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/BirdCLEF_rexnet_100_fold0.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/BirdCLEF_rexnet_100_fold1.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/BirdCLEF_rexnet_100_fold2.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/BirdCLEF_rexnet_100_fold3.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/BirdCLEF_rexnet_100_fold4.bin"]
    elif generating_pseudo_from_stage2:
        Trained_Models = [f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/BirdCLEF_rexnet_100_fold0.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/BirdCLEF_rexnet_100_fold1.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/BirdCLEF_rexnet_100_fold2.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/BirdCLEF_rexnet_100_fold3.bin",
                          f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/BirdCLEF_rexnet_100_fold4.bin"]

    config = config_p.Params()
    df_tra_mel = pd.read_csv(config_g.CSV_TRA_MEL_META)
    # df_tra_mel = df_tra_mel[0:100]  # For debug

    with utils.timer("Inference pseduo labels on CSV_TRA_MEL_META"):
        pseudo_labels_dict = {}
        for fold in range(config.nfolds):
            pseudo_labels_fold = inference_pseudo_labels(df_tra_mel[df_tra_mel['kfold'] == fold],
                                                         Trained_Models[fold], config)
            pseudo_labels_dict.update(pseudo_labels_fold)

    with open(f"{config_g.DIR_ENERGY_PROBS}/oof_pseudo_labels.pkl", 'wb') as file:
        pickle.dump(pseudo_labels_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    assert (len(pseudo_labels_dict) == len(df_tra_mel))  # Verify

    with open(f"{config_g.DIR_ENERGY_PROBS}/oof_pseudo_labels.pkl", 'rb') as file:
        read_pseudo_dict = pickle.load(file)
        assert (len(read_pseudo_dict) == len(pseudo_labels_dict))


