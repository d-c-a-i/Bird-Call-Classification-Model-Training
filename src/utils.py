import logging
import os
import random
import librosa
import numpy as np
import pandas as pd
import torch
import math
import pickle
import config.globals as config_g
import config.params_baseline as config_params
import multiprocessing as mp
import shutil
import filecmp
import json
import glob

from time import time
from datetime import timedelta
from contextlib import contextmanager
from typing import Optional
from tqdm import tqdm
from scipy.special import softmax
from pydub import AudioSegment, silence


def set_seed(seed: int = 40):
    print(f"Call set_seed() in process: {os.getpid()}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def logging_msg(msg: str, logger: logging.Logger = None):
    if logger:
        logger.info(msg)
    else:
        print(msg)


@contextmanager
def timer(name: str, logger: Optional[logging.Logger] = None, epoch: int = 0):
    if epoch == 0:
        t0 = time()
        msg = f"[{name}] START"
        logging_msg(msg, logger)
        yield
        msg = f"[{name}] DONE in {timedelta(seconds=round(time()-t0))}"
        logging_msg(msg, logger)
    else:
        yield


def get_logger(out_file=None):
    msg_logger = logging.getLogger()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    msg_logger.handlers = []
    msg_logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    msg_logger.addHandler(handler)

    if out_file is not None:
        if os.path.isfile(out_file):
            os.remove(out_file)

        fh = logging.FileHandler(out_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.INFO)
        msg_logger.addHandler(fh)
    msg_logger.info("logger set up")
    return msg_logger


def labels_to_listlike_str(labels):
    str_labels = ''
    len_list = len(labels)
    for i, label in enumerate(labels):
        str_labels += f"\'{str(label)}\'"
        if i < len_list - 1:
            str_labels += ', '

    return f"[{str_labels}]"


def labels_to_space_sep_str(labels):
    str_labels = ''
    for label in labels:
        str_labels += f"{label} "

    return str_labels


def get_labels_value_counts(labels_col: pd.Series, sep: str = 'space'):
    df_labels = pd.DataFrame(columns=['count'])
    arr_labels_all = labels_col.to_numpy()
    for labels in arr_labels_all:
        labels = labels.split() if sep == 'space' else eval(labels)
        for label in labels:
            df_labels.loc[len(df_labels)] = label
    return pd.DataFrame(df_labels['count'].value_counts())


# Spectrogram related
def cal_spectro_params(sr: int = 32000, chunk_period: int = 5, spec_height: int = 224, spec_width: int = 448):
    '''
    From 2nd place:
        sr: 21952, n_fft: 892, n_hop: 245, n_mels: 224 from 2nd place of Cornell Birdcall
        # n_mels = spec_width = 224
        # n_fft//2 + 2 = 224*2 --> n_fft = 892
        # 224*2 = 5*sr/n_hop --> sr/n_hop = 89.6  # TODO: how to calculate optimal n_hop
    '''
    # From https://www.kaggle.com/stefankahl/birdclef2021-processing-audio-data?scriptVersionId=58466029&cellId=14
    # sample rate just use original wave`s, n_fft could be adjust on case
    n_mels = spec_height
    n_hop = int(sr * chunk_period / (spec_width-1))

    return n_hop, n_mels


def mono_to_color(X: np.ndarray, mean=None, std=None, eps=1e-6):
    """
    # From 2nd place
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([config.spectro['mel_bins'], config.frames_per_period]),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    X = np.stack([X, X, X], axis=-1)
    V = (255 * X).astype(np.uint8)
    V = (trans(V)+1)/2
    """

    """ # From 3rd place
    Converts a one channel array to a 3 channel one in [0, 255]; Return np array [H x W x 3]
    """
    X = np.stack([X, X, X], axis=-1)  # to (224, 448, 3)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()
    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)

    return V


def normalize(image, mean=None, std=None):  # From 3rd
    """
    Normalizes an array in [0, 255] to the format adapted to neural network

    Arguments:
        image {np array [H x W x 3]} -- [description]

    Keyword Arguments:
        mean {None or np array} -- Mean for normalization, expected of size 3 (default: {None})
        std {None or np array} -- Std for normalization, expected of size 3 (default: {None})

    Returns:
        np array [3 x H x W] -- Normalized array
    """
    image = image / 255.0
    if mean is not None and std is not None:
        image = (image - mean) / std
    return np.moveaxis(image, 2, 0).astype(np.float32)


def random_power(images, power=1.5, c=0.7):
    images = images - images.min()
    images = images/(images.max()+0.0000001)
    images = images**(random.random()*power + c)

    return images


def get_mel_spectrogram(wave_file: str, spec_params: dict):
    y, _ = librosa.load(wave_file, sr=spec_params['sr'], mono=True, res_type="kaiser_fast")

    # Create melspectrogram
    spectro = librosa.feature.melspectrogram(y, sr=spec_params['sr'], n_mels=spec_params['mel_bins'],
                                             n_fft=spec_params['n_fft'], hop_length=spec_params['n_hop'],
                                             win_length=spec_params['n_fft'], fmin=spec_params['fmin'])
    return spectro.astype(np.float16)


def generate_nocall_csv():
    select_list = ['airplane', 'nature', 'wind', 'dog', 'cock', 'cricket', 'insects', 'rain', 'raindrops', 'thunder',
                   'water', 'motor'] + ['bird', 'birds']

    df_freefield = pd.DataFrame(columns=['mel_filename', 'ogg_filename', 'pri_label', 'sec_labels', 'wave_path'])
    freefiled_meta_dir = f"{config_g.DIR_FREE_FIELD1010}/metadataonly/**/*.json"
    for json_file in glob.glob(freefiled_meta_dir):
        has_bird_tag, has_select_tag = False, False

        with open(json_file) as f:
            data = json.load(f)
            """
            for tag in data['tags']:
                if tag == 'bird' or tag == 'birds':
                    has_bird_tag = True
                    continue
                
                if tag in select_list:
                    has_select_tag = True
                    continue
            
            if has_bird_tag:
                continue

            if has_select_tag:
            """
            wave_dir = json_file.split(sep='/')[-2]
            wave_name = json_file.split(sep='/')[-1].replace('.json', '.wav')
            if wave_name in ['134984.wav', '134988.wav', '15287.wav',  '134980.wav', '134973.wav', '134974.wav']:
                continue

            wav_full_path = f"{config_g.DIR_FREE_FIELD1010}/{wave_dir}/{wave_name}"
            mel_name = wave_name.replace('.wav', '.amp')
            df_freefield.loc[len(df_freefield)] = [mel_name, '', 'nocall', '', wav_full_path]

    df_freefield.to_csv(f"{config_g.DIR_WORKING}/df_freefield.csv", index=False)
    print(f"Selected {len(df_freefield)} samples for nocall data")
    return df_freefield


def generate_train_csv(df: pd.DataFrame):
    df_tra_mel = pd.DataFrame(columns=['mel_filename', 'ogg_filename', 'pri_label', 'sec_labels', 'wave_path'])

    for idx, row in df.iterrows():
        if row['filename'] in config_g.ABANDONED_OGG:  # abandoned audio files
            continue

        mel_filename = row['filename'].replace('.ogg', '.amp')
        pri_label, sec_labels = row['primary_label'], row['secondary_labels']
        # Clean sec labels
        sec_labels_str = ''
        sec_labels = np.array(eval(sec_labels))
        if sec_labels.size > 0:
            sec_labels = np.where(sec_labels == 'rocpig1', 'rocpig', sec_labels)
            sec_labels = np.unique(sec_labels)
            sec_labels = np.delete(sec_labels, np.where(sec_labels == pri_label))
            for sec_label in sec_labels:
                sec_labels_str += f"{sec_label} "

        df_tra_mel.loc[len(df_tra_mel)] = mel_filename, row['filename'], pri_label, sec_labels_str, ''

    return df_tra_mel
    # df_tra_mel_meta.to_csv(config_g.CSV_TRA_MEL_META, index=False)


def mp_generate_train_csv():
    df_train = pd.read_csv(config_g.CSV_TRA_META)

    num_rows = len(df_train)
    num_cores = int(mp.cpu_count() * 0.8)
    pool = mp.Pool(num_cores)
    audios_per_core = math.ceil(num_rows / num_cores)

    results = [pool.apply_async(generate_train_csv,
                                args=(df_train[i * audios_per_core: (i + 1) * audios_per_core], ))
               for i in range(num_cores)]
    results = [p.get() for p in results]

    # verify
    df_tra_mel_final = pd.concat(results)
    assert(len(df_tra_mel_final) == len(df_train) - len(config_g.ABANDONED_OGG))

    if config_g.USE_NOCALL:
        df_nocall = generate_nocall_csv()
        df_tra_mel_final = df_tra_mel_final.append(df_nocall).reset_index(drop=True)

    df_tra_mel_final.to_csv(config_g.CSV_TRA_MEL_META, index=False)
    print(f"CSV_TRA_MEL_META generated, total {len(df_tra_mel_final)} rows")


# Convert original audio to mel-spectrogram and save to local disk
def generate_mel_spec(src_ogg, dst_mel, spec_params):
    mel_spec = get_mel_spectrogram(src_ogg, spec_params)
    mel_spec = torch.from_numpy(mel_spec)  # Translate to torch
    torch.save(mel_spec, dst_mel)


def generate_train_mel_file(df: pd.DataFrame, spec_params: dict, only_nocall: bool=False):
    existing_amp = [name for name in os.listdir(f"{config_g.DIR_MEL}") if name.endswith(".amp")]
    skipped = 0

    # df_tra_mel_meta = pd.DataFrame(columns=['mel_filename', 'ogg_filename', 'pri_label', 'sec_labels'])
    for idx, row in df.iterrows():
        mel_filename, ogg_filename, pri_label, sec_labels, wav_path = row.to_numpy()
        if only_nocall and pri_label != 'nocall':
            continue

        if mel_filename in existing_amp:
            skipped += 1
            continue

        if pri_label == 'nocall':
            src_file_path = wav_path
        else:
            src_file_path = f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{ogg_filename}"

        generate_mel_spec(src_file_path, f"{config_g.DIR_MEL}/{mel_filename}", spec_params)

    print(f'Training Mel Spectrogram dataset created, skipped {skipped} files')
    return skipped


def mp_generate_train_mel_files(df_tra_mels: pd.DataFrame, only_nocall: bool=False):
    os.makedirs(config_g.DIR_MEL, exist_ok=True)

    config_spectro = config_params.Params().spectro
    num_rows = len(df_tra_mels)
    using_cores = 2  # int(mp.cpu_count() * 0.8)
    mp_pool = mp.Pool(using_cores)
    files_per_core = math.ceil(num_rows / using_cores)

    results = [mp_pool.apply_async(generate_train_mel_file,
                                   args=(df_tra_mels[i * files_per_core: (i + 1) * files_per_core],
                                         config_spectro, only_nocall))
               for i in range(using_cores)]
    results = [p.get() for p in results]

    # verify
    # existing_amp = [name for name in os.listdir(f"{config_g.DIR_MEL}") if name.endswith(".amp")]
    # assert(len(existing_amp) == len(df_tra_mels))
    # print(f"Total {len(existing_amp)} mel files generated in DIR_MEL")


def generate_manual_mel():
    spec_params = config_params.Params().spectro

    for ogg_name in tqdm(config_g.MANUALED_OGG):
        mel_filename = ogg_name.replace('.ogg', '.amp')
        ogg_file_path = f"{config_g.DIR_MANUAL_FILES}/{ogg_name}"

        generate_mel_spec(ogg_file_path, f"{config_g.DIR_MEL}/{mel_filename}", spec_params)
        #print(f"Generated manual ogg {ogg_name} to mel")


def bg_preprocess(spec_params: dict):
    df_bg_noise = pd.read_csv(f"{config_g.DIR_BG_NOISE}/BG_noise_selection.csv")
    config = config_params.Params()

    for idx, row in tqdm(df_bg_noise.iterrows()):
        suffix = f".{row['filename'].split(sep='.')[-1]}"
        mel_filename = row['filename'].replace(suffix, '.amp')
        df_bg_noise.loc[idx, 'mel_filename'] = mel_filename

        original_file_path = f"{config_g.DIR_BG_NOISE}/{row['filename']}"
        mel_spec = get_mel_spectrogram(original_file_path, spec_params)

        # padding 5s silent before and after noise data
        padding_chunk = np.zeros((config.spectro['mel_bins'], config.frames_per_period), dtype=np.float32)
        mel_spec = np.concatenate((padding_chunk, mel_spec, padding_chunk), axis=1)

        mel_spec = torch.from_numpy(mel_spec)  # Translate to torch
        torch.save(mel_spec, f"{config_g.DIR_BG_NOISE}/{mel_filename}")

    df_bg_noise.to_csv(f"{config_g.DIR_BG_NOISE}/BG_noise_selection.csv", index=False)
    print('Background noise converted to mel spectrogram')


# Energy trimming via wave file
def compute_normalized_energy(wave: np.array):
    wave = wave / np.abs(wave).max()  # Normalize np.abs(wave) to [0, 1]
    return np.power(wave, 2)  # wave value to [0, 1]


def compute_sampling_distribution(feature: np.ndarray, hop_size: int, window_size: int, probs_comp: str):
    n_steps = max(math.ceil((len(feature) - window_size)/hop_size) + 1, 1)  # +1 for tail cropping

    # areas: 每个step的window内的power的和; probs: 每个step的power和的分布概率, softmax/uniform/rm_softmax分布;
    if probs_comp == 'softmax':
        areas = [feature[hop_size*i:window_size + hop_size*i].sum() for i in range(n_steps)]
        probs = softmax(areas)
    elif probs_comp == 'uniform':
        areas = [feature[hop_size*i:window_size + hop_size*i].sum() for i in range(n_steps)]
        probs = np.array(areas) / sum(areas)
    elif probs_comp == 'rm_softmax':
        areas = [feature[hop_size*i:window_size + hop_size*i].mean() for i in range(n_steps)]
        probs = softmax(np.power(np.array(areas), 0.5))
    else:
        raise ValueError('Invalid probs_comp')

    return probs


def generate_wave_energy_probs(df: pd.DataFrame):
    spec_params = config_params.Params().spectro
    sr_using = config_g.SAMPLE_RATE  # using 32k for energy trimming
    h_s, w_s = sr_using * 1, sr_using * spec_params['period']

    ogg_files = df[['ogg_filename', 'pri_label']].to_numpy()
    ogg_energy_dict = {}

    for ogg in tqdm(ogg_files):
        ogg_filename, pri_label = ogg[0], ogg[1]
        au, _ = librosa.load(f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{ogg_filename}", sr=sr_using, mono=True)
        t_pobs_uniform = compute_sampling_distribution(feature=compute_normalized_energy(au), hop_size=h_s,
                                                       window_size=w_s, probs_comp='uniform')
        ogg_energy_dict[ogg_filename] = t_pobs_uniform

        '''
        t_pobs_softmax = compute_sampling_distribution(feature=compute_normalized_energy(au), hop_size=h_s,
                                                       window_size=w_s, probs_comp='softmax')

        t_pobs_rm_softmax = compute_sampling_distribution(feature=compute_normalized_energy(au), hop_size=h_s,
                                                          window_size=w_s, probs_comp='rm_softmax')
        step_cols = [f"step{i}" for i in range(len(t_pobs_softmax))]
        df_t_probs = pd.DataFrame(columns=['ogg_file'] + step_cols)
        df_t_probs.loc[len(df_t_probs)] = [f"{filename}_softmax"] + list(t_pobs_softmax)
        df_t_probs.loc[len(df_t_probs)] = [f"{filename}_uniform"] + list(t_pobs_uniform)
        df_t_probs.loc[len(df_t_probs)] = [f"{filename}_rm_softmax"] + list(t_pobs_rm_softmax)
        df_t_probs.to_csv('df_t_probs.csv', index=False)
        '''

    # test_energy_choice(ogg_energy_dict, 'wav_energy_uniform')  # For debug
    save_energy_dict_and_verify(ogg_energy_dict, 'wave_energy_probs_uniform.pkl')


# Energy trimming via mel spectrogram
def test_energy_choice(energy_dict: dict, probs_name):
    for k, v in energy_dict.items():
        df_test = pd.DataFrame(columns=['choose_step'])

        for i in range(100):
            start_sec = np.random.choice(len(v), size=1, p=v)[0]
            df_test.loc[len(df_test)] = start_sec
        df_choice = pd.DataFrame(df_test['choose_step'].value_counts())
        df_choice.to_csv(f"df_{k.split(sep='.')[0]}_{probs_name}_choice.csv")


def save_energy_dict_and_verify(energy_dict: dict, save_name: str):
    with open(f"{config_g.DIR_ENERGY_PROBS}/{save_name}", 'wb') as file:
        pickle.dump(energy_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    # Verify
    with open(f"{config_g.DIR_ENERGY_PROBS}/{save_name}", 'rb') as file:
        energy_probs_dict = pickle.load(file)

    assert(energy_dict.keys() == energy_probs_dict.keys())
    print(f"len of energy_probs_dict is {len(energy_probs_dict)}")

    for filename, t_probs in energy_probs_dict.items():
        if np.isnan(t_probs).sum() > 0:
            print(f"{filename} has Nan in t_probs")


def compute_mel_energy_distribution(feature: np.ndarray, hop_size: float, window_size: int, probs_comp: str):
    n_steps = max(math.ceil((feature.shape[1] - window_size)/hop_size), 1)

    # Suppress energy level for 1st or last second, as there is most likely noise
    suppress_frames = round(hop_size)
    feature[:, :suppress_frames] = feature[:, :suppress_frames] / 2
    feature[:, -suppress_frames:] = feature[:, -suppress_frames:] / 2

    # areas: 每个step的window内的power的和; probs: 每个step的power和的分布概率, softmax/uniform/rm_softmax分布;
    if probs_comp == 'softmax':
        areas = [feature[:, round(hop_size*i):window_size + round(hop_size*i)].sum() for i in range(n_steps)]
        probs = softmax(areas)
    elif probs_comp == 'uniform':
        areas = [feature[:, round(hop_size*i):window_size + round(hop_size*i)].sum() for i in range(n_steps)]
        probs = np.array(areas) / sum(areas)
    elif probs_comp == 'rm_softmax':
        areas = [feature[:, round(hop_size*i):window_size + round(hop_size*i)].mean() for i in range(n_steps)]
        probs = softmax(np.power(np.array(areas), 0.5))
    else:
        raise ValueError('Invalid probs_comp')

    return probs


def generate_mel_energy_probs(df: pd.DataFrame):
    config = config_params.Params()
    h_s, w_s = config.frames_per_1s, config.frames_per_period  # frames_per_1s is 89.6

    mel_energy_uniform_dict = {}
    # mel_energy_softmax_dict = {}

    df['n_steps'] = 0
    for idx, row in df.iterrows():
        if row['pri_label'] == 'nocall':
            continue

        mel_filename = row['mel_filename']
        y = torch.load(f"{config_g.DIR_MEL}/{mel_filename}").numpy().astype(np.float32)  # y is already powered by 2
        y = (y - y.min()) / (y.max() - y.min())  # Normalize to [0, 1]

        t_probs_uniform = compute_mel_energy_distribution(feature=y, hop_size=h_s, window_size=w_s,
                                                          probs_comp='uniform')
        df.loc[idx, 'n_steps'] = len(t_probs_uniform)
        # t_probs_softmax = compute_mel_energy_distribution(feature=y, hop_size=h_s, window_size=w_s, probs_comp='softmax')
        mel_energy_uniform_dict[f"{mel_filename}"] = t_probs_uniform
        # mel_energy_softmax_dict[f"{mel_filename}"] = t_probs_uniform

        """
        # Save to csv
        step_cols = [f"step{i}" for i in range(len(t_pobs_softmax))]
        df_t_probs = pd.DataFrame(columns=['ogg_file'] + step_cols)
        df_t_probs.loc[len(df_t_probs)] = [f"{mel_filename}_softmax"] + list(t_pobs_softmax)
        df_t_probs.loc[len(df_t_probs)] = [f"{mel_filename}_uniform"] + list(t_pobs_uniform)
        df_t_probs.to_csv('df_mel_probs.csv', index=False)
        """

    # Test choice
    # test_energy_choice(mel_energy_uniform_dict, 'mel_energy_uniform')
    # test_energy_choice(mel_energy_softmax_dict, 'mel_energy_softmax')

    # save_energy_dict_and_verify(mel_energy_uniform_dict, f"{config.energy_probs_pkl}")
    return mel_energy_uniform_dict, df


def mp_generate_mel_energy_probs(df: pd.DataFrame):
    os.makedirs(config_g.DIR_ENERGY_PROBS, exist_ok=True)

    num_rows = len(df)
    using_cores = int(mp.cpu_count() * 0.8)
    mp_pool = mp.Pool(using_cores)
    files_per_core = math.ceil(num_rows / using_cores)

    prob_dicts = [mp_pool.apply_async(generate_mel_energy_probs,
                                      args=(df[i * files_per_core: (i + 1) * files_per_core], ))
                  for i in range(using_cores)]
    prob_dicts_list = [p.get() for p in prob_dicts]

    mel_energy_final_dict = {}
    for i in range(len(prob_dicts_list)):
        mel_energy_final_dict.update(prob_dicts_list[i][0])
    assert (len(mel_energy_final_dict) == len(df[df['pri_label'] != 'nocall']))  # Verify

    df_new = prob_dicts_list[0][1][0:0]
    for i in range(len(prob_dicts_list)):
        df_new = df_new.append(prob_dicts_list[i][1])
    assert(len(df_new) == len(df))
    df_new.to_csv(config_g.CSV_TRA_MEL_META, index=False)

    save_energy_dict_and_verify(mel_energy_final_dict, f"{config_params.Params().energy_probs_pkl}")


# Sanity check for audio file
def check_audios_loudness(audios):
    super_noisy_th, noisy_th, quiet_th, silent_th = -12, -15, -65, -75
    super_noisy_files, noisy_files, quiet_files, silent_files = [], [], [], []
    files_num = audios.shape[0]

    # Check quiet or noisy files
    for idx in range(files_num):
        filename, pri_label = audios[idx, 0], audios[idx, 1]

        au = AudioSegment.from_file(f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{filename}")
        au_dbfs = au.dBFS
        if quiet_th >= au_dbfs > silent_th:
            print(f"Quiet file: {filename}, dBFS is {au_dbfs:.1f}")
            quiet_files.append(filename)
        elif au_dbfs <= silent_th:
            print(f"Silent file: {filename}, dBFS is {au_dbfs:.1f}")
            silent_files.append(filename)
        elif super_noisy_th > au_dbfs >= noisy_th:
            print(f"Noisy file: {filename}, dBFS is {au_dbfs:.1f}")
            noisy_files.append(filename)
        elif au_dbfs >= super_noisy_th:
            print(f"Super noisy file: {filename}, dBFS is {au_dbfs:.1f}")
            super_noisy_files.append(filename)

    return {'silent_files': silent_files,
            'super_noisy_files': super_noisy_files,
            'noisy_files': noisy_files,
            'quiet_files': quiet_files}


def detect_audios_silence_seg(audios):
    silent_th = -80
    silent_files = []
    files_num = audios.shape[0]

    for idx in range(files_num):
        filename, pri_label = audios[idx, 0], audios[idx, 1]

        au = AudioSegment.from_file(f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{filename}")
        silent_list = silence.detect_silence(au, min_silence_len=4000, silence_thresh=silent_th, seek_step=1000)
        if len(silent_list) > 0:
            print(f"{filename} has silent seg: {silent_list}")
            silent_files.append({f"{filename}": silent_list})

    return {'silent_files': silent_files}


def copy_need_manual_files():  # Copy need-manual files from SILENCE_CHECKED_FILES to DIR_MANUAL_FILES
    # Copy need-manual files to DIR_MANUAL_FILES
    existing_ogg = [name for name in os.listdir(f"{config_g.DIR_MANUAL_FILES}") if name.endswith(".ogg")]
    df_tra_meta = pd.read_csv(f"{config_g.CSV_TRA_META}")

    for ogg_file in config_g.SILENCE_CHECKED_FILES:
        if ogg_file in existing_ogg:
            print(f"{ogg_file} already in manual folder, skip.")
            continue

        pri_label = df_tra_meta[df_tra_meta['filename'] == ogg_file]['primary_label'].item()
        shutil.copyfile(f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{ogg_file}",
                        f"{config_g.DIR_MANUAL_FILES}/{ogg_file}")

    # Verify
    existing_ogg = [name for name in os.listdir(f"{config_g.DIR_MANUAL_FILES}") if name.endswith(".ogg")]
    assert (config_g.SILENCE_CHECKED_FILES.sort() == existing_ogg.sort())
    print(f"Total {len(existing_ogg)} ogg files in manual folder")


def clean_manual_folder():  # Remove files from DIR_MANUAL_FILES if it`s identical as original
    # existing_ogg = glob.glob(f"{config_g.DIR_MANUAL_FILES}/*.ogg")
    existing_ogg = [name for name in os.listdir(f"{config_g.DIR_MANUAL_FILES}") if name.endswith(".ogg")]
    df_tra_meta = pd.read_csv(f"{config_g.CSV_TRA_META}")

    for manual_file in existing_ogg:
        pri_label = df_tra_meta[df_tra_meta['filename'] == manual_file]['primary_label'].to_numpy()[0]
        if filecmp.cmp(f"{config_g.DIR_MANUAL_FILES}/{manual_file}",
                       f"{config_g.DIR_TRA_SHORT_AUDIO}/{pri_label}/{manual_file}"):
            print(f"Remove {manual_file} from manual folder as it`s not changed")
            os.remove(f"{config_g.DIR_MANUAL_FILES}/{manual_file}")

    existing_ogg = [name for name in os.listdir(f"{config_g.DIR_MANUAL_FILES}") if name.endswith(".ogg")]
    print(f"Remain {len(existing_ogg)} ogg files in munual folder: {existing_ogg}")


def backup_original_mel():
    for manual_ogg_name in config_g.MANUALED_OGG:
        mel_filename = manual_ogg_name.replace('.ogg', '.amp')
        shutil.copyfile(f"{config_g.DIR_MEL}/{mel_filename}",
                        f"{config_g.DIR_MEL_BACKUP}/{mel_filename}")

    backuped_mel = [name for name in os.listdir(f"{config_g.DIR_MEL_BACKUP}") if name.endswith(".amp")]
    print(f"Total {len(backuped_mel)} mel files in DIR_MEL_BACKUP folder")


def cal_pos_counts(df: pd.DataFrame, config_p):
    class_counts = np.zeros(config_g.NUM_TARGETS)

    for idx, row in df.iterrows():
        pri_label, sec_labels = row['pri_label'], row['sec_labels'].split()
        class_counts[config_g.BIRD_CODE[pri_label]] += 1  # pri_label count as 1
        for sec_label in sec_labels:
            class_counts[config_g.BIRD_CODE[sec_label]] += config_p.sec_label_target

    return class_counts


def mp_cal_pos_weights(df_tra):
    df_tra.fillna('', inplace=True)
    config_p = config_params.Params()

    num_samples = len(df_tra)
    using_cores = int(mp.cpu_count() * 0.8)
    mp_pool = mp.Pool(using_cores)
    files_per_core = math.ceil(num_samples / using_cores)

    results = [mp_pool.apply_async(cal_pos_counts,
                                   args=(df_tra[i * files_per_core: (i + 1) * files_per_core], config_p))
               for i in range(using_cores)]
    results_list = [p.get() for p in results]

    class_counts = np.zeros(config_g.NUM_TARGETS)
    for weights in results_list:
        class_counts += weights

    neg_counts = [num_samples - pos_count for pos_count in class_counts]
    pos_weights = np.ones_like(class_counts)
    for cdx, (pos_count, neg_count) in enumerate(zip(class_counts, neg_counts)):
        pos_weights[cdx] = neg_count / pos_count

    # Normalize pos_weights to [1, 5]
    pos_weights = pos_weights - pos_weights.min()
    pos_weights = pos_weights / pos_weights.max()
    pos_weights = pos_weights * 4 + 1
    pos_weights = torch.as_tensor(pos_weights, dtype=torch.float, device=config_g.DEVICE)

    # class weights inverse to class frequency
    class_weights = class_counts / num_samples  # min 0.9851 # if / class_counts.sum(), min: 0.9879
    class_weights = 1 - class_weights
    # class_weights = class_weights ** 4  # min 0.9423
    class_weights = torch.as_tensor(class_weights, dtype=torch.float, device=config_g.DEVICE)

    return pos_weights, class_weights


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def convert_nocall_logits(y_logits: np.ndarray):  # (num_samples, num_targets)
    assert(y_logits.shape[1] == config_g.NUM_TARGETS)
    y_preds = y_logits.copy()

    num_samples = len(y_logits)
    for i in range(num_samples):
        """
        if y_logits[i, config_g.BIRD_CODE['nocall']] == 1:
            y_logits[i, :] = 0
        """
        # If 'nocall' has maximum logits, then set all preiction to 0
        if y_logits[i].max() == y_logits[i, config_g.BIRD_CODE['nocall']]:
            y_preds[i, :] = 0
        else:
            y_preds[i, :] = sigmoid(y_preds[i, :])

    y_preds = y_preds[:, :-1]  # remove 'nocall' column
    binary_output = (y_preds > config_params.Params().f1_threshold).astype('int32')

    return binary_output


def generate_merged_probs(original_probs_pkl, oof_pkl):
    config = config_params.Params()
    df_tra_mel_meta = pd.read_csv(config_g.CSV_TRA_MEL_META)

    with open(original_probs_pkl, 'rb') as file:
        original_probs = pickle.load(file)

    # with open(f"{config_g.DIR_ENERGY_PROBS}/merged_probs.pkl", 'rb') as file:
    #    merged_probs = pickle.load(file)

    with open(oof_pkl, 'rb') as file:
        pseudo_dict = pickle.load(file)

    energy_pseudo_probs = {}
    # pseudo_predicts = {}
    for idx, row in df_tra_mel_meta.iterrows():
        mel_name, pri_label, sec_labels = row['mel_filename'], row['pri_label'], str(row['sec_labels'])
        pri_idx = [config_g.BIRD_CODE[f"{pri_label}"]]
        sec_idx = [config_g.BIRD_CODE[f"{sec_label}"] for sec_label in sec_labels.split()] if sec_labels != 'nan' else []

        pseudo_arr = pseudo_dict[f"{mel_name}"]
        original_mel_prob = original_probs[f"{mel_name}"]

        #if mel_name in ['XC403247.amp', 'XC252702.amp']:
        #    a = 1

        mel_pseudo_labels = np.array([pseudo_arr[i, pri_idx].sum() for i in range(pseudo_arr.shape[0])])
        if len(sec_idx) > 0:
            sec_pseudo_labels = np.array([pseudo_arr[i, sec_idx].sum() * config.sec_label_target
                                          for i in range(pseudo_arr.shape[0])])
            mel_pseudo_labels = mel_pseudo_labels + sec_pseudo_labels
        # pseudo_predicts[f"{mel_name}"] = mel_pseudo_labels  # Save predicted probs

        """
        mel_pseudo_labels = mel_pseudo_labels / mel_pseudo_labels.sum()  # convert clipwise distribution probs
        merged_prob_arr = mel_pseudo_labels + original_mel_prob
        merged_prob_arr = merged_prob_arr / merged_prob_arr.sum()
        energy_pseudo_probs[f"{mel_name}"] = merged_prob_arr
        """

        # Increase found probs for highly confident
        original_mel_prob += (mel_pseudo_labels > 0.9).astype(np.int32) * original_mel_prob.mean()
        original_mel_prob = original_mel_prob/original_mel_prob.sum()
        energy_pseudo_probs[f"{mel_name}"] = original_mel_prob
    """
    with open(f"{config_g.DIR_ENERGY_PROBS}/pseudo_predicts.pkl", 'wb') as file:
        pickle.dump(pseudo_predicts, file, protocol=pickle.HIGHEST_PROTOCOL)
    """
    with open(f"{config_g.DIR_ENERGY_PROBS}/merged_probs.pkl", 'wb') as file:
        pickle.dump(energy_pseudo_probs, file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    """
    generate_mel_spec("/home/ch/My_projects/BirdCLEF/input/birdclef-2021/train_short_audio/buggna/XC174833.ogg",
                      "/home/ch/My_projects/BirdCLEF/working/mel_spectrogram/XC174833.amp",
                      config_params.Params().spectro)
    """

    train_csv_mel_generated = True
    energy_probs_generated = True
    bg_noise_preprocessed = False
    merged_probs_generated = True
    # audio_files_checked = True

    if not train_csv_mel_generated:
        with timer("Generating mel spectrogram for train short audios"):
            mp_generate_train_csv()

            df_tra_mel = pd.read_csv(f"{config_g.CSV_TRA_MEL_META}")
            mp_generate_train_mel_files(df_tra_mel, only_nocall=False)
    elif not energy_probs_generated:
        with timer("Generating energy trimming dict"):
            df_tra_mel_meta = pd.read_csv(config_g.CSV_TRA_MEL_META)
            mp_generate_mel_energy_probs(df_tra_mel_meta)
    elif not bg_noise_preprocessed:
        bg_preprocess(config_params.Params().spectro)
    elif not merged_probs_generated:
        generating_stage2_probs = True
        generating_stage3_probs = False

        config = config_params.Params()
        if generating_stage2_probs:
            original_probs_pkl = f"{config_g.DIR_ENERGY_PROBS}/{config.energy_probs_pkl}"
            oof_pkl = f"{config_g.DIR_ENERGY_PROBS}/Stage1_energy_only_probs/oof_pseudo_labels_from_stage1.pkl"
        elif generating_stage3_probs:
            original_probs_pkl = f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/{config.stage2_probs_pkl}"
            oof_pkl = f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/oof_pseudo_labels_from_stage2.pkl"

        generate_merged_probs(original_probs_pkl, oof_pkl)

    """
    elif not audio_files_checked:
        with timer("Check quiet and silent files"):
            # audio_arr = pd.read_csv(config_g.CSV_TRA_META)[['filename', 'primary_label']].to_numpy()
            audio_arr = pd.read_csv(config_g.CSV_TRA_MEL_META)[['ogg_filename', 'pri_label']].to_numpy()
            audios_num = audio_arr.shape[0]

            num_cores = int(mp.cpu_count())
            pool = mp.Pool(num_cores)
            audios_per_core = int(audios_num/num_cores)

            #results = [pool.apply_async(check_audios_loudness, args=(audio_arr[i*audios_per_core: (i+1)*audios_per_core], ))
            results = [pool.apply_async(detect_audios_silence_seg, args=(audio_arr[i*audios_per_core: (i+1)*audios_per_core], ))
                       for i in range(num_cores)]
            results = [p.get() for p in results]
            print(results)
    """

    # First copy need-manual files to manual folder
    # copy_need_manual_files()

    # Do manual modification...

    # Then clean manual files if it`s keeping no change
    # clean_manual_folder()

    # backup_original_mel()

    # generate_manual_mel()





