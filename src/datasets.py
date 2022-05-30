import random
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import pickle
import librosa
import librosa.display
import torch.utils.data as data
import config.globals as config_g
import config.params_baseline as config_p
import src.utils as utils
import matplotlib.pyplot as plt

from librosa.feature.inverse import mel_to_audio


def get_df_labels(df: pd.DataFrame):
    df = df[['primary_label', 'secondary_labels', 'filename', 'rating', 'duration']].copy()
    """
    bird_code_str = ''
    for i, bird in enumerate(df['primary_label'].unique()):
        bird_code_str += f"'{bird}': {i}, "
        if (i+1) % 5 == 0:
            bird_code_str += '\n'
    print(bird_code_str)
    """
    df.loc[:, config_g.TARGET_COLS] = 0
    for idx, row in df.iterrows():
        df.loc[idx, row['primary_label']] = 1
        sec_labels = eval(row['secondary_labels'])
        for sec_label in sec_labels:
            sec_label = 'rocpig' if sec_label == 'rocpig1' else sec_label
            df.loc[idx, sec_label] = 1

    return df


def get_loader(phase: str, df: pd.DataFrame, config):
    if phase == 'train':
        dataset = config.dataset['train'](df, config)
        loader_config = config.tra_loader
    elif phase == 'valid':
        dataset = config.dataset['valid'](df, config)
        loader_config = config.val_loader
    elif phase == 'test':
        dataset = config.dataset['test'](df, config)
        loader_config = config.test_loader
    else:
        raise NotImplementedError

    loader = data.DataLoader(dataset, **loader_config)
    return loader


def get_energy_crop(mel_filename, pri_label, crop_frames, t_probs, config):
    y = torch.load(f"{config_g.DIR_MEL}/{mel_filename}").numpy().astype(np.float32)  # y.shape: (224, frames_y)
    height_y, frames_y = y.shape
    effective_frames = crop_frames  # config.frames_per_period
    images = np.zeros((height_y, effective_frames), dtype=np.float32)

    if frames_y < effective_frames:
        start_frame = random.randint(0, effective_frames - frames_y - 1)
        images[:, start_frame:start_frame + frames_y] = y
    elif frames_y > effective_frames:
        if pri_label == 'nocall':
            start_frame = random.randint(0, frames_y - effective_frames)
            images = y[:, start_frame:start_frame + effective_frames].astype(np.float32)
        else:
            file_t_probs = t_probs[mel_filename]
            start_second = np.random.choice(len(file_t_probs), size=1, p=file_t_probs)[0]
            start_frame = round(start_second * config.frames_per_1s)
                                # + random.randint(-config.frames_half_sec//4, config.frames_half_sec//4))
            start_frame = 0 if start_frame < 0 else start_frame

            remain_frames = frames_y - start_frame
            if remain_frames < effective_frames:  # Remain frames less than 5s_frames
                images_start = random.randint(0, effective_frames - remain_frames - 1)
                images[:, images_start:images_start + remain_frames] = y[:, start_frame:].astype(np.float32)
            else:
                images = y[:, start_frame:start_frame + effective_frames].astype(np.float32)
    else:
        images = y.astype(np.float32)
        start_frame = 0

    # assert(images.shape == (height_y, effective_frames))
    return images, start_frame


def visualize_mel(image, ogg_filename, start_sec, high_freq_ratio=0.0, desc=''):
    img = image.copy()
    img = img - img.min()
    img = img / img.max()

    if len(img.shape) == 2:
        # img = img[..., np.newaxis]

        spec_params = config_p.Params().spectro
        fig = plt.Figure()
        ax = fig.add_subplot(111)
        librosa.display.specshow(image, sr=spec_params['sr'], hop_length=spec_params['n_hop'], ax=ax,
                                 x_axis='time', y_axis='linear')
        fig.savefig(f"{config_g.DIR_VISUALIZE}/{ogg_filename}_start{start_sec:.1f}s_{desc}_{high_freq_ratio:.2f}.png")
    else:
        img = np.moveaxis(img, 0, 2)

        imgplot = plt.imshow(img)
        plt.savefig(f"{config_g.DIR_VISUALIZE}/{ogg_filename}_start{start_sec:.1f}s_{desc}_{high_freq_ratio:.2f}.png")


def save_audio_from_mel(mel_spec, config, ogg_filename, start_frame, bg_mel_filename=''):
    spec_params = config.spectro
    au = mel_to_audio(mel_spec, sr=spec_params['sr'], n_fft=spec_params['n_fft'], hop_length=spec_params['n_hop'],
                      win_length=spec_params['n_fft'], fmin=spec_params['fmin'])

    start_sec = start_frame / config.frames_per_1s
    sf.write(f"{config_g.DIR_VISUALIZE}/{ogg_filename}_start{start_sec:.1f}s_BG{bg_mel_filename}.wav",
             au, samplerate=config.spectro['sr'])


VISUALIZE = False


class TraShortAudioDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, config_params):
        self.df = df[['mel_filename', 'ogg_filename', 'pri_label', 'sec_labels']]
        self.config = config_params
        self.visualize_count = 5
        self.df_bg_noise = pd.read_csv(f"{config_g.DIR_BG_NOISE}/BG_noise_selection.csv")
        self.bg_noise_dict = {}
        self.level_noise = 0.05

        # (224, 1), value [1, decreasing...]
        self.pink_ratio = np.array([1 - np.arange(self.config.spectro['mel_bins'])/self.config.spectro['mel_bins']]).T

        # crop_probs = f"{config_g.DIR_ENERGY_PROBS}/{self.config.energy_probs_pkl}"
        crop_probs = f"{config_g.DIR_ENERGY_PROBS}/merged_probs.pkl"
        # crop_probs = f"{config_g.DIR_ENERGY_PROBS}/Stage2_energy_merge_oof_probs/{self.config.stage2_probs_pkl}"
        with open(crop_probs, 'rb') as file:
            self.t_probs = pickle.load(file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        mel_filename, ogg_filename, pri_label, sec_labels = self.df.\
            loc[idx, ['mel_filename', 'ogg_filename', 'pri_label', 'sec_labels']].to_numpy()

        image_h, image_w = self.config.spectro['mel_bins'], self.config.frames_per_period
        crop_frames = image_w  # + random.randint(-self.config.frames_half_sec, self.config.frames_half_sec)

        images = np.zeros((image_h, image_w), dtype=np.float32)
        while images.max() == 0:
            images, start_frame = get_energy_crop(mel_filename, pri_label, crop_frames, self.t_probs, self.config)

        # images = utils.random_power(images, 3, 0.5)  # Change contrast I

        # Add background noise without birds
        bg_idx = random.randint(0, len(self.df_bg_noise) - 1)
        bg_mel_file = self.df_bg_noise.loc[bg_idx, 'mel_filename']
        if bg_mel_file not in self.bg_noise_dict:
            bg_mel = torch.load(f"{config_g.DIR_BG_NOISE}/{bg_mel_file}").numpy()
            self.bg_noise_dict[f"{bg_mel_file}"] = bg_mel
            # print(f"Enter torch.load bg_noise {bg_mel_file}")
        else:
            bg_mel = self.bg_noise_dict[f"{bg_mel_file}"]

        bg_start = random.randint(0, bg_mel.shape[1] - crop_frames - 1)
        bg_mel = bg_mel[:, bg_start: bg_start + crop_frames]
        bg_mel = utils.random_power(bg_mel)
        images = images + bg_mel / (bg_mel.max()+0.0000001) * (random.random() + 0.5) * images.max()

        # power to db
        images = librosa.power_to_db(images, ref=np.max)  # convert S**2 to db[-80, 0]
        images = (images + 80) / 80  # Normalize to [0, 1]

        """
        # Add white noise, (image_h, crop_frames): [9, 10) * images.mean() * [0.3, 1.3) * 0.05
        if random.random() < 0.5:
            images = images + (np.random.random((image_h, crop_frames)).astype(np.float32) + 9) * 0.5 \
                     * images.mean() * (random.random() + 0.3) * self.level_noise
        """
        # Add pink noise
        if random.random() < 0.5:
            r = random.randint(1, image_h)
            pink_ratio = np.array([np.concatenate((1 - np.arange(r) / r, np.zeros(image_h - r)))]).T
            # pink_ratio = np.array([1 - np.arange(image_h)/image_h]).T  # (224, 1), value [1, decreasing...]
            images = images + (np.random.random((image_h, crop_frames)).astype(np.float32) + 9) \
                     * images.mean() * (random.random() + 0.3) * pink_ratio
        
        # Add bandpass noise
        if random.random() < 0.5:
            a = random.randint(0, image_h // 2)
            b = random.randint(a + 20, image_h)
            images[a:b, :] = images[a:b, :] + (np.random.random((b - a, crop_frames)).astype(np.float32) + 9) * 0.05 \
                             * images.mean() * (random.random() + 0.3) * self.level_noise
        """
        # Lower the high frequencies
        if random.random() < 0.5:
            images = images - images.min()
            # r: randomint from [112, 224], low-high freq boundary, pink_noise decrease from lowest freq until
            # freq r(1 to 0.xx), then keep the same 0.yy for all remain high freq
            # pink_noise: shape (224, 1), range [1, decreasing..., 0.xx at freq r, 0.yy...]
            r = random.randint(self.config.spectro['mel_bins']//2, self.config.spectro['mel_bins'])  # [112, 224]
            x = random.random()/2  # [0, 0.5)
            pink_ratio = np.array([np.concatenate((1-np.arange(r)*x/r,  # array (r,), value 1 decrease to 0.xx
                                                   np.zeros(self.config.spectro['mel_bins']-r)+1-x))]).T
            images = images * pink_ratio
            images = images/images.max()
        """
        # images = utils.random_power(images, 2, 0.7)  # Change contrast II
        # images = utils.random_power(images, 1.4, 0.6)  # Change contrast II

        if VISUALIZE and self.visualize_count > 0:  # and random.random() < 0.001:
            self.visualize_count -= 1
            # images = images.numpy()
            visualize_mel(images, ogg_filename, start_frame/self.config.frames_per_1s, desc='Final')

        # to 3 channels and normalize (3rd)
        images = utils.mono_to_color(images)  # Convert to 3 channels RGB array range in [0, 255], shape [H x W x 3]
        images = utils.normalize(images)  # Normalizes RGB array to range [0, 1], with shape [3 x H x W]

        # images = utils.mono_to_color(images, image_h, image_w)

        # Set target of pri_label to 1 and sec_labels to 0.3
        labels = np.zeros(config_g.NUM_TARGETS, dtype=np.float32)
        # sec_labels_mask = np.zeros(config_g.NUM_TARGETS, dtype=np.float32)
        for sec_label in sec_labels.split():
            labels[config_g.BIRD_CODE[sec_label]] = self.config.sec_label_target
            # sec_labels_mask[config_g.BIRD_CODE[sec_label]] = 1

        labels[config_g.BIRD_CODE[pri_label]] = 1

        # Temp fix for crop all 0 issue
        if images.max() == 0:
            print(f"### {mel_filename} is 0 !!!")
            labels[:] = 0

        return {'waveform': images,
                'targets': labels,
                # 'sec_labels_mask': sec_labels_mask,
                'filename': mel_filename}


class TraSoundscapeDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, config_params):
        self.df = df
        self.config = config_params
        self.ogg_to_mel = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row_id, seconds, ogg_name = self.df.loc[idx, ['row_id', 'seconds', 'filename']].to_numpy()

        if ogg_name not in self.ogg_to_mel:
            # print(f"Converting {ogg_name} to mel in worker_id: {torch.utils.data.get_worker_info().id}, "
            #      f"process id: {os.getpid()}, len(TRA_SS_OGG_MEL_DICT) is {len(self.ogg_to_mel)}")
            y_mel = utils.get_mel_spectrogram(f"{config_g.DIR_TRA_SOUNDSCAPES}/{ogg_name}", self.config.spectro)
            self.ogg_to_mel[f'{ogg_name}'] = y_mel
        else:
            y_mel = self.ogg_to_mel[f'{ogg_name}']

        start_frame = int((seconds - self.config.spectro['period']) * self.config.frames_per_1s)
        images = y_mel[:, start_frame:start_frame + self.config.frames_per_period].astype(np.float32)

        # Keep same as TraShortAudioDataset
        images = librosa.power_to_db(images, ref=np.max)  # convert S**2 to db[-80, 0]
        images = utils.mono_to_color(images)  # Convert to 3 channels RGB array range in [0, 255], shape [H x W x 3]
        images = utils.normalize(images)  # Normalizes RGB array to range [0, 1], with shape [3 x H x W]
        """
        images = (images + 80) / 80  # Normalize to [0, 1]
        image_h, image_w = self.config.spectro['mel_bins'], self.config.frames_per_period
        images = utils.mono_to_color(images, image_h, image_w)
        """

        # Hard label for both pri and sec labels
        labels = self.df.loc[idx, config_g.TARGET_COLS_WITHOUT_NOCALL].to_numpy().astype(np.float32)

        return {'waveform': images,
                'targets': labels,
                'row_ids': row_id}

