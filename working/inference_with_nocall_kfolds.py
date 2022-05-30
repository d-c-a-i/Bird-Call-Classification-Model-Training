"""
!pip install ../input/timm-pytorch-image-models/pytorch-image-models-master/
!pip install ../input/torchlibrosa/torchlibrosa-0.0.9-py2.py3-none-any.whl
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import librosa
import logging
import torch.utils.data as data
import pickle

from time import time
from datetime import timedelta
from pathlib import Path
from torchlibrosa.augmentation import SpecAugmentation
from contextlib import contextmanager
from typing import Optional
from sklearn.metrics import f1_score, precision_score, recall_score


######################## Configs ########################
USE_NOCALL = False
SAMPLE_RATE = 32000

BIRD_CODE = {
    'acafly': 0, 'acowoo': 1, 'aldfly': 2, 'ameavo': 3, 'amecro': 4,
    'amegfi': 5, 'amekes': 6, 'amepip': 7, 'amered': 8, 'amerob': 9,
    'amewig': 10, 'amtspa': 11, 'andsol1': 12, 'annhum': 13, 'astfly': 14,
    'azaspi1': 15, 'babwar': 16, 'baleag': 17, 'balori': 18, 'banana': 19,
    'banswa': 20, 'banwre1': 21, 'barant1': 22, 'barswa': 23, 'batpig1': 24,
    'bawswa1': 25, 'bawwar': 26, 'baywre1': 27, 'bbwduc': 28, 'bcnher': 29,
    'belkin1': 30, 'belvir': 31, 'bewwre': 32, 'bkbmag1': 33, 'bkbplo': 34,
    'bkbwar': 35, 'bkcchi': 36, 'bkhgro': 37, 'bkmtou1': 38, 'bknsti': 39,
    'blbgra1': 40, 'blbthr1': 41, 'blcjay1': 42, 'blctan1': 43, 'blhpar1': 44,
    'blkpho': 45, 'blsspa1': 46, 'blugrb1': 47, 'blujay': 48, 'bncfly': 49,
    'bnhcow': 50, 'bobfly1': 51, 'bongul': 52, 'botgra': 53, 'brbmot1': 54,
    'brbsol1': 55, 'brcvir1': 56, 'brebla': 57, 'brncre': 58, 'brnjay': 59,
    'brnthr': 60, 'brratt1': 61, 'brwhaw': 62, 'brwpar1': 63, 'btbwar': 64,
    'btnwar': 65, 'btywar': 66, 'bucmot2': 67, 'buggna': 68, 'bugtan': 69,
    'buhvir': 70, 'bulori': 71, 'burwar1': 72, 'bushti': 73, 'butsal1': 74,
    'buwtea': 75, 'cacgoo1': 76, 'cacwre': 77, 'calqua': 78, 'caltow': 79,
    'cangoo': 80, 'canwar': 81, 'carchi': 82, 'carwre': 83, 'casfin': 84,
    'caskin': 85, 'caster1': 86, 'casvir': 87, 'categr': 88, 'ccbfin': 89,
    'cedwax': 90, 'chbant1': 91, 'chbchi': 92, 'chbwre1': 93, 'chcant2': 94,
    'chispa': 95, 'chswar': 96, 'cinfly2': 97, 'clanut': 98, 'clcrob': 99,
    'cliswa': 100, 'cobtan1': 101, 'cocwoo1': 102, 'cogdov': 103, 'colcha1': 104,
    'coltro1': 105, 'comgol': 106, 'comgra': 107, 'comloo': 108, 'commer': 109,
    'compau': 110, 'compot1': 111, 'comrav': 112, 'comyel': 113, 'coohaw': 114,
    'cotfly1': 115, 'cowscj1': 116, 'cregua1': 117, 'creoro1': 118, 'crfpar': 119,
    'cubthr': 120, 'daejun': 121, 'dowwoo': 122, 'ducfly': 123, 'dusfly': 124,
    'easblu': 125, 'easkin': 126, 'easmea': 127, 'easpho': 128, 'eastow': 129,
    'eawpew': 130, 'eletro': 131, 'eucdov': 132, 'eursta': 133, 'fepowl': 134,
    'fiespa': 135, 'flrtan1': 136, 'foxspa': 137, 'gadwal': 138, 'gamqua': 139,
    'gartro1': 140, 'gbbgul': 141, 'gbwwre1': 142, 'gcrwar': 143, 'gilwoo': 144,
    'gnttow': 145, 'gnwtea': 146, 'gocfly1': 147, 'gockin': 148, 'gocspa': 149,
    'goftyr1': 150, 'gohque1': 151, 'goowoo1': 152, 'grasal1': 153, 'grbani': 154,
    'grbher3': 155, 'grcfly': 156, 'greegr': 157, 'grekis': 158, 'grepew': 159,
    'grethr1': 160, 'gretin1': 161, 'greyel': 162, 'grhcha1': 163, 'grhowl': 164,
    'grnher': 165, 'grnjay': 166, 'grtgra': 167, 'grycat': 168, 'gryhaw2': 169,
    'gwfgoo': 170, 'haiwoo': 171, 'heptan': 172, 'hergul': 173, 'herthr': 174,
    'herwar': 175, 'higmot1': 176, 'hofwoo1': 177, 'houfin': 178, 'houspa': 179,
    'houwre': 180, 'hutvir': 181, 'incdov': 182, 'indbun': 183, 'kebtou1': 184,
    'killde': 185, 'labwoo': 186, 'larspa': 187, 'laufal1': 188, 'laugul': 189,
    'lazbun': 190, 'leafly': 191, 'leasan': 192, 'lesgol': 193, 'lesgre1': 194,
    'lesvio1': 195, 'linspa': 196, 'linwoo1': 197, 'littin1': 198, 'lobdow': 199,
    'lobgna5': 200, 'logshr': 201, 'lotduc': 202, 'lotman1': 203, 'lucwar': 204,
    'macwar': 205, 'magwar': 206, 'mallar3': 207, 'marwre': 208, 'mastro1': 209,
    'meapar': 210, 'melbla1': 211, 'monoro1': 212, 'mouchi': 213, 'moudov': 214,
    'mouela1': 215, 'mouqua': 216, 'mouwar': 217, 'mutswa': 218, 'naswar': 219,
    'norcar': 220, 'norfli': 221, 'normoc': 222, 'norpar': 223, 'norsho': 224,
    'norwat': 225, 'nrwswa': 226, 'nutwoo': 227, 'oaktit': 228, 'obnthr1': 229,
    'ocbfly1': 230, 'oliwoo1': 231, 'olsfly': 232, 'orbeup1': 233, 'orbspa1': 234,
    'orcpar': 235, 'orcwar': 236, 'orfpar': 237, 'osprey': 238, 'ovenbi1': 239,
    'pabspi1': 240, 'paltan1': 241, 'palwar': 242, 'pasfly': 243, 'pavpig2': 244,
    'phivir': 245, 'pibgre': 246, 'pilwoo': 247, 'pinsis': 248, 'pirfly1': 249,
    'plawre1': 250, 'plaxen1': 251, 'plsvir': 252, 'plupig2': 253, 'prowar': 254,
    'purfin': 255, 'purgal2': 256, 'putfru1': 257, 'pygnut': 258, 'rawwre1': 259,
    'rcatan1': 260, 'rebnut': 261, 'rebsap': 262, 'rebwoo': 263, 'redcro': 264,
    'reevir1': 265, 'rehbar1': 266, 'relpar': 267, 'reshaw': 268, 'rethaw': 269,
    'rewbla': 270, 'ribgul': 271, 'rinkin1': 272, 'roahaw': 273, 'robgro': 274,
    'rocpig': 275, 'rotbec': 276, 'royter1': 277, 'rthhum': 278, 'rtlhum': 279,
    'ruboro1': 280, 'rubpep1': 281, 'rubrob': 282, 'rubwre1': 283, 'ruckin': 284,
    'rucspa1': 285, 'rucwar': 286, 'rucwar1': 287, 'rudpig': 288, 'rudtur': 289,
    'rufhum': 290, 'rugdov': 291, 'rumfly1': 292, 'runwre1': 293, 'rutjac1': 294,
    'saffin': 295, 'sancra': 296, 'sander': 297, 'savspa': 298, 'saypho': 299,
    'scamac1': 300, 'scatan': 301, 'scbwre1': 302, 'scptyr1': 303, 'scrtan1': 304,
    'semplo': 305, 'shicow': 306, 'sibtan2': 307, 'sinwre1': 308, 'sltred': 309,
    'smbani': 310, 'snogoo': 311, 'sobtyr1': 312, 'socfly1': 313, 'solsan': 314,
    'sonspa': 315, 'soulap1': 316, 'sposan': 317, 'spotow': 318, 'spvear1': 319,
    'squcuc1': 320, 'stbori': 321, 'stejay': 322, 'sthant1': 323, 'sthwoo1': 324,
    'strcuc1': 325, 'strfly1': 326, 'strsal1': 327, 'stvhum2': 328, 'subfly': 329,
    'sumtan': 330, 'swaspa': 331, 'swathr': 332, 'tenwar': 333, 'thbeup1': 334,
    'thbkin': 335, 'thswar1': 336, 'towsol': 337, 'treswa': 338, 'trogna1': 339,
    'trokin': 340, 'tromoc': 341, 'tropar': 342, 'tropew1': 343, 'tuftit': 344,
    'tunswa': 345, 'veery': 346, 'verdin': 347, 'vigswa': 348, 'warvir': 349,
    'wbwwre1': 350, 'webwoo1': 351, 'wegspa1': 352, 'wesant1': 353, 'wesblu': 354,
    'weskin': 355, 'wesmea': 356, 'westan': 357, 'wewpew': 358, 'whbman1': 359,
    'whbnut': 360, 'whcpar': 361, 'whcsee1': 362, 'whcspa': 363, 'whevir': 364,
    'whfpar1': 365, 'whimbr': 366, 'whiwre1': 367, 'whtdov': 368, 'whtspa': 369,
    'whwbec1': 370, 'whwdov': 371, 'wilfly': 372, 'willet1': 373, 'wilsni1': 374,
    'wiltur': 375, 'wlswar': 376, 'wooduc': 377, 'woothr': 378, 'wrenti': 379,
    'y00475': 380, 'yebcha': 381, 'yebela1': 382, 'yebfly': 383, 'yebori1': 384,
    'yebsap': 385, 'yebsee1': 386, 'yefgra1': 387, 'yegvir': 388, 'yehbla': 389,
    'yehcar1': 390, 'yelgro': 391, 'yelwar': 392, 'yeofly1': 393, 'yerwar': 394,
    'yeteup1': 395, 'yetvir': 396,
}
if USE_NOCALL:
    BIRD_CODE.update({'nocall': 397})

NUM_TARGETS = len(BIRD_CODE)
TARGET_COLS = list(BIRD_CODE.keys())
TARGET_COLS_WITHOUT_NOCALL = TARGET_COLS[:-1] if USE_NOCALL else TARGET_COLS[:]

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

root_dir = Path.cwd().parent
CSV_TRA_META = str(root_dir.joinpath(f'./input/birdclef-2021/train_metadata.csv'))
CSV_TRA_SOUNDSCAPE = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscape_labels.csv'))
CSV_SAMPLE_SUB = str(root_dir.joinpath(f'./input/birdclef-2021/sample_submission.csv'))
CSV_TEST = str(root_dir.joinpath(f'./input/birdclef-2021/test.csv'))

DIR_TRA_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/train_soundscapes/'))
DIR_TRA_SHORT_AUDIO = str(root_dir.joinpath(f'./input/birdclef-2021/train_short_audio/'))
DIR_TEST_SOUNDSCAPES = str(root_dir.joinpath(f'./input/birdclef-2021/test_soundscapes/'))
DIR_MEL = str(root_dir.joinpath(f'./working/mel_spectrogram/'))
DIR_WORKING = str(root_dir.joinpath(f'./working/'))
DIR_OUTPUT = str(root_dir.joinpath(f'./working/output/'))
DIR_TRAINED_MODELS = str(root_dir.joinpath(f'./input/chbird-kfolds/'))


class Params(object):
    def __init__(self):
        self.seed = 40
        self.f1_threshold = 0.4  # 0.35 f1_threshold: tra_ss_f1_score: 0.7254, tra_ss_prec_score: 0.9500, tra_ss_recall_score: 0.7508
        self.nocall_threshold = 0.5  # 0.3~0.95 is the same!
        self.votes_threshold = 3  # 3+ vote out of 5 is positive
        self.base_models = ['rexnet_100']  # , 'efficientnet_b0']
        self.ensemble = 'average'  # 'voting' or 'average'

        # Training related
        self.period = 5
        self.epochs = 30
        self.nfolds = 5
        self.lr = 1e-3
        self.dropout = 0.3
        self.early_stop = 0

        # Dataset and dataloader
        self.dataset = {
            'test': TestSoundscapeDataset
        }

        self.test_loader = {
            'batch_size': 320,
            'shuffle': False,
            'num_workers': 4,
            'drop_last': False,
            'pin_memory': True
        }

        self.transforms = {
            'apply_aug': False,
            'mixup_proba': 0.5,
            'mixup_alpha': 5  # TODO
        }

        # Spectrogram related
        self.spectro = {
            'period': 5,
            'sr': 21952,  # 32000,
            'n_fft': 892,  # 2048
            'n_hop': 245,  # 512
            'mel_bins': 224,  # 128
            'fmin': 300  # 20
            # 'fmax': 16000
        }
        self.frames_per_1s = self.spectro['sr'] / self.spectro['n_hop']
        self.frames_half_sec = int(self.frames_per_1s / 2)
        self.frames_per_period = int(self.frames_per_1s * self.spectro['period'])


######################## From utils.py ########################
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


def get_mel_spectrogram(wave_file: str, spec_params: dict):
    y, _ = librosa.load(wave_file, sr=spec_params['sr'], mono=True, res_type="kaiser_fast")

    # Create melspectrogram
    spectro = librosa.feature.melspectrogram(y, sr=spec_params['sr'], n_mels=spec_params['mel_bins'],
                                             n_fft=spec_params['n_fft'], hop_length=spec_params['n_hop'],
                                             win_length=spec_params['n_fft'], fmin=spec_params['fmin'])
    return spectro.astype(np.float16)


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
        msg = f"[{name}] DONE in {timedelta(seconds=round(time() - t0))}"
        logging_msg(msg, logger)
    else:
        yield


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


"""
def convert_nocall_preds(y_preds: np.ndarray):  # (num_samples, num_targets)
    assert (y_preds.shape[1] == NUM_TARGETS and y_preds.min() == 0 and y_preds.max() == 1)

    num_samples = len(y_preds)
    for i in range(num_samples):
        if y_preds[i, BIRD_CODE['nocall']] == 1:
            y_preds[i, :] = 0

    y_preds = y_preds[:, :-1]  # remove 'nocall'

    return y_preds
"""
def convert_nocall_logits(y_logits: np.ndarray):  # (num_samples, num_targets)
    assert(y_logits.shape[1] == NUM_TARGETS)
    y_preds = y_logits.copy()

    num_samples = len(y_logits)
    for i in range(num_samples):
        """
        if y_logits[i, config_g.BIRD_CODE['nocall']] == 1:
            y_logits[i, :] = 0
        """
        # If 'nocall' has maximum logits, then set all preiction to 0
        if y_logits[i].max() == y_logits[i, BIRD_CODE['nocall']]:
            y_preds[i, :] = 0
        else:
            y_preds[i, :] = sigmoid(y_preds[i, :])

    y_preds = y_preds[:, :-1]  # remove 'nocall' column
    binary_output = (y_preds > Params().f1_threshold).astype('int32')

    return binary_output


######################## From models.py ########################
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


class BirdCLEFModel(nn.Module):
    def __init__(self, config: Params, base_model='efficientnet_b0', pretrain=True):
        super(BirdCLEFModel, self).__init__()

        self.config = config
        self.base_model = timm.create_model(base_model, pretrained=pretrain)
        if base_model.startswith('efficientnet'):
            in_features = self.base_model.classifier.in_features  # self.base_model.fc.in_features
        elif base_model.startswith('rexnet'):
            in_features = self.base_model.head.fc.in_features
        else:
            raise NotImplementedError

        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2,
                                               freq_drop_width=8, freq_stripes_num=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(in_features, NUM_TARGETS)

        self.init_weight()

    def init_weight(self):
        # init_bn(self.bn)
        init_layer(self.fc)

    def forward(self, input_x, y_batch=None):  # input_x: (bs, 3, 224, 448)
        """
        if self.training and self.trans_config['mixup_proba'] > 0:  # Mixup on spectrogram
            with torch.no_grad():
                if np.random.rand() < self.trans_config['mixup_proba']:
                    # print(">>>>>>> do mixup")
                    x, y_original, y_permuted, _ = mixup_data(x, y_batch, alpha=self.trans_config['mixup_alpha'])
                    y_batch = torch.clamp(y_original + y_permuted, 0, 1)
        """
        x = self.base_model.forward_features(input_x)  # output: (bs, 1280, 7, 14)

        x = self.avg_pool(x)  # output: (bs, 1280, 1, 1)
        x = x.flatten(1)  # output: (bs, 1280)
        x = self.dropout(x)
        x = self.fc(x)  # output: (bs, 397)

        return x, y_batch


######################## Test Dataset ########################
def get_loader(phase: str, df: pd.DataFrame, config, data_dir=None):
    if phase == 'train':
        dataset = config.dataset['train'](df, config)
        loader_config = config.tra_loader
    elif phase == 'valid':
        dataset = config.dataset['valid'](df, config)
        loader_config = config.val_loader
    elif phase == 'test':
        dataset = config.dataset['test'](df, data_dir, config)
        loader_config = config.test_loader
    else:
        raise NotImplementedError

    loader = data.DataLoader(dataset, **loader_config)
    return loader


class TestSoundscapeDataset(data.Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: str, config_params):
        self.df = df
        self.data_dir = data_dir
        self.config = config_params
        self.audio_id_to_mel = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row_id, site, audio_id, seconds = self.df.loc[idx, ['row_id', 'site', 'audio_id', 'seconds']].to_numpy()

        if audio_id not in self.audio_id_to_mel:
            found_ogg = list(OGG_PATH.glob(f"{audio_id}_{site}*.ogg"))
            assert (len(found_ogg) == 1)
            ogg_name = found_ogg[0].name

            # print(f"Converting {ogg_name} to mel in worker_id: {torch.utils.data.get_worker_info().id}, "
            #      f"process id: {os.getpid()}, len(TRA_SS_OGG_MEL_DICT) is {len(self.audio_id_to_mel)}")
            y_mel = get_mel_spectrogram(f"{self.data_dir}/{ogg_name}", self.config.spectro)
            self.audio_id_to_mel[audio_id] = y_mel
        else:
            y_mel = self.audio_id_to_mel[audio_id]

        start_frame = int((seconds - self.config.spectro['period']) * self.config.frames_per_1s)
        images = y_mel[:, start_frame:start_frame + self.config.frames_per_period].astype(np.float32)

        # Keep same as TraShortAudioDataset
        images = librosa.power_to_db(images, ref=np.max)  # convert S**2 to db[-80, 0]
        images = mono_to_color(images)  # Convert to 3 channels RGB array range in [0, 255], shape [H x W x 3]
        images = normalize(images)  # Normalizes RGB array to range [0, 1], with shape [3 x H x W]

        return {'waveform': images,
                'row_ids': row_id}


######################## From tra_val_test.py ########################
class MetricMeter(object):  # F1 score as competition metric
    def __init__(self):
        # self.y_pred = []
        self.y_logits = []

    def update(self, y_pred):
        y_preds = y_pred.detach().cpu()
        self.y_logits.extend(y_preds.numpy().tolist())
        # self.y_pred.extend(torch.sigmoid(y_preds).numpy().tolist())

    """
    @property
    def preds(self):
        return np.array(self.y_pred)
    """

    @property
    def logits(self):
        return np.array(self.y_logits)


def inference_on_test_soundscape(model, loader):
    model.eval()
    row_ids = []
    metric = MetricMeter()

    with torch.no_grad():
        for sample in loader:
            inputs = sample['waveform'].to(DEVICE)

            preds, _ = model(inputs)
            metric.update(preds)

            row_ids.extend(sample['row_ids'])

    return metric.logits, np.array(row_ids)


IN_TEST = (len(list(Path(DIR_TEST_SOUNDSCAPES).glob("*.ogg"))) != 0)
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold0.bin"  # 0.54
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold0_energytrimming_0502.bin"  # 0.57
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_energytrim_mixup3rd_0502_fold0.bin"  # 0.62
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup2nd_fold0_0503.bin"  # 0.62
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup3rd_3files_fold0_0503.bin"  # 0.60
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup3rd_RandomPower2ndParam_fold0.bin"  # 0.64
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup3rd_RandomPowerAfterPtoDb_fold0.bin"  # 0.62
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup3rd_RandomPowerAfterPtoDb_05_35_fold0.bin"  # 0.62
# Trained_Model = f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet100_energytrim_mixup3rd_RanPow2ndParam_BGNoise_fold0.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergyTrim_mixup3rd_RanPow2ndParam_BGNoise_fold0.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergyTrim_mixup3rd_RanPow2ndParam_BGNoise_Rm33Files_fold0.bin"  # 0.67
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup3rd_RanPow2nd_BGNoise_Rm33Files_LowerHiFreq_fold0.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup0p6_RanPow2nd_BGNoise_Rm33Files_fold0.bin"  # 0.67
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup3rd_RanPow2nd_BGNoise_Rm34Files_188Manual_fold0.bin"  # 0.67
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup3rd_RanPow2nd_BGNoise_Rm34Files_188Manual_StartFrame_fold0.bin"  # 0.67
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup3rd_RanPow2nd_Rm34Files_188Manual_SFrame_BGNoiseMore_fold0v.bin"  # 0.65
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergyFmin500_mixup3rd_RanPow2nd_BGNoise_Rm33Files_fold0.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/rexnet100_MelEnergy_mixup3rd_RanPow2nd_BGNoise_Rm34Files_fmin200.bin"  # 0.64
# Trained_Model = f"{DIR_TRAINED_MODELS}/Rebaseline_fold0.bin"  # 0.67
# Trained_Model = f"{DIR_TRAINED_MODELS}/Rebaseline_SecLabel0.45.bin"  # 0.67, moveup 1 place
# Trained_Model = f"{DIR_TRAINED_MODELS}/Rebaseline_SecLabel0.5_PinkBandNoise.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/Rebaseline_SecLabel0.5_fold0.bin" # 0.65
# Trained_Model = f"{DIR_TRAINED_MODELS}/PosWeight12_fold0.bin"  # 0.66
# Trained_Model = f"{DIR_TRAINED_MODELS}/Nocall_ModifyMix_fold0.bin"
"""
Trained_Models = [f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold0.bin",
                  f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold1.bin",
                  f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold2.bin",
                  f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold3.bin",
                  f"{DIR_TRAINED_MODELS}/BirdCLEF_rexnet_100_fold4.bin"]
"""
Trained_Models = [f"{DIR_OUTPUT}/BirdCLEF_rexnet_100_fold0.bin",
                  f"{DIR_OUTPUT}/BirdCLEF_rexnet_100_fold1.bin",
                  f"{DIR_OUTPUT}/BirdCLEF_rexnet_100_fold2.bin",
                  f"{DIR_OUTPUT}/BirdCLEF_rexnet_100_fold3.bin",
                  f"{DIR_OUTPUT}/BirdCLEF_rexnet_100_fold4.bin"]
#Trained_Models = [f"{DIR_WORKING}/output/BirdCLEF_rexnet_100_fold0.bin"]

if __name__ == '__main__':
    params = Params()

    # Prepare test dataset and loader
    if IN_TEST:
        OGG_PATH = Path(DIR_TEST_SOUNDSCAPES)
        df_test = pd.read_csv(CSV_TEST)
        data_dir = DIR_TEST_SOUNDSCAPES
    else:
        OGG_PATH = Path(DIR_TRA_SOUNDSCAPES)
        df_test = pd.read_csv(CSV_TRA_SOUNDSCAPE)
        data_dir = DIR_TRA_SOUNDSCAPES

    test_loader = get_loader('test', df_test, params, data_dir)

    # Inference with trained models and get ensembled logits
    logits_all = []
    for model in Trained_Models:
        with timer(f"Inference with {model}"):
            # Load trained models
            bird_test_model = BirdCLEFModel(params, base_model=params.base_models[0], pretrain=False)
            bird_test_model.load_state_dict(torch.load(f"{model}", map_location=DEVICE))
            bird_test_model = bird_test_model.to(DEVICE)

            pred_logits, test_row_ids = inference_on_test_soundscape(bird_test_model, test_loader)
            assert (np.array_equal(df_test['row_id'].to_numpy(), test_row_ids))
            logits_all.append(pred_logits)

    logits_all = np.stack(logits_all)

    with open("./logits_all.pkl", 'wb') as file:
        pickle.dump(logits_all, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open("./logits_all.pkl", 'rb') as file:
        logits_all = pickle.load(file)

    if params.ensemble == 'average':
        ensemble_logits = logits_all.mean(axis=0)
        binary_outputs = convert_nocall_logits(ensemble_logits) if USE_NOCALL \
            else (sigmoid(ensemble_logits) > params.f1_threshold).astype('int32')
    else:  # voting ensemble
        logits_all = sigmoid(logits_all)
        if USE_NOCALL:
            binary_all = (logits_all[:, :, :-1] > params.f1_threshold).astype('int32')
            is_nocall = (logits_all[:, :, -1:] > params.nocall_threshold).astype('int32')
            binary_all = binary_all * (is_nocall ^ 1)  # Clean all targets if nocall
        else:
            binary_all = (logits_all > params.f1_threshold).astype('int32')

        binary_outputs = binary_all.sum(axis=0)
        binary_outputs = (binary_outputs >= params.votes_threshold).astype('int32')

    # Prepare submission
    df_sub = pd.read_csv(CSV_SAMPLE_SUB)[0:0]
    df_sub['row_id'] = df_test['row_id'].to_numpy()
    df_sub['birds'] = ''

    positive_idx = binary_outputs.nonzero()
    for row_idx, col_idx in zip(positive_idx[0], positive_idx[1]):
        df_sub.loc[row_idx, 'birds'] += f"{INV_BIRD_CODE[col_idx]} "

    df_sub['birds'].replace('', 'nocall', inplace=True)
    df_sub.to_csv('submission.csv', index=False)
    print(f"df_sub.shape is {df_sub.shape}")

    # Verify df_sub
    for idx, row in df_sub.iterrows():
        if row['birds'] == 'nocall':
            assert (binary_outputs[idx, :].sum() == 0)
        else:
            assert (binary_outputs[idx, :].sum() == len(row['birds'].split()))

    if not IN_TEST:
        # Calculate F1/Precision/Recall value
        df_tra_soundscape = pd.read_csv(f"{DIR_WORKING}/df_tra_soundscape.csv")
        y_true = df_tra_soundscape[TARGET_COLS_WITHOUT_NOCALL].to_numpy().astype('int32')

        if params.ensemble == 'average':
            df_f1thresh_search = pd.DataFrame(columns=['thresh', 'F1', 'prec', 'recall'])

            for f1_thresh in np.arange(0.30, 0.60, 0.01):
                f1_thresh = round(f1_thresh, 2)
                binary_pred = (sigmoid(ensemble_logits) > f1_thresh).astype('int32')
                ss_f1 = f1_score(y_true, binary_pred, average='samples', zero_division=1)
                ss_prec = precision_score(y_true, binary_pred, average='samples', zero_division=1)
                ss_recall = recall_score(y_true, binary_pred, average='samples', zero_division=1)
                df_f1thresh_search.loc[len(df_f1thresh_search)] = f"{f1_thresh:.2f}", f"{ss_f1:.4f}", \
                                                                  f"{ss_prec:.4f}", f"{ss_recall:.4f}",
            df_f1thresh_search.to_csv("df_f1thresh_search.csv", index=False)

        tra_ss_f1_score = f1_score(y_true, binary_outputs, average='samples', zero_division=1)
        tra_ss_prec_score = precision_score(y_true, binary_outputs, average='samples', zero_division=1)
        tra_ss_recall_score = recall_score(y_true, binary_outputs, average='samples', zero_division=1)
        print(f"With {params.f1_threshold} f1_threshold - tra_ss_f1_score: {tra_ss_f1_score:.4f}, tra_ss_prec_score: "
              f"{tra_ss_prec_score:.4f}, tra_ss_recall_score: {tra_ss_recall_score:.4f}")
