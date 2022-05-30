from src.datasets import TraShortAudioDataset, TraSoundscapeDataset


class Params(object):
    def __init__(self):
        self.seed = 40
        self.f1_threshold = 0.5
        self.base_models = ['rexnet_100']  # , 'efficientnet_b0']

        # Training related
        self.period = 5
        self.epochs = 40
        self.nfolds = 5
        self.lr = 1e-3
        self.dropout = 0.3
        self.early_stop = 0
        self.energy_probs_pkl = 'mel_energy_probs_uniform.pkl'
        self.stage2_probs_pkl = 'merged_energy_oof_probs.pkl'
        self.sec_label_target = 0.4

        # Dataset and dataloader
        self.dataset = {
            'train': TraShortAudioDataset,
            'valid': TraShortAudioDataset,
            'test': TraSoundscapeDataset
        }

        self.tra_loader = {
            'batch_size': 160,
            'shuffle': True,
            'num_workers': 8,
            'drop_last': False,
            'pin_memory': True
        }
        self.val_loader = {
            'batch_size': 160,
            'shuffle': False,
            'num_workers': 8,
            'drop_last': False,
            'pin_memory': True
        }
        self.test_loader = {
            'batch_size': 32,
            'shuffle': False,
            'num_workers': 4,
            'drop_last': False,
            'pin_memory': True
        }

        self.transforms = {
            'apply_aug': False,
            'mixup_proba': 0.5,
            'mixup_alpha': 5
        }

        # Spectrogram related
        self.spectro = {
            'period': 5,
            'sr': 21952,  # 32000,
            'n_fft': 892,  # 2048
            'n_hop': 245,  # 512
            'mel_bins': 224,  # 128
            'fmin': 300,  # 20
            # 'fmax': 16000
        }
        self.frames_per_1s = self.spectro['sr']/self.spectro['n_hop']
        self.frames_half_sec = int(self.frames_per_1s/2)
        self.frames_per_period = int(self.frames_per_1s * self.spectro['period'])

