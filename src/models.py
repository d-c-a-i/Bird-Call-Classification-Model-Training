import torch
import torch.nn as nn
import timm
import random
import numpy as np
import config.globals as config_g
import config.params_baseline as params

# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from torch.cuda.amp import autocast, GradScaler

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


""" # Mixup from PANNs, input x -> 0.5 size odd-even mixed x
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=1234):
        # Mixup coefficient generator.
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    def get_lambda(self, batch_size):
        '''Get mixup random coefficients. # batch_size大小的array: [lam1, 1-lam1, lam2, 1-lam2, ...]
        Args:
          batch_size: int
        Returns:
          mixup_lambdas: (batch_size,)
        '''
        mixup_lambdas = []
        for n in range(0, batch_size, 2):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]  # beta分布，均匀取值(0, 1)
            mixup_lambdas.append(lam)
            mixup_lambdas.append(1. - lam)

        return np.array(mixup_lambdas)
        # return torch.tensor(mixup_lambdas, requires_grad=False).float().to(device) # modified by 1st CornellBird

def do_mixup(x, mixup_lambda):
    '''
    Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes (1, 3, 5, ...).

    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)

    Returns:
      out: (batch_size, ...)
    '''
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out

# in train():
if 'mixup' in augmentation:
    mixup_augmenter = Mixup(mixup_alpha=1.)
    batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(batch_size=len(batch_data_dict['waveform']))
    batch_output_dict = model(batch_data_dict['waveform'], batch_data_dict['mixup_lambda'])
    batch_target_dict = {'target': do_mixup(batch_data_dict['target'], batch_data_dict['mixup_lambda'])}
"""  # Mixup from PANNs: input x -> 0.5 size odd-even mixed x

""" # My random Mixup, input x -> 0.5 size random mixed x + 1 size original x
class Mixup(object):
    def __init__(self, mixup_alpha, random_seed=40):
        # Mixup coefficient generator.
        self.mixup_alpha = mixup_alpha
        self.random_state = np.random.RandomState(random_seed)

    # Random mixup
    def get_lambda(self, batch_size, device):
        # Get mixup random coefficients
        idx_input = np.arange(batch_size)
        self.random_state.shuffle(idx_input)
        # print(f"In get_lambda(): getpid: {os.getpid()}, shuffled idx_input is {idx_input}")
        _idx_a, _idx_b = np.split(idx_input, 2)
        assert (batch_size == len(_idx_a) + len(_idx_b))

        mixup_lambdas = [0] * batch_size
        for i in range(len(_idx_a)):
            lam = self.random_state.beta(self.mixup_alpha, self.mixup_alpha, 1)[0]  # beta分布，均匀取值(0, 1)
            mixup_lambdas[_idx_a[i]], mixup_lambdas[_idx_b[i]] = lam, 1.0 - lam

        return torch.tensor(mixup_lambdas, requires_grad=False).float().to(device), _idx_a, _idx_b


def do_mixup(x, mixup_lambda):
    # Random mixup
    idx_a, idx_b = mixup_lambda['idx_a'], mixup_lambda['idx_b']
    mixed = (x[idx_a].transpose(0, -1) * mixup_lambda['lambda'][idx_a] +
             x[idx_b].transpose(0, -1) * mixup_lambda['lambda'][idx_b]).transpose(0, -1)

    out = torch.cat((x, mixed))
    return out
"""  # My random Mixup: input x -> 0.5 size random mixed x + 1 size original x


# Mixup from 3rd place of Cornell Birdcall: input x -> 0.5 size random mixed x + 0.5 size original x
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    """
    Arguments:
        x {torch tensor} -- Input batch
        y {torch tensor} -- Labels
        alpha {float} -- Parameter of the beta distribution (default: {0.4})
    Returns:
        torch tensor  -- Mixed input
        torch tensor  -- Labels of the original batch
        torch tensor  -- Labels of the shuffle batch
        float  -- Probability samples by the beta distribution
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    # print(f"lam = {lam}")

    index = torch.randperm(x.size()[0], device=DEVICE)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_original, y_permuted = y, y[index]

    return mixed_x, y_original, y_permuted, lam


class BirdCLEFModel(nn.Module):
    def __init__(self, config: params.Params, base_model='efficientnet_b0', pretrain=True):
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
        self.fc = nn.Linear(in_features, config_g.NUM_TARGETS)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc)

    def forward(self, input_x, y_batch=None):  # input_x: (bs, 3, 224, 448)
        x = input_x

        if self.training and self.config.transforms['mixup_proba'] > 0:  # Mixup on spectrogram
            with torch.no_grad():
                if random.random() < self.config.transforms['mixup_proba']:
                    # print(">>>>>>> do mixup")
                    x, y_original, y_permuted, _ = mixup_data(x, y_batch, alpha=self.config.transforms['mixup_alpha'])
                    y_batch = torch.clamp(y_original + y_permuted, 0, 1)

                    if config_g.USE_NOCALL:
                        for i in range(len(y_batch)):
                            if y_batch[i, :-1].sum() > 0:
                                y_batch[i, -1] = 0

        x = self.base_model.forward_features(x)  # output: (bs, 1280, 7, 14)

        x = self.avg_pool(x)  # output: (bs, 1280, 1, 1)
        x = x.flatten(1)  # output: (bs, 1280)
        x = self.dropout(x)
        x = self.fc(x)  # output: (bs, 397)

        return x, y_batch
