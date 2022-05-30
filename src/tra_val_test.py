import numpy as np
import pandas as pd
import torch
import src.datasets as datasets
import src.models as models
import src.utils as utils
import src.criterions as criterions
import config.globals as config_g
import config.params_baseline as params

from torch.nn import BCEWithLogitsLoss
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class AverageMeter(object):  # Computes and stores the average value
    def __init__(self, training=True):
        self.avg, self.sum, self.count = torch.tensor(0, device=DEVICE, dtype=torch.float), \
                                         torch.tensor(0, device=DEVICE, dtype=torch.float), \
                                         torch.tensor(0, device=DEVICE, dtype=torch.float)

    def update(self, val, n):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MetricMeter(object):  # F1 score as competition metric
    def __init__(self, threshold=0.5):
        self.y_true, self.y_pred, self.y_logits, self._score, self.threshold = [], [], [], 0, threshold

    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.detach().cpu().numpy().tolist())
        y_preds = y_pred.detach().cpu()
        self.y_logits.extend(y_preds.numpy().tolist())
        self.y_pred.extend(torch.sigmoid(y_preds).numpy().tolist())

    @property
    def score(self):
        if config_g.USE_NOCALL:
            y_preds = utils.convert_nocall_logits(np.array(self.y_logits))
        else:
            y_preds = (np.array(self.y_pred) > self.threshold).astype('int32')

        self._score = f1_score(np.array(self.y_true).astype('int32'), y_preds, average='samples', zero_division=1)
        return self._score

    @property
    def preds(self):
        return np.array(self.y_pred)

    @property
    def logits(self):
        return np.array(self.y_logits)


class InferLabels(object):  # F1 score as competition metric
    def __init__(self, threshold=0.5):
        self.y_preds, self.y_logits, self.threshold = [], [], threshold

    def update(self, y_logits):
        self.y_logits.extend(y_logits.detach().cpu().numpy().tolist())
        self.y_preds.extend(torch.sigmoid(y_logits).detach().cpu().numpy().tolist())

    @property
    def preds(self):
        return np.array(self.y_preds)

    @property
    def logits(self):
        return np.array(self.y_logits)


def train_epoch(model, loader, criterion, optimizer, scheduler, grad_scaler, config):
    losses = AverageMeter()
    batch_size = torch.tensor(config.tra_loader['batch_size'], device=DEVICE, dtype=torch.float)
    model.train()

    for sample in loader:
        inputs = sample['waveform'].to(DEVICE)
        targets = sample['targets'].to(DEVICE)
        # sec_labels_mask = sample['sec_labels_mask'].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            preds, y_batch = model(inputs, targets)
            loss = criterion(preds, y_batch)
        """
        loss.backward()
        optimizer.step()
        scheduler.step()
        """
        grad_scaler.scale(loss).backward()  # Scales loss. Calls backward() on scaled loss to create scaled gradients
        grad_scaler.step(optimizer)  # Call optimizer.step() if unscaled gradients have no INFs or NaNs, otherwise skip
        # scale = grad_scaler.get_scale()
        grad_scaler.update()  # Updates the scale for next iteration
        # skip_lr_sched = (scale != grad_scaler.get_scale())
        # if not skip_lr_sched:
        scheduler.step()

        losses.update(loss, batch_size)

    return losses.avg.item()


def valid_epoch(model, loader, config):
    losses = AverageMeter()
    batch_size = torch.tensor(config.val_loader['batch_size'], device=DEVICE, dtype=torch.float)

    metric = MetricMeter(threshold=config.f1_threshold)
    model.eval()
    ogg_file_list = []

    with torch.no_grad():
        for sample in loader:
            inputs = sample['waveform'].to(DEVICE)
            targets = sample['targets'].to(DEVICE)

            preds, _ = model(inputs, targets)
            loss = criterions.bce_loss_with_logits(preds, targets)
            losses.update(loss, batch_size)

            metric.update(targets, preds)
            ogg_file_list.extend(sample['filename'])

    return losses.avg.item(), metric.logits, np.array(ogg_file_list)


def valid_on_tra_soundscape(model, loader, config):
    model.eval()
    row_ids = []
    metric = MetricMeter(threshold=config.f1_threshold)

    with torch.no_grad():
        for sample in loader:
            inputs = sample['waveform'].to(DEVICE)
            targets = sample['targets'].to(DEVICE)

            preds, _ = model(inputs, targets)
            metric.update(targets, preds)

            row_ids.extend(sample['row_ids'])

    return metric.score, metric.logits, np.array(row_ids)


def infer_hard_labels(model, loader, config):
    model.eval()
    row_ids = []
    infer_buffer = InferLabels(threshold=config['infer_hard_threshold'])

    with torch.no_grad():
        for sample in loader:
            inputs = sample['waveform'].to(DEVICE)

            pred_logits, _ = model(inputs)
            infer_buffer.update(pred_logits)
            row_ids.extend(sample['row_ids'])

    return infer_buffer.preds, np.array(row_ids)


def train_1_fold(_fold: int, config: params.Params, base_model: str, tra_val_df: pd.DataFrame,
                 test_df: pd.DataFrame, logger):
    tra_idx, val_idx = tra_val_df[tra_val_df['kfold'] != _fold].index, tra_val_df[tra_val_df['kfold'] == _fold].index
    train_df = tra_val_df[tra_val_df['kfold'] != _fold].reset_index(drop=True)
    valid_df = tra_val_df[tra_val_df['kfold'] == _fold].reset_index(drop=True)

    tra_loader = datasets.get_loader('train', train_df, config)
    val_loader = datasets.get_loader('valid', valid_df, config) if len(valid_df) > 0 else None
    test_loader = datasets.get_loader('test', test_df, config)

    bird_model = models.BirdCLEFModel(config=config, base_model=base_model, pretrain=True)
    bird_model = bird_model.to(DEVICE)

    # pos_weights, class_weights = utils.mp_cal_pos_weights(train_df)
    # pos_weights = torch.as_tensor(np.ones(config_g.NUM_TARGETS, dtype=np.float32) * 2, dtype=torch.float, device=DEVICE)
    # criterion = BCEWithLogitsLoss(pos_weight=pos_weights)
    # criterion = criterions.BirdBCEwLogitsLoss(class_weights=class_weights)
    criterion = BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(bird_model.parameters(), lr=config.lr)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=round(len(train_loader)/2), T_mult=2)
    # print(f"num of batches is {len(tra_loader)}")
    num_train_steps = int(len(tra_loader) * config.epochs)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps)
    grad_scaler = GradScaler()  # Creates a GradScaler once at the beginning of training

    # Training with EPOCHS, and save best LWLRAP model
    oof = np.zeros((len(tra_val_df), config_g.NUM_TARGETS))

    best_score, best_epoch, no_improve_epochs, epoch = np.inf, 0, 0, 0
    with utils.timer(f"{base_model}: Training on fold {_fold}", logger):
        for epoch in range(config.epochs):
            with utils.timer(f"Training epoch {epoch}", logger, epoch):
                train_loss = train_epoch(bird_model, tra_loader, criterion, optimizer, scheduler,
                                         grad_scaler, config)
                utils.logging_msg(f"Fold{_fold}, Epoch {epoch}, lr {optimizer.param_groups[0]['lr']:.6f}, "
                                  f"Training loss: {train_loss:0.4f}", logger)

            """
            if val_loader:
                with utils.timer(f"Validating epoch {epoch}", logger, epoch):
                    val_loss, val_logits, val_filename = valid_epoch(bird_model, val_loader, config)
                    assert(np.array_equal(df.loc[val_idx, 'filename'].to_numpy(), val_filename))
                    utils.logging_msg(f"Val loss: {val_loss:0.4f}", logger)

                oof[val_idx] = val_logits
                if val_loss < best_score:
                    utils.logging_msg(f">>> Model`s val score improved From {best_score:0.4f} to {val_loss:0.4f}", logger)
                    # oof[val_idx] = val_preds
                    # torch.save(bootstrap_model_a.state_dict(), f'bootstrap_model_a_fold{_fold}_best.bin')
                    best_score = val_loss
                    best_epoch = epoch
                    no_improve_epochs = 0
                elif config['training']['early_stop'] > 0:
                    no_improve_epochs += 1
                    if no_improve_epochs >= config['training']['early_stop']:
                        utils.logging_msg(f"Early stopped at epoch {epoch}", logger)
                        break

                utils.logging_msg(f"Training fold{_fold} with {epoch + 1} epochs,"
                                  f"best_score: {best_score:0.4f} from epoch {best_epoch}", logger)
            """

    torch.save(bird_model.state_dict(), f"{config_g.DIR_OUTPUT}/BirdCLEF_{base_model}_fold{_fold}.bin")

    # Load the best model
    # bootstrap_model_a.load_state_dict(torch.load(f'best_model_fold{_fold}.bin', map_location=DEVICE))
    # bootstrap_model_a = bootstrap_model_a.to(DEVICE)

    # Predict on test dataset
    with utils.timer(f"{base_model} fold {_fold}: validating on train_soundscape:", logger):
        score, test_logits, row_ids = valid_on_tra_soundscape(bird_model, test_loader, config)
        assert(np.array_equal(test_df['row_id'].to_numpy(), row_ids))
        utils.logging_msg(f"Model bootstrap_model_a_fold{_fold}.bin validating on train soundscape - "
                          f"f1 score: {score:.4f}", logger)

    return oof, test_logits


def train_k_folds(config: params.Params, base_model: str, tra_val_df: pd.DataFrame, test_df: pd.DataFrame, logger):
    oof = np.zeros((len(tra_val_df), config_g.NUM_TARGETS))
    predicts = np.zeros((len(test_df), config_g.NUM_TARGETS))

    # train_folds = 1  # For debug, only train 1 fold
    train_folds = config.nfolds
    for _fold in range(train_folds):
        oof_, pred_ = train_1_fold(_fold, config, base_model, tra_val_df, test_df, logger)
        predicts += pred_ / train_folds  # Blending predictions on nfolds
        oof += oof_

    return oof, predicts

