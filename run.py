import argparse
import os
import random

import dill as pickle
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateLogger
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import AdvancedProfiler, SimpleProfiler
from mmcv import Config
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import translation

from model import Transformer


class LightningTransformer(pl.LightningModule):

    PAD_WORD = '<blank>'
    UNK_WORD = '<unk>'
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'

    def __init__(self, cfg):
        super(LightningTransformer, self).__init__()

        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.train_cfg = cfg.train_cfg
        self.lr_cfg = cfg.lr_cfg
        self._update_model_cfg_by_data()

        self.transformer = Transformer(**self.model_cfg)

    def forward(self, src_seq, trg_seq):
        out = self.transformer(src_seq, trg_seq)
        return out

    def training_step(self, batch, batch_idx):
        src_seq, trg_seq = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        trg_seq, gold = trg_seq[:, :-1], trg_seq[:, 1:].contiguous().view(-1)

        pred = self.transformer(src_seq, trg_seq)

        loss = self._cal_loss(pred, gold, self.model_cfg.trg_pad_idx, self.train_cfg.smoothing)
        n_correct, n_word = self._cal_performance(pred, gold, self.model_cfg.trg_pad_idx)

        return {"loss": loss, "n_correct": n_correct, "n_word": n_word}

    def training_epoch_end(self, outputs):
        total_correct = sum([output["n_correct"] for output in outputs])
        total_word = sum([output["n_word"] for output in outputs])
        total_loss = torch.stack([output["loss"] for output in outputs]).sum()

        logs = {"train_loss_per_word": total_loss / total_word, "train_acc": total_correct / total_word}
        return {"train_loss": total_loss / total_word, "log": logs}

    def validation_step(self, batch, batch_idx):
        src_seq, trg_seq = batch.src.transpose(0, 1), batch.trg.transpose(0, 1)
        trg_seq, gold = trg_seq[:, :-1], trg_seq[:, 1:].contiguous().view(-1)

        pred = self.transformer(src_seq, trg_seq)
        loss = self._cal_loss(pred, gold, self.model_cfg.trg_pad_idx, self.train_cfg.smoothing)
        n_correct, n_word = self._cal_performance(pred, gold, self.model_cfg.trg_pad_idx)

        return {"val_loss": loss, "n_correct": n_correct, "n_word": n_word}

    def validation_epoch_end(self, outputs):
        total_correct = sum([output["n_correct"] for output in outputs])
        total_word = sum([output["n_word"] for output in outputs])
        total_loss = torch.stack([output["val_loss"] for output in outputs]).sum()

        logs = {"val_loss_per_word": total_loss / total_word, "val_acc": total_correct / total_word}
        return {"val_loss_per_word": total_loss / total_word, "log": logs}

    @staticmethod
    def _cal_loss(pred, gold, trg_pad_idx, smoothing=False):
        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.1
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            non_pad_mask = gold.ne(trg_pad_idx)
            loss = -(one_hot * log_prb).sum(dim=1)
            loss = loss.masked_select(non_pad_mask).sum()  # average later
        else:
            loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction='sum')

        return loss

    @staticmethod
    def _cal_performance(pred, gold, trg_pad_idx):
        pred_idx = pred.detach().max(1)[1]
        gold = gold.contiguous().view(-1)
        non_pad_mask = gold.ne(trg_pad_idx)
        n_correct = pred_idx.eq(gold).masked_select(non_pad_mask).sum().item()
        n_word = non_pad_mask.sum().item()

        return n_correct, n_word

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98), eps=1e-9)
        return optimizer

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=False,
                       using_native_amp=False, using_lbfgs=False):

        d_model, steps, warmup_steps = self.model_cfg.d_model, self.global_step + 1, self.lr_cfg.warmup_steps
        lr_scale = (d_model ** (- 0.5)) * min(steps ** (- 0.5), steps * warmup_steps ** (- 1.5))

        for pg in optimizer.param_groups:
            pg['lr'] = lr_scale * self.lr_cfg.init_lr

        # update params
        optimizer.step()
        optimizer.zero_grad()

    def _update_model_cfg_by_data(self):
        data = pickle.load(open(self.data_cfg.data_path, "rb"))

        self.model_cfg.update(max_len=data['settings'].max_len,
                              src_pad_idx=data['vocab']['src'].vocab.stoi[self.PAD_WORD],
                              trg_pad_idx=data['vocab']['trg'].vocab.stoi[self.PAD_WORD],
                              n_src_vocab=len(data['vocab']['src'].vocab),
                              n_trg_vocab=len(data['vocab']['trg'].vocab))

    def prepare_data(self):
        batch_size = self.data_cfg.batch_size
        data = pickle.load(open(self.data_cfg.data_path, "rb"))

        if self.model_cfg.emb_src_trg_weight_sharing:
            assert data['vocab']['src'].vocab.stoi == data['vocab']['trg'].vocab.stoi, \
                'To sharing word embedding the src/trg word2idx table shall be the same.'

        fields = {'src': data['vocab']['src'], 'trg': data['vocab']['trg']}
        self.train_dataset = Dataset(examples=data['train'], fields=fields)
        self.val_dataset = Dataset(examples=data['valid'], fields=fields)

    def train_dataloader(self):
        return BucketIterator(self.train_dataset, batch_size=self.data_cfg.batch_size, train=True)

    def val_dataloader(self):
        return BucketIterator(self.val_dataset, batch_size=self.data_cfg.batch_size)


def parse_args():
    parser = argparse.ArgumentParser("Train model.")
    parser.add_argument("config", help="Train config file path.")

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_seed(cfg.random_seed)

    model = LightningTransformer(cfg)

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(cfg.checkpoint_path, cfg.name, cfg.version,
                              "{}_{}_{{epoch}}_{{val_loss_per_word}}".format(cfg.name, cfg.version)),
        save_last=True,
        save_top_k=8,
        verbose=True,
        monitor='val_loss_per_word',
        mode='min',
        prefix=''
    )

    lr_logger_callback = LearningRateLogger(logging_interval='step')

    logger = TensorBoardLogger(save_dir=cfg.log_path, name=cfg.name, version=cfg.version)
    logger.log_hyperparams(model.hparams)

    profiler = SimpleProfiler() if cfg.simple_profiler else AdvancedProfiler()

    trainer = pl.Trainer(
        gpus=cfg.num_gpus,
        max_epochs=cfg.max_epochs,
        logger=logger,
        profiler=profiler,
        weights_summary="top",
        callbacks=[lr_logger_callback],
        checkpoint_callback=checkpoint_callback,
        resume_from_checkpoint=cfg.resume_from_checkpoint,
        accumulate_grad_batches=cfg.batch_size_times)

    if cfg.load_from_checkpoint is not None:
        ckpt = torch.load(cfg.load_from_checkpoint, map_location=lambda storage, loc: storage)
        model.load_state_dict(ckpt['state_dict'])
    trainer.fit(model)



if __name__ == '__main__':
    main()
    