from typing import Any, List

import hydra
import torch.nn.functional
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torchmetrics import F1Score, MaxMetric, Precision, Recall


class Model(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model=None,
        scheduler=None,
        optimizer=None,
        loss=None,
        watcher=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.model = hydra.utils.instantiate(self.hparams.model)
        ignore_index: int = 0
        self.train_f1 = F1Score(ignore_index=ignore_index)
        self.val_f1 = F1Score(ignore_index=ignore_index)
        self.val_recall = Recall(ignore_index=ignore_index)
        self.val_precision = Precision(ignore_index=ignore_index)
        self.val_f1_best = MaxMetric()
        self.val_recall_best = MaxMetric()
        self.val_precision_best = MaxMetric()
        # loss function
        self.criterion = hydra.utils.instantiate(self.hparams.loss)

    def step(self, batch: Any):
        x, labels = batch
        y_hat = self.model(x, self.device)
        return y_hat, labels

    def fn_loss(self, y_hat, labels):
        return self.criterion(y_hat.view(-1, 1), labels.view(-1, 1))

    @staticmethod
    def fit_for_metrics(y_hat, labels):
        return torch.nn.functional.sigmoid(y_hat), labels.long()

    def training_step(self, batch: Any, batch_idx: int):
        y_hat, labels = self.step(batch)
        loss = self.fn_loss(y_hat, labels)
        # log train metrics
        self.train_f1(*self.fit_for_metrics(y_hat[:, :, 0], labels))
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train/f1", self.train_f1, on_step=True, on_epoch=True)
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        y_hat, labels = self.step(batch)
        # log train metrics
        y_logits = y_hat[:, :, 0]
        self.val_f1(*self.fit_for_metrics(y_logits, labels))
        self.val_recall(*self.fit_for_metrics(y_logits, labels))
        self.val_precision(*self.fit_for_metrics(y_logits, labels))
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val/recall", self.val_recall, on_step=False, on_epoch=True)
        self.log("val/precision", self.val_precision, on_step=False, on_epoch=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"val/f1": self.val_f1}

    def validation_epoch_end(self, outputs: List[Any]):
        f1 = self.val_f1.compute()
        self.val_f1_best.update(f1)
        self.log("val/f1_best", self.val_f1_best.compute(), on_epoch=True, prog_bar=True)

        recall = self.val_recall.compute()
        self.val_recall_best.update(recall)
        self.log(
            "val/recall_best",
            self.val_recall_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )

        precision = self.val_precision.compute()
        self.val_precision_best.update(precision)
        self.log(
            "val/precision_best",
            self.val_precision_best.compute(),
            on_epoch=True,
            prog_bar=False,
        )

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=self.model.parameters())
        if self.hparams.scheduler is None:
            return {
                "optimizer": optimizer,
            }
        scheduler = hydra.utils.instantiate(self.hparams.scheduler, optimizer=optimizer)
        watcher = dict(self.hparams.watcher.copy())
        watcher["scheduler"] = scheduler
        return {
            "lr_scheduler": watcher,
            "optimizer": optimizer,
        }

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass
