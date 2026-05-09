import collections
import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm

from assignment_2_code.wandb_logger import WandBLogger
from assignment_2_code.dataset.oxfordpets import OxfordPetsCustom


class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class of all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def _val_epoch(self) -> Tuple[float, float]:
        pass

    @abstractmethod
    def _train_epoch(self) -> Tuple[float, float]:
        pass


class ImgSemSegTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image semantic segmentation.
    """

    def __init__(self,
                 model,
                 optimizer,
                 loss_fn,
                 lr_scheduler,
                 train_metric,
                 val_metric,
                 train_data,
                 val_data,
                 device,
                 num_epochs: int,
                 training_save_dir: Path,
                 batch_size: int = 4,
                 val_frequency: int = 5):

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.val_frequency = val_frequency
        self.checkpoint_dir = training_save_dir

        # OxfordPets labels are 1-based, so we subtract 1 to make them 0-based
        self.subtract_one = isinstance(train_data, OxfordPetsCustom)

        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=2
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=1
        )

        self.num_train_data = len(train_data)
        self.num_val_data = len(val_data)

        self.wandb_logger = WandBLogger(
            enabled=False, model=model, run_name=model.net._get_name()
        )

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Training logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch.
        """
        self.model.train()
        self.train_metric.reset()
        epoch_loss = 0.0

        for batch in tqdm(self.train_loader, desc=f"Train epoch {epoch_idx}"):
            inputs, labels = batch
            labels = labels.squeeze(1) - int(self.subtract_one)
            batch_size = inputs.shape[0]

            self.optimizer.zero_grad()

            outputs = self.model(inputs.to(self.device))
            if isinstance(outputs, collections.OrderedDict):
                outputs = outputs['out']

            loss = self.loss_fn(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * batch_size
            self.train_metric.update(outputs.detach().cpu(), labels.detach().cpu())

        self.lr_scheduler.step()
        epoch_loss /= self.num_train_data
        epoch_mIoU = self.train_metric.mIoU()

        print(f"\nEpoch {epoch_idx} - Train Loss: {epoch_loss:.4f} | {self.train_metric}")

        return epoch_loss, epoch_mIoU

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float]:
        """
        Validation logic for one epoch.
        Prints current metrics at end of epoch.
        Returns loss, mean IoU for this epoch on the validation dataset.
        """
        self.model.eval()
        self.val_metric.reset()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Val epoch {epoch_idx}"):
                inputs, labels = batch
                labels = labels.squeeze(1) - int(self.subtract_one)
                batch_size = inputs.shape[0]

                outputs = self.model(inputs.to(self.device))
                if isinstance(outputs, collections.OrderedDict):
                    outputs = outputs['out']

                loss = self.loss_fn(outputs, labels.to(self.device))
                epoch_loss += loss.item() * batch_size
                self.val_metric.update(outputs.cpu(), labels.cpu())

        epoch_loss /= self.num_val_data
        epoch_mIoU = self.val_metric.mIoU()

        print(f"Epoch {epoch_idx} - Val Loss: {epoch_loss:.4f} | {self.val_metric}")

        return epoch_loss, epoch_mIoU

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs.
        Saves the model when validation mIoU improves, and always at the end.
        """
        best_mIoU = 0.0

        for epoch_idx in range(self.num_epochs):
            train_loss, train_mIoU = self._train_epoch(epoch_idx)

            wandb_log = {
                'epoch': epoch_idx,
                'train/loss': train_loss,
                'train/mIoU': train_mIoU,
            }

            if epoch_idx % self.val_frequency == 0:
                val_loss, val_mIoU = self._val_epoch(epoch_idx)
                wandb_log['val/loss'] = val_loss
                wandb_log['val/mIoU'] = val_mIoU

                if val_mIoU > best_mIoU:
                    best_mIoU = val_mIoU
                    print(f"New best mIoU: {best_mIoU:.4f} — saving model.")
                    self.model.save(Path(self.checkpoint_dir), suffix="best")

            if epoch_idx == self.num_epochs - 1:
                self.model.save(Path(self.checkpoint_dir), suffix="last")

            self.wandb_logger.log(wandb_log)

    def dispose(self) -> None:
        self.wandb_logger.finish()