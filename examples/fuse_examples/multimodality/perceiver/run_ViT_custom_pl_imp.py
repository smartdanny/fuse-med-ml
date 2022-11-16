"""
(C) Copyright 2021 IBM Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Created on June 30, 2021

==========================

MNIST classfier implementation that demonstrate end to end training, inference and evaluation using FuseMedML
This example shows how to directly train a model using custom (your own) pytorch lightning module implementation
"""
import copy
import logging
import os
from typing import Any, List, OrderedDict, Sequence, Tuple
from fuse.utils.ndict import NDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from fuse.eval.evaluator import EvaluatorDefault
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.data.utils.collates import CollateDefault

from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
import fuse.dl.lightning.pl_funcs as fuse_pl

from fuse.utils.utils_debug import FuseDebug
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe
import fuse.utils.gpu as GPU

from fuseimg.datasets.mnist import MNIST, SHATZ_MNIST

import numpy as np
from typing import Sequence
from fuse.dl.models.backbones.backbone_transformer import Transformer
from fuse_examples.imaging.classification.mnist import lenet
from fuse.dl.models.backbones.backbone_vit import ViT

###########################################################################################################
# Fuse
###########################################################################################################

##########################################
# Lightning Module
##########################################
class ProjectPatchesTokenizer(nn.Module):
    """
    Projects a 1D/2D/3D images to tokens using patches
    Assumes to have one of the forms:
    (1) batch_size, channels, height
    (2) batch_size, channels, height, width
    (3) batch_size, channels, height, width, depth

    The output shape is always:
    batch_size, num_tokens, token_dim
    """

    def __init__(self, *, image_shape: Sequence[int], patch_shape: Sequence[int], channels: int, token_dim: int):
        super().__init__()
        assert len(image_shape) == len(patch_shape), "patch and image must have identical dimensions"
        image_shape = np.array(image_shape)
        patch_shape = np.array(patch_shape)
        assert (image_shape % patch_shape == 0).all(), "Image dimensions must be divisible by the patch size."
        self.num_tokens = int(np.prod(image_shape // patch_shape))
        patch_shape = tuple(patch_shape)
        self.image_dim = len(image_shape)
        if self.image_dim == 1:
            self.proj_layer = nn.Conv1d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        elif self.image_dim == 2:
            self.proj_layer = nn.Conv2d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        elif self.image_dim == 3:
            self.proj_layer = nn.Conv3d(
                in_channels=channels, out_channels=token_dim, kernel_size=patch_shape, stride=patch_shape
            )
        else:
            raise NotImplementedError("only supports 1D/2D/3D images")

    def forward(self, x):
        assert len(x.shape) == self.image_dim + 2, "input should be [batch, channels] + image_shape"
        x = self.proj_layer(x)
        x = x.flatten(start_dim=2, end_dim=-1)  # x.shape == (batch_size, token_dim, num_tokens)
        x = x.transpose(1, 2)  # x.shape == (batch_size, num_tokens, token_dim)
        return x

class MyMmvit(nn.Module):
    """
    Projects a 1D/2D/3D image into tokens, and then runs it through a transformer
    Then puts a linear (fc) head onto backbone from fuse.dl.models.backbones.backbone_vit.ViT
    """

    def __init__(self, token_dim: int, projection_kwargs: dict, transformer_kwargs: dict):
        """
        :param token_dim: the dimension of each token in the transformer
        :param projection_kwargs: positional arguments for the ProjectPatchesTokenizer class
        :param transformer_kwargs: positional arguments for the Transformer class
        """
        super().__init__()
        self.projection_layer = ProjectPatchesTokenizer(token_dim=token_dim, **projection_kwargs)
        num_tokens = self.projection_layer.num_tokens
        self.transformer = Transformer(num_tokens=num_tokens, token_dim=token_dim, **transformer_kwargs)
        self._head = nn.Linear(token_dim, 10)

    def forward(self, x: torch.Tensor, mod2: torch.Tensor, pool: str = "none"):
        """
        :param pool: returns all tokens (pool='none'), only cls token (pool='cls') or the average token (pool='mean')
        """
        assert pool in ["none", "cls", "mean"]
        x = self.projection_layer(x)
        x = torch.cat((x, mod2.unsqueeze(1)), 1)
        x = self.transformer(x)
        if pool == "cls":
            x = x[:, 0]
        if pool == "mean":
            x = x.mean(dim=1)
        x = self._head(x[:, 0])
        return x


class LightningModuleMnist(pl.LightningModule):
    """
    Implementation of pl.LightningModule
    Demonstrates how to use FuseMedML with your own PyTorch lightining implementaiton.
    """

    def __init__(self, model_dir: str, opt_lr: float, opt_weight_decay: float, **kwargs):
        """
        :param model_dir: location for checkpoints and logs
        :param opt_lr: learning rate for Adam optimizer
        :param opt_weight_decay: weight decay for Adam optimizer
        """
        super().__init__(**kwargs)
        self.save_hyperparameters(ignore=["model_dir"])

        # store arguments
        self._model_dir = model_dir
        self._opt_lr = opt_lr
        self._opt_weight_decay = opt_weight_decay

        # init state
        self._prediction_keys = None

        # model
        token_dim = 64
        # input is a 3d image with shape [128,128,128] and 1 channel
        # projected using 3d patches of size [16,16,16]
        projection_kwargs = dict(image_shape=[28, 28], patch_shape=[14, 14], channels=1)
        # the transformer specification
        transformer_kwargs = dict(depth=4, heads=5, mlp_dim=256, dim_head=64, dropout=0.0, emb_dropout=0.0)
        torch_model = MyMmvit(token_dim=token_dim, projection_kwargs=projection_kwargs, transformer_kwargs=transformer_kwargs)

        # wrap basic torch model to automatically read inputs from batch_dict and save its outputs to batch_dict
        self._model = ModelWrapSeqToDict(
            model=torch_model,
            model_inputs=["data.image", "data.mod2"],
            post_forward_processing_function=perform_softmax,
            model_outputs=["model.logits.classification", "model.output.classification"],
        )

        # losses
        self._losses = {
            "cls_loss": LossDefault(
                pred="model.logits.classification", target="data.label", callable=F.cross_entropy, weight=1.0
            ),
        }

        # metrics
        self._train_metrics = OrderedDict(
            [
                ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
                ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            ]
        )

        self._validation_metrics = copy.deepcopy(self._train_metrics)  # use the same metrics in validation as well

    ## forward
    def forward(self, batch_dict: NDict) -> NDict:
        x = self._model(batch_dict)
        return x

    ## Step
    def training_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss and save losses values in batch_dict["losses"]
        total_loss = fuse_pl.step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        fuse_pl.step_metrics(self._train_metrics, batch_dict)

        # return the total_loss, the losses and drop everything else
        return {"loss": total_loss, "losses": batch_dict["losses"]}

    def validation_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # given the batch_dict and FuseMedML style losses - compute the losses, return the total loss (ignored) and save losses values in batch_dict["losses"]
        _ = fuse_pl.step_losses(self._losses, batch_dict)
        # given the batch_dict and FuseMedML style losses - collect the required values to compute the metrics on epoch_end
        fuse_pl.step_metrics(self._validation_metrics, batch_dict)

        # return just the losses and drop everything else
        return {"losses": batch_dict["losses"]}

    def predict_step(self, batch_dict: NDict, batch_idx: int) -> dict:
        if self._prediction_keys is None:
            raise Exception(
                "Error: predict_step expectes list of prediction keys to extract from batch_dict. Please specify it using set_predictions_keys() method "
            )
        # run forward function and store the outputs in batch_dict["model"]
        batch_dict = self.forward(batch_dict)
        # extract the requried keys - defined in self.set_predictions_keys()
        return fuse_pl.step_extract_predictions(self._prediction_keys, batch_dict)

    ## Epoch end
    def training_epoch_end(self, step_outputs) -> None:
        # calc average epoch loss and log it
        fuse_pl.epoch_end_compute_and_log_losses(self, "train", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        fuse_pl.epoch_end_compute_and_log_metrics(self, "train", self._train_metrics)

    def validation_epoch_end(self, step_outputs) -> None:
        # calc average epoch loss and log it
        fuse_pl.epoch_end_compute_and_log_losses(self, "validation", [e["losses"] for e in step_outputs])
        # evaluate  and log it
        fuse_pl.epoch_end_compute_and_log_metrics(self, "validation", self._validation_metrics)

    def configure_callbacks(self) -> Sequence[pl.Callback]:
        """Create callbacks to monitor the metrics and print epoch summary"""
        best_epoch_source = dict(
            monitor="validation.metrics.accuracy",
            mode="max",
        )
        return fuse_pl.model_checkpoint_callbacks(self._model_dir, best_epoch_source)

    def configure_optimizers(self) -> Any:
        """See pl.LightningModule.configure_optimizers return value for all options"""
        # create optimizer
        optimizer = optim.Adam(self._model.parameters(), lr=self._opt_lr, weight_decay=self._opt_weight_decay)

        # create learning scheduler
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
        return dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    def set_predictions_keys(self, keys: List[str]) -> None:
        """Define which keys to extract from batch_dict  on prediction mode"""
        self._prediction_keys = keys


##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
ROOT = "_examples/shatz"  # TODO: fill path here
model_dir = os.path.join(ROOT, "model_dir")
PATHS = {
    "model_dir": model_dir,
    "cache_dir": os.path.join(ROOT, "cache_dir"),
    "inference_dir": os.path.join(model_dir, "infer_dir"),
    "eval_dir": os.path.join(model_dir, "eval_dir"),
}

NUM_GPUS = 1
##########################################
# Train Common Params
##########################################
TRAIN_COMMON_PARAMS = {}

# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = 100
TRAIN_COMMON_PARAMS["data.train_num_workers"] = 0
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = 8

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = 8
TRAIN_COMMON_PARAMS["trainer.num_devices"] = NUM_GPUS
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
# use "dp" strategy temp when working with multiple GPUS - workaround for pytorch lightning issue: https://github.com/Lightning-AI/lightning/issues/11807
TRAIN_COMMON_PARAMS["trainer.strategy"] = "dp" if TRAIN_COMMON_PARAMS["trainer.num_devices"] > 1 else None
TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None  # path to the checkpoint you wish continue the training from

# ===============
# PL Module
# ===============
TRAIN_COMMON_PARAMS["pl_module.opt_lr"] = 1e-4
TRAIN_COMMON_PARAMS["pl_module.opt_weight_decay"] = 0.001


def perform_softmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds


#################################
# Train Template
#################################
def run_train(paths: dict, train_params: dict):
    # ==============================================================================
    # Logger(s)
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)
    lightning_csv_logger = CSVLogger(save_dir=paths["model_dir"]+"/lightning_logs/", name="lightning_csv_logs")
    lightning_tb_logger = TensorBoardLogger(save_dir=paths["model_dir"]+"/lightning_logs/", name="lightning_tb_logs_ViT")
    print("Fuse Train")

    # ==============================================================================
    # Data
    # ==============================================================================
    # Train Data
    print("Data - trainset:")
    train_dataset = SHATZ_MNIST.dataset(paths["cache_dir"], train=True)
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.label",
        num_balanced_classes=10,
        batch_size=train_params["data.batch_size"],
        balanced_class_weights=None,
    )
    print("- Create sampler: Done")

    # Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_params["data.train_num_workers"],
    )
    print("Data - trainset: Done")

    ## Validation data
    print("Data - validation set:")
    # wrapping torch dataset
    validation_dataset = SHATZ_MNIST.dataset(paths["cache_dir"], train=False)

    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        batch_size=train_params["data.batch_size"],
        collate_fn=CollateDefault(),
        num_workers=train_params["data.validation_num_workers"],
    )
    print("Data - validation set: Done")

    # ==============================================================================
    # Train
    # ==============================================================================
    # create instance of PL module
    pl_module = LightningModuleMnist(
        model_dir=paths["model_dir"],
        opt_lr=train_params["pl_module.opt_lr"],
        opt_weight_decay=train_params["pl_module.opt_weight_decay"],
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_params["trainer.num_epochs"],
        accelerator=train_params["trainer.accelerator"],
        strategy=train_params["trainer.strategy"],
        devices=train_params["trainer.num_devices"],
        auto_select_gpus=True,
        logger=[lightning_csv_logger, lightning_tb_logger],
    )

    # train
    pl_trainer.fit(pl_module, train_dataloader, validation_dataloader, ckpt_path=train_params["trainer.ckpt_path"])
    print("Train: Done")


######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"
INFER_COMMON_PARAMS["trainer.num_devices"] = 1  # infer should use single device
INFER_COMMON_PARAMS["trainer.accelerator"] = "gpu"
INFER_COMMON_PARAMS["trainer.strategy"] = None

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict):
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])
    #### Logger
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    ## Data
    # Create dataset
    validation_dataset = MNIST.dataset(paths["cache_dir"], train=False)
    # dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset, collate_fn=CollateDefault(), batch_size=2, num_workers=2
    )

    # load pytorch lightning module
    pl_module = LightningModuleMnist.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], map_location="cpu", strict=True
    )
    # set the prediction keys to extract (the ones used be the evaluation function).
    pl_module.set_predictions_keys(
        ["model.output.classification", "data.label"]
    )  # which keys to extract and dump into file

    print("Model: Done")
    # create a trainer instance
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        strategy=infer_common_params["trainer.strategy"],
        auto_select_gpus=True,
        logger=None,
    )
    predictions = pl_trainer.predict(pl_module, validation_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = fuse_pl.convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)


######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


######################################
# Eval Template
######################################
def run_eval(paths: dict, eval_common_params: dict):
    create_dir(paths["eval_dir"])
    infer_file = os.path.join(paths["inference_dir"], eval_common_params["infer_filename"])
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)

    print("Fuse Eval")

    # metrics
    class_names = [str(i) for i in range(10)]

    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.label")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.classification",
                    target="data.label",
                    class_names=class_names,
                    output_filename=os.path.join(paths["eval_dir"], "roc_curve.png"),
                ),
            ),
            ("auc", MetricAUCROC(pred="model.output.classification", target="data.label", class_names=class_names)),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(ids=None, data=infer_file, metrics=metrics, output_dir=paths["eval_dir"])

    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ["train"]  # Options: 'train', 'infer', 'eval'
    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
