from collections import OrderedDict
import pathlib
from fuse.utils.utils_logger import fuse_logger_start
import os
from typing import Any, List, OrderedDict, Sequence, Tuple
import yaml

import pandas as pd
from fuse.dl.models import ModelMultiHead
from fuse.dl.models.backbones.backbone_resnet_3d import BackboneResnet3D
from fuse.dl.models.heads.heads_3D import Head3D
from fuseimg.datasets.knight import KNIGHT
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault

from fuse.eval.metrics.classification.metrics_classification_common import MetricAUCROC, MetricAccuracy, MetricConfusion
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
import torch.optim as optim
import fuse.utils.gpu as GPU
from fuse.utils.rand.seed import Seed
import logging
import time
import copy
from fuse.dl.losses.loss_default import LossDefault
from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.models.model_wrapper import ModelWrapSeqToDict
import pytorch_lightning as pl

import torch
import numpy as np
from typing import Sequence
from fuse.dl.models.backbones.backbone_transformer import Transformer
from fuse_examples.imaging.classification.mnist import lenet
from fuse.dl.models.backbones.backbone_vit import ViT

## Parameters:
##############################################################################
# Data sources to use in model. Set {'imaging': True, 'clinical': False} for imaging only setting,
# and vice versa, or set both to True to use both.
# allocate gpus
# uncomment if you want to use specific gpus instead of automatically looking for free ones

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
        mod2 = mod2.unsqueeze(1)
        mod2_zeros = torch.zeros((x.shape[0], 1, 64))
        mod2_zeros[:, :, :11] = mod2
        mod2 = mod2_zeros.cuda()
        x = torch.cat((x, mod2), 1)
        x = self.transformer(x)
        if pool == "cls":
            x = x[:, 0]
        if pool == "mean":
            x = x.mean(dim=1)
        x = self._head(x[:, 0])
        return x

def perform_softmax(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cls_preds = F.softmax(logits, dim=1)
    return logits, cls_preds

def make_model(use_data: dict, num_classes: int, imaging_dropout: float, fused_dropout: float):
    if use_data["imaging"]:
        backbone = BackboneResnet3D(in_channels=1)
        conv_inputs = [("model.backbone_features", 512)]
    else:
        backbone = nn.Identity()
        conv_inputs = None
    if use_data["clinical"]:
        append_features = [("data.input.clinical.all", 11)]
    else:
        append_features = None
    
    # model
    token_dim = 64
    # input is a 3d image with shape [128,128,128] and 1 channel
    # projected using 3d patches of size [16,16,16]
    projection_kwargs = dict(image_shape=[70, 256, 256], patch_shape=[7, 16, 16], channels=1)
    # the transformer specification
    transformer_kwargs = dict(depth=4, heads=5, mlp_dim=256, dim_head=64, dropout=0.0, emb_dropout=0.0)
    torch_model = MyMmvit(token_dim=token_dim, projection_kwargs=projection_kwargs, transformer_kwargs=transformer_kwargs)

    model = ModelWrapSeqToDict(
        model=torch_model,
        model_inputs=["data.input.img", "data.input.clinical.all"],
        post_forward_processing_function=perform_softmax,
        model_outputs=["model.logits.head_0", "model.output.head_0"],
    )
    # model = ModelMultiHead(
    #     conv_inputs=(("data.input.img", 1),),
    #     backbone=backbone,
    #     heads=[
    #         Head3D(
    #             head_name="head_0",
    #             mode="classification",
    #             conv_inputs=conv_inputs,
    #             dropout_rate=imaging_dropout,
    #             num_outputs=num_classes,
    #             append_features=append_features,
    #             append_layers_description=(256, 128),
    #             fused_dropout_rate=fused_dropout,
    #         ),
    #     ],
    # )
    return model


def main(cfg_path):
    # read config params
    cfg = yaml.safe_load(open(cfg_path))
    task_num = cfg["task_num"]
    num_classes = cfg[task_num]["num_classes"]
    target_name = cfg[task_num]["target_name"]

    if cfg["num_gpus"] == 0:
        use_gpu = False
    else:
        use_gpu = True

    # read train/val splits file. for convenience, we use the one
    # auto-generated by the nnU-Net framework for the KiTS21 data
    dir_path = pathlib.Path(__file__).parent.resolve()
    splits = pd.read_pickle(os.path.join(dir_path, "splits_final.pkl"))
    # For this example, we use split 0 out of the 5 available cross validation splits
    split = splits[0]
    if cfg["testing"]:
        split["train"] = split["train"][:5]
        split["val"] = split["val"][:5]
    # read environment variables for data, cache and results locations
    data_path = os.environ["KNIGHT_DATA"]
    cache_path = './_examples/shatz_knight/'
    results_path = os.path.join(cache_path, "results")
    # cache_path = os.path.join(os.environ["KNIGHT_CACHE"], str(cfg["experiment_num"]))
    # results_path = os.environ["KNIGHT_RESULTS"]

    ## Basic settings:
    ##############################################################################
    # create model results dir:
    # we use a time stamp in model directory name, to prevent re-writing
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(results_path, timestr)
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    # start logger
    fuse_logger_start(output_path=model_dir, console_verbose_level=logging.INFO)
    print("Done")

    # set constant seed for reproducibility.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for pytorch deterministic mode
    rand_gen = Seed.set_seed(1234, deterministic_mode=True)

    # select gpus
    if use_gpu:
        GPU.choose_and_enable_multiple_gpus(cfg["num_gpus"], force_gpus=None)

    ## Model definition
    ##############################################################################

    model = make_model(cfg["use_data"], num_classes, cfg["imaging_dropout"], cfg["fused_dropout"])

    ## FuseMedML dataset preparation
    ##############################################################################

    train_ds, valid_ds = KNIGHT.dataset(
        data_path=data_path,
        cache_dir=cache_path,
        split=split,
        reset_cache=False,
        resize_to=cfg["resize_to"],
    )

    ## Create dataloader

    sampler = BatchSamplerDefault(
        dataset=train_ds,
        balanced_class_name=target_name,
        num_balanced_classes=num_classes,
        batch_size=cfg["batch_size"],
        balanced_class_weights=[1.0 / num_classes] * num_classes if task_num == "task_2" else None,
        mode="approx",
    )

    train_dl = DataLoader(
        dataset=train_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=8,
    )

    valid_dl = DataLoader(
        dataset=valid_ds,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=cfg["batch_size"],
        num_workers=8,
        collate_fn=CollateDefault(),
        generator=rand_gen,
    )

    # train_iter = next(iter(train_dl))
    # clinical_train_feat = train_iter["data"]["input"]["clinical"]
    # clinical_val_feat = train_iter["data"]["input"]["clinical"]
    # import pdb; pdb.set_trace()

    # Loss definition:
    ##############################################################################
    losses = {
        "cls_loss": LossDefault(pred="model.logits.head_0", target=target_name, callable=F.cross_entropy, weight=1.0),
    }

    # Metrics definition:
    ##############################################################################
    train_metrics = OrderedDict(
        [
            ("op", MetricApplyThresholds(pred="model.output.head_0")),  # will apply argmax
            ("auc", MetricAUCROC(pred="model.output.head_0", target=target_name)),
            ("accuracy", MetricAccuracy(pred="results:metrics.op.cls_pred", target=target_name)),
            (
                "sensitivity",
                MetricConfusion(pred="results:metrics.op.cls_pred", target=target_name, metrics=("sensitivity",)),
            ),
        ]
    )
    val_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(
        monitor=cfg[task_num]["target_metric"],  # can be any key from losses or metrics dictionaries
        mode="max",  # can be either min/max
    )

    # Optimizer definition:
    ##############################################################################
    optimizer = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=0.001)

    # Scheduler definition:
    ##############################################################################
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    ## Training
    ##############################################################################

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=model_dir,
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=val_metrics,
        # best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )
    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=model_dir,
        max_epochs=cfg["num_epochs"],
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg["num_gpus"] if use_gpu else None,
        strategy=None,
        auto_select_gpus=True if use_gpu else False,
        num_sanity_val_steps=-1,
    )

    # train
    pl_trainer.fit(pl_module, train_dl, valid_dl, ckpt_path=None)


if __name__ == "__main__":
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    main(config_path)
