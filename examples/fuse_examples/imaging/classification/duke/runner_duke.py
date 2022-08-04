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

"""

import os
import copy
from typing import Any, OrderedDict
import logging

import torch
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import torch.nn.functional as F

from fuse.utils.utils_debug import FuseDebug
import fuse.utils.gpu as GPU
from fuse.utils.utils_logger import fuse_logger_start
from fuse.utils.file_io.file_io import create_dir, save_dataframe, load_pickle
from fuse.data.utils.split import dataset_balanced_division_to_folds

# import fuse_examples.imaging.classification.duke.duke_utils as duke_utils
from fuse_examples.imaging.utils.backbone_3d_multichannel import Fuse_model_3d_multichannel, ResNet
from fuse.dl.models.heads import Head1DClassifier
from fuse.dl.losses.loss_default import LossDefault

from fuse.data.utils.collates import CollateDefault
from fuse.data.utils.samplers import BatchSamplerDefault
from fuse.eval.metrics.classification.metrics_thresholding_common import MetricApplyThresholds
from fuse.eval.metrics.classification.metrics_classification_common import MetricAccuracy, MetricAUCROC, MetricROCCurve

from fuse.dl.lightning.pl_module import LightningModuleDefault
from fuse.dl.lightning.pl_funcs import convert_predictions_to_dataframe

from fuse.eval.evaluator import EvaluatorDefault
from fuseimg.datasets.duke.duke import Duke
from fuseimg.datasets.duke.duke_label_type import DukeLabelType


##########################################
# Debug modes
##########################################
mode = "default"  # Options: 'default', 'debug'. See details in FuseDebug
debug = FuseDebug(mode)

##########################################
# Output Paths
##########################################
# ROOT = duke_utils.get_duke_user_dir()
ROOT = "/tmp/_duke_sagi"
data_dir = os.environ["DUKE_DATA_PATH"]

# if mode == "debug": # super temp :)
if True:  # super temp :)
    data_split_file = "DUKE_folds_debug.pkl"

    selected_positive = [1, 2, 3, 5, 6, 10, 12, 596, 900, 901]
    selected_negative = [4, 6, 7, 8, 11, 13, 14, 120, 902, 903]
    selected_sample_ids = [f"Breast_MRI_{ii:03d}" for ii in selected_positive + selected_negative]

    model_dir = os.path.join(ROOT, "model_dir_debug")
    cache_dir = os.path.join(ROOT, "cache_dir_debug")
    force_reset_model_dir = True

    num_workers = 10
    batch_size = 2
    num_epochs = 2
    num_devices = 1

else:
    data_split_file = "DUKE_folds.pkl"

    selected_sample_ids = None

PATHS = {
    "model_dir": model_dir,
    "force_reset_model_dir": force_reset_model_dir,  # If True will reset model dir automatically - otherwise will prompt 'are you sure' message.
    "data_dir": data_dir,
    "cache_dir": cache_dir,
    "inference_dir": os.path.join(model_dir, "infer"),
    "eval_dir": os.path.join(model_dir, "eval"),
    "data_split_filename": os.path.join(ROOT, data_split_file),
}

##########################################
# Train Common Params
##########################################
num_folds = 5
heldout_fold = 4
label_type: str = DukeLabelType.STAGING_TUMOR_SIZE

train_folds = [i % num_folds for i in range(heldout_fold + 1, heldout_fold + num_folds - 1)]
validation_folds = [(heldout_fold - 1) % num_folds]


TRAIN_COMMON_PARAMS = {}
# ============
# Data
# ============
TRAIN_COMMON_PARAMS["data.batch_size"] = batch_size
TRAIN_COMMON_PARAMS["data.train_num_workers"] = num_workers
TRAIN_COMMON_PARAMS["data.validation_num_workers"] = num_workers
TRAIN_COMMON_PARAMS["data.cache_num_workers"] = num_workers
TRAIN_COMMON_PARAMS["data.sample_ids"] = selected_sample_ids
TRAIN_COMMON_PARAMS["data.num_folds"] = num_folds
TRAIN_COMMON_PARAMS["data.train_folds"] = train_folds
TRAIN_COMMON_PARAMS["data.validation_folds"] = validation_folds

# ===============
# PL Trainer
# ===============
TRAIN_COMMON_PARAMS["trainer.num_epochs"] = num_epochs
TRAIN_COMMON_PARAMS["trainer.num_devices"] = num_devices
TRAIN_COMMON_PARAMS["trainer.accelerator"] = "gpu"
TRAIN_COMMON_PARAMS["trainer.ckpt_path"] = None

# ===============
# Optimizer
# ===============
TRAIN_COMMON_PARAMS["opt.lr"] = 1e-5
TRAIN_COMMON_PARAMS["opt.weight_decay"] = 1e-3

# ===================================================================================================================
# Model
# ===================================================================================================================
TRAIN_COMMON_PARAMS["model.momentum"] = 0.9
TRAIN_COMMON_PARAMS["model.dropout_rate"] = 0.5
TRAIN_COMMON_PARAMS["model.imaging_dropout"] = 0.25
TRAIN_COMMON_PARAMS["model.post_concat_inputs"] = None  # [('data.clinical_features',9),]
TRAIN_COMMON_PARAMS["model.post_concat_model"] = None  # (256,256)

TRAIN_COMMON_PARAMS["model.classification_task"] = label_type
TRAIN_COMMON_PARAMS["model.class_num"] = label_type.get_num_classes()
## Backbone parameters
TRAIN_COMMON_PARAMS["model.bb.input_channels_num"] = 1
TRAIN_COMMON_PARAMS["model.bb.num_features_imaging"] = 512
TRAIN_COMMON_PARAMS["model.bb.num_features_clinical"] = None  # 256


# TODO sagi, maybe set 'clinical' to 0 ?
if TRAIN_COMMON_PARAMS["model.bb.num_features_clinical"] is None:
    TRAIN_COMMON_PARAMS["model.bb.num_features"] = TRAIN_COMMON_PARAMS["model.bb.num_features_imaging"]
else:
    TRAIN_COMMON_PARAMS["model.bb.num_features"] = (
        TRAIN_COMMON_PARAMS["model.bb.num_features_imaging"] + TRAIN_COMMON_PARAMS["model.bb.num_features_clinical"]
    )

######################################
# Inference Common Params
######################################
INFER_COMMON_PARAMS = {}
INFER_COMMON_PARAMS["infer_filename"] = "infer_file.gz"
INFER_COMMON_PARAMS["checkpoint"] = "best_epoch.ckpt"  # Fuse TIP: possible values are 'best', 'last' or epoch_index.
INFER_COMMON_PARAMS["data.infer_folds"] = [heldout_fold]  # infer validation set
INFER_COMMON_PARAMS["data.batch_size"] = 4
INFER_COMMON_PARAMS["data.num_workers"] = num_workers
INFER_COMMON_PARAMS["model.classification_task"] = TRAIN_COMMON_PARAMS["model.classification_task"]
INFER_COMMON_PARAMS["trainer.accelerator"] = "gpu"
INFER_COMMON_PARAMS["trainer.ckpt_path"] = None
INFER_COMMON_PARAMS["trainer.num_devices"] = num_devices

INFER_COMMON_PARAMS["model.bb.input_channels_num"] = TRAIN_COMMON_PARAMS["model.bb.input_channels_num"]
INFER_COMMON_PARAMS["model.bb.num_features"] = TRAIN_COMMON_PARAMS["model.bb.num_features"]
INFER_COMMON_PARAMS["model.post_concat_inputs"] = TRAIN_COMMON_PARAMS["model.post_concat_inputs"]
INFER_COMMON_PARAMS["model.post_concat_model"] = TRAIN_COMMON_PARAMS["model.post_concat_model"]
INFER_COMMON_PARAMS["model.dropout_rate"] = TRAIN_COMMON_PARAMS["model.dropout_rate"]

######################################
# Analyze Common Params
######################################
EVAL_COMMON_PARAMS = {}
EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


def create_model(
    conv_inputs: Any,
    backbone_ch_num: Any,
    num_backbone_features: Any,
    post_concat_inputs: Any,
    post_concat_model: Any,
    dropout_rate: float,
) -> torch.nn.Module:
    """
    TODO: docu, fix type Any
    """
    model = Fuse_model_3d_multichannel(
        conv_inputs=conv_inputs,  # previously 'data.input'. could be either 'data.input.patch_volume' or  'data.input.patch_volume_orig'
        backbone=ResNet(conv_inputs=conv_inputs, ch_num=backbone_ch_num),
        # since backbone resnet contains pooling and fc, the feature output is 1D,
        # hence we use Head1dClassifier as classification head
        heads=[
            Head1DClassifier(
                head_name="classification",
                conv_inputs=[("model.backbone_features", num_backbone_features)],
                post_concat_inputs=post_concat_inputs,
                post_concat_model=post_concat_model,
                dropout_rate=dropout_rate,
                # append_dropout_rate=train_params['clinical_dropout'],
                # fused_dropout_rate=train_params['fused_dropout'],
                shared_classifier_head=None,
                layers_description=None,
                num_classes=2,
                # append_features=[("data.input.clinical", 8)],
                # append_layers_description=(256,128),
            ),
        ],
    )

    return model


#################################
# Train Template
#################################
def run_train(paths: dict, train_common_params: dict) -> None:
    # ==============================================================================
    # Logger
    # ==============================================================================
    fuse_logger_start(output_path=paths["model_dir"], console_verbose_level=logging.INFO)

    print("Fuse Train")
    print(f'model_dir={paths["model_dir"]}')
    print(f'cache_dir={paths["cache_dir"]}')

    # ==============================================================================
    # Data
    # ==============================================================================

    # Holds all datasets' params in one dictionary.
    # because most of the datasets params are common
    # TODO: consider use this method. only when runner is running.
    # For now we'll define it seperatly
    common_dataset_params = {}
    # example
    train_dataset_params = common_dataset_params
    train_dataset_params["train"] = True
    validation_dataset_params = common_dataset_params
    validation_dataset_params["train"] = False

    #### Split Data
    all_dataset = Duke.dataset(
        label_type=train_common_params["model.classification_task"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        # reset_cache=train_common_params["data.reset_cache"],
        reset_cache=False,
        sample_ids=train_common_params["data.sample_ids"],
        num_workers=train_common_params["data.train_num_workers"],
        train=False,
        verbose=False,
    )
    folds = dataset_balanced_division_to_folds(
        dataset=all_dataset,
        output_split_filename=paths["data_split_filename"],
        keys_to_balance=["data.ground_truth"],
        workers=0,  # todo: stuck in Export to dataframe
        nfolds=train_common_params["data.num_folds"],
        verbose=True,
    )

    ## sample ids:
    train_sample_ids = []
    for fold in train_common_params["data.train_folds"]:
        train_sample_ids += folds[fold]
    validation_sample_ids = []
    for fold in train_common_params["data.validation_folds"]:
        validation_sample_ids += folds[fold]

    #### Train Data
    print("Train Data:")

    ## Create dataset
    print("- Create Dataset:")
    train_dataset = Duke.dataset(
        label_type=train_common_params["model.classification_task"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        # reset_cache=train_common_params["data.reset_cache"],
        reset_cache=False,
        sample_ids=train_sample_ids,
        num_workers=train_common_params["data.train_num_workers"],
        train=True,
        verbose=False,
    )

    ## Create batch sampler
    print("- Create sampler:")
    sampler = BatchSamplerDefault(
        dataset=train_dataset,
        balanced_class_name="data.ground_truth",
        num_balanced_classes=train_common_params["model.class_num"],
        batch_size=train_common_params["data.batch_size"],
        balanced_class_weights=None,
        # workers = 0 # from michal, not sure if applicable / why
    )

    print("- Create sampler: Done")

    ## Create dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=sampler,
        collate_fn=CollateDefault(),
        num_workers=train_common_params["data.train_num_workers"],
    )
    print("Train Data: Done")

    #### Validation data
    print("Validation Data:")

    validation_dataset = Duke.dataset(
        label_type=train_common_params["model.classification_task"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        # reset_cache=train_common_params["data.reset_cache"],
        reset_cache=False,
        sample_ids=validation_sample_ids,
        num_workers=train_common_params["data.validation_num_workers"],
        train=False,
        verbose=False,
    )

    ## Create dataloader
    validation_dataloader = DataLoader(
        dataset=validation_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=train_common_params["data.batch_size"],
        num_workers=train_common_params["data.validation_num_workers"],
        collate_fn=CollateDefault(),
    )
    print("Validation Data: Done")

    ## Create model
    print("Model:")
    model = create_model(
        conv_inputs=(("data.input.patch_volume", 1),),
        backbone_ch_num=train_common_params["model.bb.input_channels_num"],
        num_backbone_features=train_common_params["model.bb.num_features"],
        post_concat_inputs=train_common_params["model.post_concat_inputs"],
        post_concat_model=train_common_params["model.post_concat_model"],
        dropout_rate=train_common_params["model.dropout_rate"],
    )
    print("Model: Done")

    # ==========================================================================================================================================
    #   Loss
    # ==========================================================================================================================================
    losses = {
        "cls_loss": LossDefault(
            pred="model.logits.classification", target="data.ground_truth", callable=F.cross_entropy, weight=1.0
        ),
    }

    # =========================================================================================================
    # Metrics
    # =========================================================================================================
    train_metrics = OrderedDict([("auc", MetricAUCROC(pred="model.output.classification", target="data.ground_truth"))])
    validation_metrics = copy.deepcopy(train_metrics)  # use the same metrics in validation as well

    best_epoch_source = dict(monitor="validation.metrics.auc", mode="max")

    # =====================================================================================
    #  Train - using PyTorch Lightning
    #  Create training objects, PL module and PL trainer.
    # =====================================================================================
    print("Fuse Train:")

    # create optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=train_common_params["opt.lr"],
        weight_decay=train_common_params["opt.weight_decay"],
    )

    # create scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    lr_sch_config = dict(scheduler=lr_scheduler, monitor="validation.losses.total_loss")

    # optimizier and lr sch - see pl.LightningModule.configure_optimizers return value for all options
    optimizers_and_lr_schs = dict(optimizer=optimizer, lr_scheduler=lr_sch_config)

    # create instance of PL module - FuseMedML generic version
    pl_module = LightningModuleDefault(
        model_dir=paths["model_dir"],
        model=model,
        losses=losses,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        best_epoch_source=best_epoch_source,
        optimizers_and_lr_schs=optimizers_and_lr_schs,
    )

    # create lightining trainer.
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        max_epochs=train_common_params["trainer.num_epochs"],
        accelerator=train_common_params["trainer.accelerator"],
        devices=train_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )

    # train
    pl_trainer.fit(
        pl_module, train_dataloader, validation_dataloader, ckpt_path=train_common_params["trainer.ckpt_path"]
    )

    print("Fuse Train: Done")


######################################
# Inference Common Params
######################################
# INFER_COMMON_PARAMS = {}
# INFER_COMMON_PARAMS["data.num_workers"] = TRAIN_COMMON_PARAMS["data.train_num_workers"]
# INFER_COMMON_PARAMS["data.batch_size"] = 4
# INFER_COMMON_PARAMS["infer_filename"] = os.path.join(PATHS["inference_dir"], "validation_set_infer.pickle")
# INFER_COMMON_PARAMS["checkpoint"] = "best"  # Fuse TIP: possible values are 'best', 'last' or epoch_index.

######################################
# Inference Template
######################################


def run_infer(paths: dict, infer_common_params: dict) -> None:
    create_dir(paths["inference_dir"])
    infer_file = os.path.join(paths["inference_dir"], infer_common_params["infer_filename"])
    checkpoint_file = os.path.join(paths["model_dir"], infer_common_params["checkpoint"])

    ## Logger
    fuse_logger_start(output_path=paths["inference_dir"], console_verbose_level=logging.INFO)
    print("Fuse Inference")
    print(f"infer_filename={infer_file}")

    #### create infer dataset
    folds = load_pickle(paths["data_split_filename"])  # assume exists and created in train func
    infer_sample_ids = []
    for fold in infer_common_params["data.infer_folds"]:
        infer_sample_ids += folds[fold]

    infer_dataset = Duke.dataset(
        label_type=infer_common_params["model.classification_task"],
        data_dir=paths["data_dir"],
        cache_dir=paths["cache_dir"],
        # reset_cache=train_common_params["data.reset_cache"],
        reset_cache=False,
        sample_ids=infer_sample_ids,
        num_workers=infer_common_params["data.num_workers"],
        train=False,
        verbose=False,
    )

    ## Create dataloader
    infer_dataloader = DataLoader(
        dataset=infer_dataset,
        shuffle=False,
        drop_last=False,
        batch_sampler=None,
        batch_size=infer_common_params["data.batch_size"],
        num_workers=infer_common_params["data.num_workers"],
        collate_fn=CollateDefault(),
    )

    print("- Create Model:")
    model = create_model(
        conv_inputs=(("data.input.patch_volume", 1),),
        backbone_ch_num=infer_common_params["model.bb.input_channels_num"],
        num_backbone_features=infer_common_params["model.bb.num_features"],
        post_concat_inputs=infer_common_params["model.post_concat_inputs"],
        post_concat_model=infer_common_params["model.post_concat_model"],
        dropout_rate=infer_common_params["model.dropout_rate"],
    )

    # load python lightning module
    pl_module = LightningModuleDefault.load_from_checkpoint(
        checkpoint_file, model_dir=paths["model_dir"], model=model, map_location="cpu", strict=True
    )

    # set the prediction keys to extract and dump into file (the ones used be the evaluation function).
    pl_module.set_predictions_keys(["model.output.classification", "data.ground_truth"])

    # create a trainer instance and predict
    pl_trainer = pl.Trainer(
        default_root_dir=paths["model_dir"],
        accelerator=infer_common_params["trainer.accelerator"],
        devices=infer_common_params["trainer.num_devices"],
        auto_select_gpus=True,
    )
    predictions = pl_trainer.predict(pl_module, infer_dataloader, return_predictions=True)

    # convert list of batch outputs into a dataframe
    infer_df = convert_predictions_to_dataframe(predictions)
    save_dataframe(infer_df, infer_file)

    print("Fuse Inference: Done")


######################################
# Eval Template
######################################
# EVAL_COMMON_PARAMS = {}
# EVAL_COMMON_PARAMS["infer_filename"] = INFER_COMMON_PARAMS["infer_filename"]


def run_eval(paths: dict, eval_common_params: dict) -> None:
    fuse_logger_start(output_path=None, console_verbose_level=logging.INFO)
    print("Fuse Eval")

    infer_file = os.path.join(paths["inference_dir"], eval_common_params["infer_filename"])

    # metrics
    metrics = OrderedDict(
        [
            ("operation_point", MetricApplyThresholds(pred="model.output.classification")),  # will apply argmax
            ("accuracy", MetricAccuracy(pred="results:metrics.operation_point.cls_pred", target="data.ground_truth")),
            (
                "roc",
                MetricROCCurve(
                    pred="model.output.classification",
                    target="data.ground_truth",
                    output_filename=os.path.join(paths["inference_dir"], "roc_curve.png"),
                ),
            ),
            ("auc", MetricAUCROC(pred="model.output.classification", target="data.ground_truth")),
        ]
    )

    # create evaluator
    evaluator = EvaluatorDefault()

    # run
    results = evaluator.eval(
        ids=None,
        data=infer_file,
        metrics=metrics,
        output_dir=paths["eval_dir"],
    )

    print("Fuse Eval: Done")
    return results


######################################
# Run
######################################
if __name__ == "__main__":
    # allocate gpus
    NUM_GPUS = 1

    # uncomment if you want to use specific gpus instead of automatically looking for free ones
    force_gpus = None  # [0]
    GPU.choose_and_enable_multiple_gpus(NUM_GPUS, force_gpus=force_gpus)

    RUNNING_MODES = ["train", "infer", "eval"]  # Options: 'train', 'infer', 'eval'

    # train
    if "train" in RUNNING_MODES:
        run_train(paths=PATHS, train_common_params=TRAIN_COMMON_PARAMS)

    # infer
    if "infer" in RUNNING_MODES:
        run_infer(paths=PATHS, infer_common_params=INFER_COMMON_PARAMS)

    # eval
    if "eval" in RUNNING_MODES:
        run_eval(paths=PATHS, eval_common_params=EVAL_COMMON_PARAMS)
