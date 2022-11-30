import torch
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from .loss import get_loss_func

class MultiModalityPerceiverWrapper(torch.nn.Module):
    def __init__(self,        
        ligand:dict,
        target:dict,
        classifier_head:dict,
        loss_fn:str,
        loss_fn_kwargs:dict,        
    ):
        super().__init__()

        self._ligand = ligand
        self._target = target
        self._classifier_head = classifier_head
        self._loss_fn = loss_fn
        self._loss_fn_kwargs = loss_fn_kwargs        
        self._loss_func = get_loss_func(self._loss_fn, **self._loss_fn_kwargs)      
                    
        ligand_modality = InputModality(
            name='ligand',
            input_channels=self._ligand['num_tokens'],  # number of channels for each token of the input
            input_axis=1,  # number of axes, 3 for video)
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
        )
        target_modality = InputModality(
            name='target',
            input_channels=self._target['num_tokens'],  # number of channels for each token of the input
            input_axis=1,  # number of axes, 2 for images
            num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
            max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
        )

        self.body = MultiModalityPerceiver(
            modalities=(ligand_modality, target_modality,),
            depth=8,  # depth of net, combined with num_latent_blocks_per_layer to produce full Perceiver
            num_latents=12,
            # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim=64,  # latent dimension
            cross_heads=1,  # number of heads for cross attention. paper said 1
            latent_heads=8,  # number of heads for latent self attention, 8
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=self._classifier_head['num_classes'],  # output number of classes   #TODO: fixme: change into 2 classes
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=True,
            num_latent_blocks_per_layer=6  # Note that this parameter is 1 in the original Lucidrain implementation
            # whether to weight tie layers (optional, as indicated in the diagram)
        )                    

    def forward(self, batch):

        ligand = batch['data.input.tokenized_ligand']
        ligand = F.one_hot(ligand, num_classes=self._ligand['num_tokens'])
        ligand = ligand.to(torch.float32)
        ligand.requires_grad_()
        ligand_attention_mask = batch['data.input.tokenized_ligand_attention_mask']

        target = batch['data.input.tokenized_target']
        target = F.one_hot(target, num_classes=self._target['num_tokens'])
        target = target.to(torch.float32)
        target.requires_grad_()
        target_attention_mask = batch['data.input.tokenized_target_attention_mask']

        logits = self.body({
            'ligand': ligand, #[...,None],
            'target': target, #[...,None],
        })

        cls_preds = F.softmax(logits, dim=1)

        batch[f"model.logits.{self._classifier_head['head_name']}"] = logits
        batch[f"model.output.{self._classifier_head['head_name']}"] = cls_preds

        return batch

    def loss(self, pred, gt):        
        val = self._loss_func(pred, gt)
        return val
        
