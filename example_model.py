from transformers import PreTrainedModel, PretrainedConfig
import torch
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
import pytorch_lightning as pl
import transformers

from deepspeed.ops.adam import DeepSpeedCPUAdam

BLOCK_MAPPING = {"basic": BasicBlock, "bottleneck": Bottleneck}
class Config(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="basic",
        **kwargs,
    ):
        self.block_type = block_type
        super().__init__(**kwargs)

class Model(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        block_layer = BLOCK_MAPPING[config.block_type]
        self.model = ResNet(
            block_layer,
            [3, 3, 4, 3],
            num_classes=200,
        )

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

