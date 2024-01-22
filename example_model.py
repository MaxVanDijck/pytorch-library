from transformers import PreTrainedModel, PretrainedConfig
import torch
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
from models.cnn.alexnet import AlexNet
from models.cnn.resnet import ResNet50
import pytorch_lightning as pl
import transformers

from deepspeed.ops.adam import DeepSpeedCPUAdam



from fastai.vision.all import xse_resnext50, Mish, MaxPool

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
        self.model = xse_resnext50(n_out=10, act_cls=Mish, sa=1, sym=0, pool=MaxPool)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor.to(self.device))
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels.to(self.device), label_smoothing=0.1)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}






