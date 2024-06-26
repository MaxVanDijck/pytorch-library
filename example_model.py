from transformers import PreTrainedModel, PretrainedConfig
import torch
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
import transformers
from timm import create_model

# from deepspeed.ops.adam import DeepSpeedCPUAdam


from transformers import ConvNextConfig, ConvNextForImageClassification
from transformers import ResNetConfig, ResNetForImageClassification


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
        self.model = create_model('resnext50d_32x4d', pretrained=False, num_classes=10)

    def forward(self, tensor, labels=None):
        logits = self.model(tensor)
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels.to(self.device), label_smoothing=0.1)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

