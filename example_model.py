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
        self.model = create_model('resnet18', pretrained=True, num_classes=1)

    def forward(self, tensor, labels=None):
        logits = torch.nn.functional.sigmoid(self.model(tensor))
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(logits, labels.to(self.device).unsqueeze(1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

