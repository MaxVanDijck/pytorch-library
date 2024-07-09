from transformers import PreTrainedModel, PretrainedConfig
import torch
from timm.models.resnet import BasicBlock, Bottleneck, ResNet
import transformers
from timm import create_model
import torch.nn as nn

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
        self.model = ISICModel(config)

    def forward(self, tensor, labels=None):
        logits = torch.nn.functional.sigmoid(self.model(tensor))
        if labels is not None:
            loss = torch.nn.functional.binary_cross_entropy(logits, labels.to(self.device).unsqueeze(1))
            return {"loss": loss, "logits": logits}
        return {"logits": logits}

class ISICModel(PreTrainedModel):
    def __init__(self, config):
        super(ISICModel, self).__init__(config)
        self.model = create_model('tf_efficientnet_b1.ns_jft_in1k', pretrained=False, num_classes=1)

        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        output = self.linear(pooled_features)
        return output

import torch.nn.functional as F

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'
