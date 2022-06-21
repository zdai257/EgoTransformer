import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from models import utils
from models.caption import MLP
from transformers import ViTModel, ViTConfig
from transformers import ViTFeatureExtractor, ViTModel


class ViTEncoder(nn.Module):

    def __init__(self, num_layers=2, hidden_dim=256):
        super().__init__()
        backbone = ViTModel.from_pretrained("./vit-base-224")
        #print(backbone)
        for name, parameter in backbone.named_parameters():
            if 1:
                parameter.requires_grad_(False)

        return_layers = {'encoder': 'vit2'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.classifier = MLP(768, hidden_dim, 3, num_layers)

        self.where_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3)
        )
        self.when_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3)
        )
        self.whom_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3)
        )
        for name, parameter in self.classifier.named_parameters():
            if 1:
                parameter.requires_grad_(True)

    def forward(self, x):
        #print(self.body)

        xs = self.body(x)
        for name, x in xs.items():
            xs = x.last_hidden_state
            break
        print(xs.shape)
        xf = self.classifier(xs.permute(0, 2, 1))
        print(xf.shape)

        return {
            'where': self.where_head(xf),
            'when': self.when_head(xf),
            'whom': self.whom_head(xf)
        }


def build_ViTEncoder(config):
    vit = ViTEncoder(hidden_dim=config.hidden_dim)
    #print(vit.body)

    return vit
