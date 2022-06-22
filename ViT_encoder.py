import os.path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from models import utils
from models.caption import MLP
from transformers import ViTModel, ViTConfig
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification


class ViTEncoder(nn.Module):

    def __init__(self, num_layers=2, hidden_dim=256):
        super().__init__()
        if os.path.exists("./vit_classify-base-patch16-224"):
            backbone = ViTForImageClassification.from_pretrained("./vit_classify-base-patch16-224")
        else:
            backbone = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
            backbone.save_pretrained("./vit_classify-base-patch16-224")
        #print(backbone)
        #for k, v in backbone.named_parameters():
        #    print(k, v.shape)

        for name, parameter in backbone.named_parameters():
            if 1:
                parameter.requires_grad_(True)

        return_layers = {'vit': 'vit2'}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.classifier0 = nn.Linear(in_features=768, out_features=hidden_dim)

        self.classifier = MLP(hidden_dim * 197, hidden_dim=512, output_dim=hidden_dim, num_layers=num_layers)

        self.where_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3),
            nn.Softmax(),
        )
        self.when_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3),
            nn.Softmax(),
        )
        self.whom_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features=hidden_dim, out_features=3),
            nn.Softmax(),
        )
        #for name, parameter in self.classifier.named_parameters():
        #    if 1:
        #        parameter.requires_grad_(True)

    def forward(self, x):
        #print(self.body)

        xs = self.body(x)
        xs = xs[next(iter(xs))].last_hidden_state
        #print(xs.shape)
        xs = F.relu(self.classifier0(xs))
        #print(xs.shape)
        xf = F.relu(self.classifier(xs.flatten(1)))
        #print(xf.shape)

        return {
            'where': self.where_head(xf),
            'when': self.when_head(xf),
            'whom': self.whom_head(xf)
        }


def build_ViTEncoder(config):
    vit = ViTEncoder(hidden_dim=config.hidden_dim)
    #print(vit)
    #exit()
    return vit
