import transformers
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor, ViTModel
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from models import utils
from datasets import coco
import ViT_encoder
from configuration import *
from PIL import Image, ImageOps
import os
from os.path import join
import numpy as np
import math
import sys
import tqdm


'''
#feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
feature_extractor = ViTFeatureExtractor.from_pretrained("./vit-feature_extractor")
#model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("./vit-base-224")
'''


def criteria(loss_fun, output, context, dev, weights=(0.9, 0.69, 0.49)):
    losses = 0
    for i, key in enumerate(output):
        losses += weights[i] / sum(weights) * loss_fun(output[key], context[key].to(dev))
    return losses


def train_an_epoch(config, model, loss_func, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    #loss_func.train()
    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for i, tuples in enumerate(data_loader):
            inputs, contexts = tuples[6], tuples[7]

            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)

            outputs = model(inputs['pixel_values'])

            loss = criteria(loss_func, outputs, contexts, device)
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(config, model, loss_func, data_loader, device):
    model.eval()
    loss_func.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for i, tuples in enumerate(data_loader):
            inputs, contexts = tuples[6], tuples[7]
            inputs['pixel_values'] = inputs['pixel_values'].squeeze(1).to(device)

            outputs = model(inputs['pixel_values'])

            loss = criteria(loss_func, outputs, contexts, device)
            loss_value = loss.item()
            validation_loss += loss_value

            pbar.update(1)

    return validation_loss / total


def main(config):
    available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    for idx, dev in enumerate(available_gpus):
        print("Available GPU-{} name: {}".format(idx, dev))
    device = torch.device(config.device)
    print(f'Initializing Device: {device}')

    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ViT_encoder.build_ViTEncoder(config)
    #for k,v in model.named_parameters():
    #    print(k)
    #print(list(model.parameters())[0].shape)
    model.to(device)

    n_parameters = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters}")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=1e-5, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20)

    criterion = torch.nn.CrossEntropyLoss()

    # Dataset
    if config.modality == 'ego':
        dataset_train = coco.build_dataset_egocap(config, mode='training')
        dataset_val = coco.build_dataset_egocap(config, mode='validation')
    else:
        raise TypeError("Input Modality not supported!")
    print(f"Train: {len(dataset_train)}")
    print(f"Valid: {len(dataset_val)}")

    # Sampler


    # DataLoader
    data_loader_train = DataLoader(dataset_train, config.batch_size,
                                   drop_last=False, num_workers=config.num_workers)
    data_loader_val = DataLoader(dataset_val, config.batch_size,
                                 drop_last=False, num_workers=config.num_workers)

    # Free GPU memory n allow growth
    torch.cuda.empty_cache()

    save_dir = 'vit_checks'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print("Start Training..")
    for epoch in range(0, 30):
        print(f"Epoch: {epoch}")
        epoch_loss = train_an_epoch(config, model, criterion, data_loader_train,
                                    optimizer, device, epoch, config.clip_max_norm)
        lr_scheduler.step()
        print(f"Training Loss: {epoch_loss}")

        validation_loss = evaluate(config, model, criterion, data_loader_val, device)
        print(f"Validation Loss: {validation_loss}")

        if (epoch + 1) % 5 == 0:
            model_name = 'ViT-epoch{}_loss{}.pth'.format(epoch, round(validation_loss * 100))
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }, join(save_dir, model_name))
        print()


if __name__ == "__main__":
    config = ConfigEgo()
    main(config)
