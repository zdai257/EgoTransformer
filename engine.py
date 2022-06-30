# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch

import math
import sys
import tqdm

from models import utils


def train_one_epoch(config, model, criterion, data_loader,
                    optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()

    epoch_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for tuples in data_loader:
            images, masks, caps, cap_masks = tuples[0], tuples[1], tuples[2], tuples[3]
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            if config.modality == 'ego':
                img = tuples[6]
                img_tensor = img['pixel_values'].squeeze(1).to(device)

                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1], img_tensor)
            else:
                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])
            loss_value = loss.item()
            epoch_loss += loss_value

            if not math.isfinite(loss_value):
                print(f'Loss is {loss_value}, stopping training')
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

            pbar.update(1)

    return epoch_loss / total


@torch.no_grad()
def evaluate(config, model, criterion, data_loader, device):
    model.eval()
    criterion.eval()

    validation_loss = 0.0
    total = len(data_loader)

    with tqdm.tqdm(total=total) as pbar:
        for tuples in data_loader:
            images, masks, caps, cap_masks = tuples[0], tuples[1], tuples[2], tuples[3]
            samples = utils.NestedTensor(images, masks).to(device)
            caps = caps.to(device)
            cap_masks = cap_masks.to(device)

            if config.modality == 'ego':
                img = tuples[6]
                img_tensor = img['pixel_values'].squeeze(1).to(device)

                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1], img_tensor)
            else:
                outputs = model(samples, caps[:, :-1], cap_masks[:, :-1])
            loss = criterion(outputs.permute(0, 2, 1), caps[:, 1:])

            validation_loss += loss.item()

            pbar.update(1)

    return validation_loss / total