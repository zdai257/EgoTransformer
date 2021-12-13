import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision as tv

from PIL import Image
import numpy as np
import random
import os
from os.path import join
from transformers import BertTokenizer

from .utils import nested_tensor_from_tensor_list, read_json

MAX_DIM = 299


def under_max(image):
    if image.mode != 'RGB':
        image = image.convert("RGB")

    shape = np.array(image.size, dtype=np.float)
    long_dim = max(shape)
    scale = MAX_DIM / long_dim

    new_shape = (shape * scale).astype(int)
    image = image.resize(new_shape)

    return image


class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


train_transform = tv.transforms.Compose([
    RandomRotation(),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

train_transform_msvd = tv.transforms.Compose([
    RandomRotation(angles=[0]),
    tv.transforms.Lambda(under_max),
    tv.transforms.ColorJitter(brightness=[0.5, 1.3], contrast=[
                              0.8, 1.5], saturation=[0.2, 1.5]),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

val_transform = tv.transforms.Compose([
    tv.transforms.Lambda(under_max),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def read_msvd(msvd_ana_file, skipped_dir, min_frame_per_clip=5):
    # TODO: split MSVD into train / test
    pairs, Anns = [], []
    vid_anns = {}

    with open(msvd_ana_file, "r") as file:
        lines = file.readlines()
        print("Num of lines = ", len(lines))
        for line in lines:
            if line != "\n" and line[0] != "#":
                pairs.append(line)

    for pair in pairs:
        img_name = pair.split(' ')[0]
        sent_ana = pair[len(img_name) + 1:-1]

        if img_name not in vid_anns:
            # Create a list of [path_to_image]
            img_keys = []
            # Discard clip with less than N frames
            if len(os.listdir(join(skipped_dir, img_name))) < min_frame_per_clip:
                continue

            for frame in sorted(os.listdir(join(skipped_dir, img_name)), key=lambda x: int(x.split('.')[0][6:])):
                img_keys.append(join(skipped_dir, img_name, frame))

            vid_anns[img_name] = (img_keys, [sent_ana])
        else:
            if sent_ana in vid_anns[img_name][1]:
                continue
            vid_anns[img_name][1].append(sent_ana)

    for idx, (key, val) in enumerate(vid_anns.items()):
        for index in range(len(val[1])):
            for i in range(len(val[0]) - min_frame_per_clip + 1):
                tuple_item = (val[0][i:i + min_frame_per_clip], val[1][index])
                Anns.append(tuple_item)

    return Anns


def read_deepdiary(dirname, filename):
    with open(filename, "r") as file:
        pairs_str = file.read()
        pairs = pairs_str.split('\n')[1:]

    anns = []
    for pair in pairs:
        img_name = pair.split(' ')[0]
        sent_ana = pair.split('.jpg ')[-1]
        if img_name in os.listdir(dirname):
            anns.append((img_name, sent_ana))

    img_names = {}

    for pair in pairs:
        img_name = pair.split(' ')[0]
        sent_ana = pair.split('.jpg ')[-1]

        if img_name not in img_names:
            img_names[img_name] = [sent_ana]
        else:
            img_names[img_name].append(sent_ana)

    return anns, img_names


class DeepDiary(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=val_transform, mode='deepdiary'):
        super().__init__()

        self.root = root
        self.transform = transform

        if mode == 'deepdiary':
            # self.annot is a list of tuple of ('000000XXXXXX.jpg', "A man is sitting")
            self.annot = ann
        else:
            raise ValueError("DeepDiary does not support this mode.")

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))
        if self.transform:
            image = self.transform(image)

        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


class MSVDCaption(Dataset):
    def __init__(self, root, vid_ann, max_length, limit, transform=train_transform_msvd, mode='training',
                 frame_per_clip=5):
        super().__init__()
        self.root = root
        self.transform = transform

        # self.annot is a list of tuple (['frame0.jpg', 'frame1.jpg' ...], ["A man is sitting", "An old man", ...])
        if mode == 'training':
            self.annot = vid_ann
        elif mode == 'validation':
            pass
        else:
            raise ValueError("MSVD does not support this mode.")

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_lst, caption = self.annot[idx]
        images = []
        for img_path in image_lst:
            image = Image.open(img_path)
            if self.transform:
                image = self.transform(image)
            #image = nested_tensor_from_tensor_list(image.unsqueeze(0))
            images.append(image.unsqueeze(3))

        #print(images[0].shape, images[-1].shape)
        tensor_images = torch.cat(images, dim=3)
        #print("After cat shape = ", tensor_images.shape)
        nest_images = nested_tensor_from_tensor_list(tensor_images.unsqueeze(0))

        #print(caption)
        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)
        # caption_encoded is a dict of {'input_ids': <a list of vocab indexes>, 'attention_mask': [1, 1, 1, 0 ,0 ...]
        #print(caption_encoded)
        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return nest_images.tensors.squeeze(0), nest_images.mask.squeeze(0), caption, cap_mask


class CocoCaption(Dataset):
    def __init__(self, root, ann, max_length, limit, transform=train_transform, mode='training'):
        super().__init__()

        self.root = root
        self.transform = transform
        self.annot = [(self._process(val['image_id']), val['caption'])
                      for val in ann['annotations']]
        if mode == 'validation':
            self.annot = self.annot
        if mode == 'training':
            self.annot = self.annot[: limit]

        self.tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower=True, local_files_only=True)
        self.max_length = max_length + 1

    def _process(self, image_id):
        val = str(image_id).zfill(12)
        return val + '.jpg'

    def __len__(self):
        return len(self.annot)

    def __getitem__(self, idx):
        image_id, caption = self.annot[idx]
        image = Image.open(os.path.join(self.root, image_id))

        if self.transform:
            image = self.transform(image)
        image = nested_tensor_from_tensor_list(image.unsqueeze(0))

        caption_encoded = self.tokenizer.encode_plus(
            caption, max_length=self.max_length, pad_to_max_length=True, return_attention_mask=True, return_token_type_ids=False, truncation=True)

        caption = np.array(caption_encoded['input_ids'])
        cap_mask = (
            1 - np.array(caption_encoded['attention_mask'])).astype(bool)

        return image.tensors.squeeze(0), image.mask.squeeze(0), caption, cap_mask


def build_dataset(config, mode='training'):
    if mode == 'training':
        train_dir = os.path.join(config.dir, 'train2017')
        train_file = os.path.join(
            config.dir, 'annotations', 'captions_train2017.json')
        data = CocoCaption(train_dir, read_json(
            train_file), max_length=config.max_position_embeddings, limit=config.limit, transform=train_transform, mode='training')
        return data

    elif mode == 'validation':
        val_dir = os.path.join(config.dir, 'val2017')
        val_file = os.path.join(
            config.dir, 'annotations', 'captions_val2017.json')
        data = CocoCaption(val_dir, read_json(
            val_file), max_length=config.max_position_embeddings, limit=config.limit, transform=val_transform, mode='validation')
        return data

    else:
        raise NotImplementedError(f"{mode} not supported")

def build_dataset_deepdiary(config, mode='deepdiary'):
    data_dir = join(config.dir, 'amt_data')
    data_file = join(data_dir, 'amt_list.txt')
    anns, _ = read_deepdiary(data_dir, data_file)
    data = DeepDiary(data_dir, anns, max_length=config.max_position_embeddings,
                     limit=config.limit, transform=val_transform, mode=mode)
    return data

def build_dataset_msvd(config, mode='training'):
    if mode == 'training':
        msvd_data_dir = config.msvd_data_dir
        msvd_ana_file = join(msvd_data_dir, 'AllVideoDescriptions.txt')
        skipped_dir = join(msvd_data_dir, 'skipped')
        vid_anns = read_msvd(msvd_ana_file, skipped_dir, min_frame_per_clip=config.frame_per_clip)
        data = MSVDCaption(msvd_data_dir, vid_anns, max_length=config.max_position_embeddings,
                           limit=config.limit, transform=train_transform_msvd, mode=mode,
                           frame_per_clip=config.frame_per_clip)
        return data
    else:
        raise NotImplementedError(f"{mode} not supported")

