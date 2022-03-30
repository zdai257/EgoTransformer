import torch
import os
from os.path import join
import time
from transformers import BertTokenizer
from PIL import Image, ImageOps
import argparse
import matplotlib.pyplot as plt
from models import caption
from datasets import coco, utils
from configuration import *
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu, BleuScorer
from pycocoevalcap.meteor.meteor import Meteor, METEOR_JAR
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider, CiderScorer
from pycocoevalcap.spice.spice import Spice, SPICE_JAR



def create_caption_and_mask(start_t, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_t
    mask_template[:, 0] = False

    return caption_template, mask_template


def create_tag_token(tags=('na', 'na')):
    where_dict = {'indoor': 0, 'outdoor': 1, 'na': 2}
    when_dict = {'daytime': 0, 'night': 1, 'na': 2}
    tag_token = len(where_dict) * where_dict[tags[0]] + when_dict[tags[1]]
    return torch.tensor([tag_token])


'''
def calc_scores(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        (Spice(), "SPICE")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
'''


def predict_qualitative(config, sample_path, tags=None, checkpoint_path=None, map_location='cpu'):

    if checkpoint_path is None:
        model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
    elif os.path.exists(checkpoint_path):
        ### Select Model ###
        if config.modality == 'image':
            # Original CATR
            model, criterion = caption.build_model(config)
        elif config.modality == 'ego':
            # Ego Model
            model, criterion = caption.build_model_ego(config)
        elif config.modality == 'video':
            # Video Model
            model, criterion = caption.build_model_bs(config)
        print("Loading Checkpoint...")
        checkpoint_tmp = torch.load(checkpoint_path, map_location=map_location)
        model.load_state_dict(checkpoint_tmp['model'])
        print("Current checkpoint epoch = %d" % checkpoint_tmp['epoch'])

    else:
        raise NotImplementedError('Give valid checkpoint path')

    device = torch.device(map_location)
    print(f'Initializing Device: {device}')

    start_t_tokenizer = time.time()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True, local_files_only=False)
    print("Loading pretrained Tokenizer takes: %.2fs" % (time.time() - start_t_tokenizer))

    start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
    end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)
    print("Total Vocal = ", tokenizer.vocab_size)
    print("Start Token: {}; End Token: {}; Padding: {}".format(tokenizer._cls_token, tokenizer._sep_token,
                                                               tokenizer._pad_token))

    @torch.no_grad()
    def evaluate(sample_t, cap_t, cap_mask_t, tag_token_t):
        model.eval()
        decoded_batch_beams = None

        for i in range(config.max_position_embeddings - 1):
            if config.modality == 'ego':
                predictions = model(sample_t, cap_t, cap_mask_t, tag_token_t)
            else:
                predictions = model(sample_t, cap_t, cap_mask_t)
            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            if predicted_id[0] == 102:
                break

            cap_t[:, i + 1] = predicted_id[0]
            cap_mask_t[:, i + 1] = False
        out = cap_t
        '''
        ### Greedy ###
        #out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=None, diverse_m=3)
        ### Beam Search ###
        out, decoded_batch_beams = model.decode(sample, cap, cap_mask, beam_width=5, diverse_m=3)
        '''
        return out, decoded_batch_beams

    if isinstance(sample_path, str):
        # Load Image
        image = Image.open(sample_path)
        # Transpose with respect to EXIF data
        image = ImageOps.exif_transpose(image)
        w, h = image.size
        print("PIL Image width: {}, height: {}".format(w, h))
        sample = coco.val_transform(image)
        sample = sample.unsqueeze(0)

        # Load skeleton caption
        cap, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

        # Load tag_token, tag_mask
        if tags is not None:
            tag_token = create_tag_token(tags)
        else:
            tag_token = torch.tensor([8])

        output, outputs = evaluate(sample, cap, cap_mask, tag_token)

        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        print('\n' + result.capitalize() + '\n')

        cap_dict = {sample_path.split('/')[-1]: [result]}

    elif isinstance(sample_path, list):
        cap_dict = {}

        for idx, s_path in enumerate(sample_path):
            # Load Image
            image = Image.open(s_path)
            # Transpose with respect to EXIF data
            image = ImageOps.exif_transpose(image)
            w, h = image.size
            print("PIL Image width: {}, height: {}".format(w, h))
            sample = coco.val_transform(image)
            sample = sample.unsqueeze(0)

            # Load skeleton caption
            cap, cap_mask = create_caption_and_mask(start_token, config.max_position_embeddings)

            # Load tag_token, tag_mask
            tag = tags[idx]
            if tag is not None:
                tag_token = create_tag_token(tag)
            else:
                tag_token = torch.tensor([8])

            output, outputs = evaluate(sample, cap, cap_mask, tag_token)

            result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
            print('\n' + result.capitalize() + '\n')
            sample_dict = {s_path.split('/')[-1]: [result]}
            cap_dict.update(sample_dict)

    else:
        raise TypeError("Sample_path invalid!")

    return cap_dict


