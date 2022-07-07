import os.path
import pickle
import random

import torch.utils.data
from configs.intention_net_config import IntentionNetConfig
import torch
from PIL import Image
import json
import numpy as np
from torchvision.transforms import Resize, ToTensor, Normalize, CenterCrop
import itertools
import operator
from utils.metrics import kl
from tqdm import tqdm
import ast


def most_common(L):
  # https://stackoverflow.com/questions/1518522/find-the-most-common-element-in-a-list
  SL = sorted((x, i) for i, x in enumerate(L))
  groups = itertools.groupby(SL, key=operator.itemgetter(0))
  def _auxfun(g):
    item, iterable = g
    count = 0
    min_index = len(L)
    for _, where in iterable:
      count += 1
      min_index = min(min_index, where)
    return count, -min_index
  return max(groups, key=_auxfun)[0]


def get_image(image_path, img_size, normalized=True):
    img = Image.open(image_path)
    img = CenterCrop((img.size[0]//2, img.size[0]))(img)
    img = Resize(img_size)(img)
    img = ToTensor()(img)
    if normalized:
        img = Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])(img)
    return img


class IntentionDataset(torch.utils.data.Dataset):
    def __init__(self, config, mode, transform=None, single_gaussian=False, seed=0):
        self.config = config
        self.mode = mode
        self.single_gaussian = single_gaussian # make the webcam gaze to be a selected centered single gaussian
        self.transform = transform
        self.dataset_path = self.config.dataset_path
        if self.config.mind_type != 'distraction':
            seq_json_path = 'dataset/{}/{}_{}_seq.json'.format('intention_large', self.config.data_mode, self.mode)
        else:
            seq_json_path = 'dataset/{}/{}_{}_{}_seq.json'.format('intention_large', self.config.distraction_mode, self.config.data_mode, self.mode)

        with open(seq_json_path, 'r') as f:
            self.intention_dataset_seqs = json.load(f)
        count = 0
        if self.mode == 'train':
            self.sequences = {}
            for intention, v in self.intention_dataset_seqs.items():
                intention = int(intention)
                for s in v:
                    curr_seq = [intention, *s]
                    edge_id = curr_seq[1][:2]
                    if intention != 3:
                        if edge_id not in self.sequences:
                            self.sequences[edge_id] = {}
                        if intention not in self.sequences[edge_id]:
                            self.sequences[edge_id][intention] = []
                        self.sequences[edge_id][intention].append(curr_seq)
                        count += len(curr_seq[2])
                    # else:
                    #     edge_id = '' # for go_straight intention
                    #     if edge_id not in self.sequences:
                    #         self.sequences[edge_id] = {}
                    #     if intention not in self.sequences[edge_id]:
                    #         self.sequences[edge_id][intention] = []
                    #     self.sequences[edge_id][intention].append(curr_seq)
            self.sequence_ids = list(self.sequences.keys())
            self.sequence_ids.sort()
        else:
            self.sequences = []
            for intention, v in self.intention_dataset_seqs.items():
                intention = int(intention)
                for s in v:
                    if intention != 3:
                        curr_seq = [intention, *s]
                        self.sequences.append(curr_seq)
        self.cache_dir = 'cc-attention/cache/{}/{}/{}/'.format('intention_large', self.config.data_mode, self.mode)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.prepare_cache()

    def prepare_cache(self):
        if self.mode == 'train':
            for edge_id in tqdm(self.sequences.keys(), total=len(self.sequences)):
                curr_interserction_cluster = self.sequences[edge_id]
                for intention in curr_interserction_cluster.keys():
                    for seq_id, seq in enumerate(curr_interserction_cluster[intention]):
                        pickle_path = os.path.join(self.cache_dir, str(edge_id) + "_" + str(intention) + "_" + str(seq_id)) + ".pl"
                        if not os.path.exists(pickle_path):
                            res = self.load_sequence(seq[2], intention)
                            with open(pickle_path, 'wb') as f:
                                pickle.dump(res, f)
        else:
            for seq_id, seq in tqdm(enumerate(self.sequences), total=len(self.sequences)):
                intention = int(self.sequences[seq_id][0])
                pickle_path = os.path.join(self.cache_dir,
                                           str(seq_id) + ".pl")
                if not os.path.exists(pickle_path):
                    res = self.load_sequence(seq[2], intention)
                    with open(pickle_path, 'wb') as f:
                        pickle.dump(res, f)

    def get_intention_sample_weights(self):
        weights = []
        assert self.mode == 'train'
        for edge_id in self.sequence_ids:
            cumulative_gaze_path = os.path.join('dataset/intersection_intentions/manual', edge_id)
            cumulative_gazes = []
            for file in os.listdir(cumulative_gaze_path):
                if '.npy' in file:
                    cumulative_gaze = np.load(os.path.join(cumulative_gaze_path, file))
                    cumulative_gazes.append(cumulative_gaze)
            kl_score = 1
            for i in range(len(cumulative_gazes)-1):
                for j in range(i+1, len(cumulative_gazes)):
                    kl_score += kl(cumulative_gazes[i], cumulative_gazes[j])
            if len(cumulative_gazes) > 1:
                kl_score /= len(cumulative_gazes) * (len(cumulative_gazes)-1) / 2

            num_seq_in_cluster = 0
            for _, seq in self.sequences[edge_id].items():
                num_seq_in_cluster += len(seq)
            kl_score *= num_seq_in_cluster
            weights.append(kl_score)
        return weights

    def __getitem__(self, item):
        # if not os.path.exists(os.path.join(self.cache_dir, str(item) + ".pl")):

        if self.mode != 'train':
            sequence_index = item
            seq = self.sequences[sequence_index][2]
            intention = int(self.sequences[item][0])
            return self.load_sequence(seq, intention)
        else:
            intersection_edge_id = self.sequence_ids[item]
            curr_interserction_cluster = self.sequences[intersection_edge_id]
            results = []
            intenions = list(curr_interserction_cluster.keys())
            random.shuffle(intenions)
            for intention in intenions:
                if len(results) == 2:
                    break
                seq_id = random.choice(np.arange(0, len(curr_interserction_cluster[intention])))
                pickle_path = os.path.join(self.cache_dir, str(intersection_edge_id) + "_" + str(intention) + "_" + str(seq_id)) + ".pl"
                with open(pickle_path, 'rb') as f:
                    res = pickle.load(f)
                results.append(res)
            if len(results) < 2:
                results.append(results[-1]) # hacky way to circumvent the batch size issue
            return results

    def load_sequence(self, seq, intention):
        eye_tracker_gaze_maps = torch.zeros([self.config.sliding_window_size, 32, 64], dtype=torch.float)
        webcam_gaze_maps = torch.zeros([self.config.sliding_window_size, 32, 64], dtype=torch.float)
        rgbs = torch.zeros([self.config.sliding_window_size, 3, 256, 512], dtype=torch.float)
        orig_rgbs = torch.zeros([self.config.sliding_window_size, 3, 256, 512], dtype=torch.float)
        masks = torch.zeros([self.config.sliding_window_size, 1], dtype=torch.float)
        for i in range(len(seq)):
            img_path = seq[i]
            eye_tracker_path = img_path.replace('images_4hz', 'heatmap_4hz_20_eye_tracker')
            rgbs[i] = get_image(img_path, (256, 512), True)
            orig_rgbs[i] = get_image(img_path, (256, 512), False)
            eye_tracker_gaze_maps[i] = get_image(eye_tracker_path, (32, 64), False)[0]

            if self.config.data_mode == 'manual':
                webcam_path = img_path.replace('images_4hz', 'raw_heatmap_4hz_20_gaze_recorder')
                webcam_gaze_maps[i] = get_image(webcam_path, (32, 64), False)[0]

            masks[i] = 1
        return orig_rgbs, rgbs, webcam_gaze_maps, eye_tracker_gaze_maps, torch.tensor([intention]), masks

    def __len__(self):
        if self.mode == 'train':
            return len(self.sequence_ids)
        else:
            return len(self.sequences)
