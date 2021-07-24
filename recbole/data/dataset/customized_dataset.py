# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

"""
recbole.data.customized_dataset
##################################

We only recommend building customized datasets by inheriting.

Customized datasets named ``[Model Name]Dataset`` can be automatically called.
"""

import os

import pandas as pd
import torch

from recbole.data.dataset import Dataset
from recbole.data.dataset import Kg_Seq_Dataset
from recbole.utils import FeatureSource, FeatureType


class GRU4RecKGDataset(Kg_Seq_Dataset):

    def __init__(self, config):
        super().__init__(config)


class KSRDataset(Kg_Seq_Dataset):

    def __init__(self, config):
        super().__init__(config)


class GitHubDataset(Dataset):

    def __init__(self, config, root, debug=True, overwrite=True):
        self.root = root
        self.inter_processed = "github.train"
        self.debug = debug
        self.overwrite = overwrite
        super().__init__(config)

    def _load_data(self, token, dataset_path):

        self._load_inter_feat(token, dataset_path)
        self.user_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.USER, 'uid_field')
        self.item_feat = self._load_user_or_item_feat(token, dataset_path, FeatureSource.ITEM, 'iid_field')
        self._load_additional_feat(token, dataset_path)

    def _load_github_data(self, inter_feat_path, split='train'):
        if split == 'train':
            inter_feat_path_raw = os.path.join(self.root, "Events_202105.csv")
        elif split == 'test' or split == 'val':
            inter_feat_path_raw = os.path.join(self.root, "Events_202106.csv")
        else:
            raise NotImplementedError
        self.logger.info(f"Processing {inter_feat_path_raw}")
        if self.debug:
            df = pd.read_csv(inter_feat_path_raw, sep=',', nrows=1000)
        else:
            reader = pd.read_csv(inter_feat_path_raw, sep=',', chunksize=10000, iterator=True)
            df = pd.concat(reader, ignore_index=True)
        df[['created_at']] = pd.to_datetime(df.created_at, infer_datetime_format=True)
        df.created_at = pd.to_numeric(df.created_at) / 1e9
        events_li = df.type.unique().tolist()
        events_d = {v: k for k, v in enumerate(events_li)}
        df.type = df.type.apply(lambda x: events_d[x])

        ##########################################
        # Drop unused columns
        ##########################################
        # TODO
        df.drop(['repo_name', 'actor_login', 'actor_url', "repo_owner_id", "type"], axis=1, inplace=True)
        df.astype({
            'created_at': float
        }, copy=False)
        df[['rating']] = 1.0
        df.rename(columns={
            'actor_id': 'user_id',
            'repo_id': 'item_id',
            'created_at': 'timestamp',
        }, inplace=True)
        print(df.info())
        self.logger.info(f"Saving processed file: {inter_feat_path_raw}")
        print("Saving processed files ...")
        torch.save(df, inter_feat_path)
        return df

    def _load_inter_feat(self, token, dataset_path):
        if self.benchmark_filename_list is None:
            inter_feat_path = os.path.join(self.root, self.inter_processed)
            if not os.path.isfile(inter_feat_path) or self.overwrite:
                self.logger.info(f"{self.inter_processed} does not exists, processing from raw files...")
                self.inter_feat = self._load_github_data(inter_feat_path, split='train')
            else:
                self.inter_feat = torch.load(inter_feat_path)

        else:
            sub_inter_lens = []
            sub_inter_feats = []
            for filename in self.benchmark_filename_list:
                file_path = os.path.join(dataset_path, f'{token}.{filename}.inter')
                if os.path.isfile(file_path) and not self.overwrite:
                    # if os.stat("file").st_size == 0:
                    #
                    #     length = 0
                    # else:
                    temp = torch.load(file_path)
                else:
                    self.logger.info(f"{token}.{filename}.inter does not exists, processing from raw files...")
                    temp = self._load_github_data(file_path, split=filename)
                length = len(temp)
                sub_inter_feats.append(temp)
                sub_inter_lens.append(length)
            inter_feat = pd.concat(sub_inter_feats)
            self.inter_feat, self.file_size_list = inter_feat, sub_inter_lens
        self.set_github_dataset_types()

    def set_github_dataset_types(self):
        params = [
            ('user_id', FeatureType.TOKEN),
            ('item_id', FeatureType.TOKEN),
            ('timestamp', FeatureType.FLOAT),
            ('rating', FeatureType.FLOAT),
        ]
        for field, ftype in params:
            self.field2source[field] = FeatureSource.INTERACTION
            self.field2type[field] = ftype
            self.field2seqlen[field] = 1

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()
        if self.benchmark_filename_list is None:
            self._data_filtering()

        self._remap_ID_all()
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()

    def _load_user_or_item_feat(self, token, dataset_path, source, field_name):
        # TODO
        pass
