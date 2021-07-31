# @Time   : 2020/10/19
# @Author : Yupeng Hou
# @Email  : houyupeng@ruc.edu.cn

# UPDATE
# @Time   : 2021/7/9
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

import numpy as np
import torch

from recbole.data.dataset import Kg_Seq_Dataset, SequentialDataset, Dataset
from recbole.data.interaction import Interaction
from recbole.sampler import SeqSampler
from recbole.utils.enum_type import FeatureType, FeatureSource


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

class DIENDataset(SequentialDataset):
    """:class:`DIENDataset` is based on :class:`~recbole.data.dataset.sequential_dataset.SequentialDataset`.
    It is different from :class:`SequentialDataset` in `data_augmentation`.
    It add users' negative item list to interaction.

    The original version of sampling negative item list is implemented by Zhichao Feng (fzcbupt@gmail.com) in 2021/2/25,
    and he updated the codes in 2021/3/19. In 2021/7/9, Yupeng refactored SequentialDataset & SequentialDataLoader,
    then refactored DIENDataset, either.

    Attributes:
        augmentation (bool): Whether the interactions should be augmented in RecBole.
        seq_sample (recbole.sampler.SeqSampler): A sampler used to sample negative item sequence.
        neg_item_list_field (str): Field name for negative item sequence.
        neg_item_list (torch.tensor): all users' negative item history sequence.
    """
    def __init__(self, config):
        super().__init__(config)

        list_suffix = config['LIST_SUFFIX']
        neg_prefix = config['NEG_PREFIX']
        self.seq_sampler = SeqSampler(self)
        self.neg_item_list_field = neg_prefix + self.iid_field + list_suffix
        self.neg_item_list = self.seq_sampler.sample_neg_sequence(self.inter_feat[self.iid_field])

    def data_augmentation(self):
        """Augmentation processing for sequential dataset.

        E.g., ``u1`` has purchase sequence ``<i1, i2, i3, i4>``,
        then after augmentation, we will generate three cases.

        ``u1, <i1> | i2``

        (Which means given user_id ``u1`` and item_seq ``<i1>``,
        we need to predict the next item ``i2``.)

        The other cases are below:

        ``u1, <i1, i2> | i3``

        ``u1, <i1, i2, i3> | i4``
        """
        self.logger.debug('data_augmentation')

        self._aug_presets()

        self._check_field('uid_field', 'time_field')
        max_item_list_len = self.config['MAX_ITEM_LIST_LENGTH']
        self.sort(by=[self.uid_field, self.time_field], ascending=True)
        last_uid = None
        uid_list, item_list_index, target_index, item_list_length = [], [], [], []
        seq_start = 0
        for i, uid in enumerate(self.inter_feat[self.uid_field].numpy()):
            if last_uid != uid:
                last_uid = uid
                seq_start = i
            else:
                if i - seq_start > max_item_list_len:
                    seq_start += 1
                uid_list.append(uid)
                item_list_index.append(slice(seq_start, i))
                target_index.append(i)
                item_list_length.append(i - seq_start)

        uid_list = np.array(uid_list)
        item_list_index = np.array(item_list_index)
        target_index = np.array(target_index)
        item_list_length = np.array(item_list_length, dtype=np.int64)

        new_length = len(item_list_index)
        new_data = self.inter_feat[target_index]
        new_dict = {
            self.item_list_length_field: torch.tensor(item_list_length),
        }

        for field in self.inter_feat:
            if field != self.uid_field:
                list_field = getattr(self, f'{field}_list_field')
                list_len = self.field2seqlen[list_field]
                shape = (new_length, list_len) if isinstance(list_len, int) else (new_length,) + list_len
                list_ftype = self.field2type[list_field]
                dtype = torch.int64 if list_ftype in [FeatureType.TOKEN, FeatureType.TOKEN_SEQ] else torch.float64
                new_dict[list_field] = torch.zeros(shape, dtype=dtype)

                value = self.inter_feat[field]
                for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                    new_dict[list_field][i][:length] = value[index]

                # DIEN
                if field == self.iid_field:
                    new_dict[self.neg_item_list_field] = torch.zeros(shape, dtype=dtype)
                    for i, (index, length) in enumerate(zip(item_list_index, item_list_length)):
                        new_dict[self.neg_item_list_field][i][:length] = self.neg_item_list[index]

        new_data.update(Interaction(new_dict))
        self.inter_feat = new_data

