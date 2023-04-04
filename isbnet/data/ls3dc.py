import numpy as np
import torch
import csv
from os.path import join
import re
import h5py
import pickle

from .custom import CustomDataset


class LS3DCDataset(CustomDataset):

    CLASSES = (
        'plane',
        'cylinder',
        'sphere',
        'cone'
    )

    BENCHMARK_SEMANTIC_IDXS = [i for i in range(4)]  # NOTE DUMMY values just for save results

    def get_filenames(self):
        if self.prefix == 'trainval':
            with open(join(self.data_folder_name, 'train_models.csv'), 'r', newline='') as f:
                filenames_train = list(csv.reader(f, delimiter=',', quotechar='|'))[0]
            with open(join(self.data_folder_name, 'test_models.csv'), 'r', newline='') as f:
                filenames_val = list(csv.reader(f, delimiter=',', quotechar='|'))[0]
            filenames = filenames_train + filenames_val 
        else:
            with open(join(self.data_folder_name, self.mode + '_models.csv'), 'r', newline='') as f:
                filenames = list(csv.reader(f, delimiter=',', quotechar='|'))[0]
                
        assert len(filenames) > 0, "Empty dataset."
        filenames = sorted(filenames * self.repeat)
        return filenames

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        ret = super().getInstanceInfo(xyz, instance_label, semantic_label)
        instance_num, instance_pointnum, instance_cls, pt_offset_label = ret
        # ignore instance of class 0 and reorder class id
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def load(self, filename):
        with h5py.File(filename, 'r') as h5_file:
            xyz = h5_file['noisy_points'][()] if 'noisy_points' in h5_file.keys() else None
            normals = h5_file['gt_normals'][()] if 'gt_normals' in h5_file.keys() else None
            instance_label = h5_file['gt_labels'][()] if 'gt_labels' in h5_file.keys() else None

            found_soup_ids = []
            soup_id_to_key = {}
            soup_prog = re.compile('(.*)_soup_([0-9]+)$')
            for key in list(h5_file.keys()):
                m = soup_prog.match(key)
                if m is not None:
                    soup_id = int(m.group(2))
                    found_soup_ids.append(soup_id)
                    soup_id_to_key[soup_id] = key

            features_data = []            
            found_soup_ids.sort()
            for i in range(len(found_soup_ids)):
                g = h5_file[soup_id_to_key[i]]
                meta = pickle.loads(g.attrs['meta'])
                features_data.append(meta)

            semantic_label = np.array([LS3DCDataset.CLASSES.index(features_data[inst]['type'].lower()) for inst in instance_label])


        if self.prefix == "test":
            semantic_label = np.zeros(xyz.shape[0], dtype=np.long)
            instance_label = np.zeros(xyz.shape[0], dtype=np.long)
        
        rgb = np.zeros(xyz.shape)

        # NOTE currently stpls3d does not have spps, we will add later
        # spp = np.zeros(xyz.shape[0], dtype=np.long)
        spp = np.arange(xyz.shape[0], dtype=np.long)

        return xyz, rgb, semantic_label, instance_label, spp
