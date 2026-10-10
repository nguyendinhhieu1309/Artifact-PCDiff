""""
Implements a data structure for loading point clouds from a dataset
"""

import csv
import os
import numpy as np
import torch as th


class BrokenDataset(th.utils.data.Dataset):
    def __init__(self, path, num_points, num_nn, norm_mode, num_samples=1, eval=False, augment=False):
        super().__init__()
        self.directory = path
        self.num_points = num_points
        self.num_nn = num_nn
        self.database = []
        self.norm_mode = norm_mode
        self.defects = ['broken']
        self.num_samples = num_samples
        self.eval = eval
        self.augment = augment

        missing = []
        with open(self.directory, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if not row:
                    continue
                datapoint = dict()
                path, obj = row[0].split('complete')
                datapoint['broken'] = path + 'broken' + obj.split('.')[0] + '_0.npy'
                datapoint['repair'] = path + 'repair' + obj.split('.')[0] + '_0.npy'

                required = ['broken'] if self.eval else ['broken', 'repair']
                for key in required:
                    normalized_path = datapoint[key].replace('\\', '/')
                    if not os.path.isfile(normalized_path):
                        missing.append(normalized_path)
                self.database.append(datapoint)

        if missing:
            preview = '\n'.join('  - ' + path for path in missing[:12])
            remainder = len(missing) - min(len(missing), 12)
            suffix = '\n  ... and {} more'.format(remainder) if remainder else ''
            raise FileNotFoundError(
                'Dataset is missing {} required file(s) for {} mode:\n{}{}'.format(
                    len(missing), 'evaluation' if self.eval else 'training', preview, suffix
                )
            )

    def __getitem__(self, file):
        filedict = self.database[file]
        name = filedict['broken']
        raw_path_broken = filedict['broken']
        # Normalize path separators to forward slashes, which work cross-platform
        normalized_path_broken = raw_path_broken.replace('\\', '/')
        pc_np = np.load(normalized_path_broken)  # Points belonging to the defective structure

        # Downsample point clouds
        num_pc = pc_np.shape[0]
        idx_pc = np.random.randint(0, num_pc, self.num_points-self.num_nn)
        pc_np = pc_np[idx_pc, :]

        # During training
        if not self.eval:  # Load and concat points belonging to the ground truth implant (just for training)
            raw_path_repair = filedict['repair']
            # Normalize path separators to forward slashes, which work cross-platform
            normalized_path_repair = raw_path_repair.replace('\\', '/')
            pc_i_np = np.load(normalized_path_repair)
            num_pc_i = pc_i_np.shape[0]
            idx_pc_i = np.random.randint(0, num_pc_i, self.num_nn)
            pc_i_np = pc_i_np[idx_pc_i, :]
            pc_c = np.concatenate((pc_np, pc_i_np), axis=0)

            if self.augment:  # Add some data augmentation (random rotation around x, y and z axis)
                import open3d as o3d

                rot_x, rot_y, rot_z = np.random.uniform(low=-10, high=10, size=3) * np.pi / 180
                pc_o3d = o3d.geometry.PointCloud()
                pc_o3d.points = o3d.utility.Vector3dVector(pc_c)
                R = pc_o3d.get_rotation_matrix_from_xyz((rot_x, rot_y, rot_z))
                pc_o3d = pc_o3d.rotate(R, center=pc_o3d.get_center())

                pc_c = np.asarray(pc_o3d.points)

        # During evaluation
        else:
            pc_c = pc_np

        # Normalize point cloud
        if self.norm_mode == 'shape_unit':
            shift = pc_c.mean(axis=0).reshape(1, 3)
            scale = pc_c.flatten().std().reshape(1, 1)

        elif self.norm_mode == 'shape_bbox':
            pc_max = pc_c.max(axis=0)
            pc_min = pc_c.min(axis=0)
            shift = ((pc_min + pc_max) / 2).reshape(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
            scale = scale / 3  # Adjust scale

        pc_c = (pc_c - shift) / scale

        pc = th.from_numpy(pc_c).float()

        out = {
            'idx': file,
            'train_points': pc.float(),
            'shift': shift,
            'scale': scale,
            'name': name
        }

        return out

    def __len__(self):
        return len(self.database)
