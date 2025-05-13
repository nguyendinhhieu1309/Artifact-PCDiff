import multiprocessing
from tqdm import tqdm
import numpy as np
import subprocess 
import argparse
import mcubes
import shutil
import os

parser = argparse.ArgumentParser()
parser.add_argument('--multiprocessing', type=eval, default=True, help="set multiprocessing True/False")
parser.add_argument('--forced', type=eval, default=False, help='Override created breaks')
parser.add_argument('--threads', type=int, default=14, help="define number of threads")
parser.add_argument('--dataset', type=str, default='', help='directory housing a dataset complete objects .npy files') 
parser.add_argument('--breakage', type=float, default=0.20, help='percentage of the bottom body to be broken off completely')
parser.add_argument('--variance', type=float, default=0.1, help='random factor by which to modify breakage height, such that breakage_height = breakage * (1 +- variance*random(0,1) )')
parser.add_argument('--voxsize', type=int, default=256, help='length of the cubic space for meshes')
parser.add_argument('--maxx', type=float, default=0, help='maximum rotation appliable to the objects in the X axis in both directions (in degrees)')
parser.add_argument('--maxy', type=float, default=0, help='maximum rotation appliable to the objects in the Y axis in both directions (in degrees)')
parser.add_argument('--maxz', type=float, default=0, help='maximum rotation appliable to the objects in the Z axis in both directions (in degrees)')
parser.add_argument('--seed', type=int, default=1234, help='sets numpy random seed')
parser.add_argument('--n_breaks', type=int, default=1, help='how many broken objects per object')
parser.add_argument('--exclude_aug', type=str, default='', help='A string pattern to match to file names to exclude them from augmentation')
opt = parser.parse_args()
breakage = opt.breakage
variance = opt.variance
dataset = opt.dataset
voxsize = opt.voxsize
exclude_aug = opt.exclude_aug
grid_range = [(0,voxsize) for _ in range(3)]
downward = np.array([0,-1,0])
np.random.seed(opt.seed)

def rotX(theta):
    "Theta in radians"
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), -np.sin(theta)],
                     [0, np.sin(theta), np.cos(theta)]])

def rotY(theta):
    "Theta in radians"
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def rotZ(theta):
    "Theta in radians"
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                     [np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])

def make_rotation_matrix(points, rotation_number):
    """Uses points' data for random rotation seed. Returns points, randomState and rotation matrix"""
    theta_x, theta_y, theta_z = opt.maxx, opt.maxy, opt.maxz
    thetas = [th * np.pi /180 for th in [theta_x, theta_y, theta_z]]
    identity = np.identity(3) 
    # np.random.random doesn't work randomly through threads
    #   solution: make a random state taking a coordinate sum from the 
    #             coordinates of a point of the object shape
    rs = np.random.RandomState(int(sum( points[points.shape[0]//2], rotation_number)% (1<<30) ))
    for th, rotation in zip(thetas, [rotX, rotY, rotZ]):
        if not th:
            continue
        applied_th = (1 - 2* rs.random()) * th
        identity = rotate(identity, rotation(applied_th))
    return points, rs, identity

def rotate(points, rotation_mat):
    return np.dot(points, rotation_mat)


def volume_from_points(points):
    max_val = np.max(points)
    points = (points / max_val) * voxsize * 0.975
    volume, _ = np.histogramdd(points, bins=voxsize, range=grid_range)
    return volume

def remove_bottom_points(point_cloud, rand_state=np.random.RandomState(0)):
    y_min = np.min(point_cloud[:, 1]) 
    y_max = np.max(point_cloud[:, 1])
    
    full_breakoff = y_min + (y_max - y_min) * breakage * (1 + variance * (1 - 2 * rand_state.random()) ) 
    broken, piece = point_cloud[point_cloud[:, 1] > full_breakoff], point_cloud[point_cloud[:, 1] <= full_breakoff]
    return broken, piece

def break_piece(npy_path, break_number):
    pc = np.load(npy_path)
    pc, rs, rot_mat = make_rotation_matrix(pc, break_number)
    pc = rotate(pc, rot_mat)
    broken_cloud, piece_cloud = remove_bottom_points(pc, rs)
    broken_cloud = rotate(broken_cloud, rot_mat.T)
    piece_cloud = rotate(piece_cloud, rot_mat.T)
    return broken_cloud, piece_cloud

def break_and_save(npy_path, n):
    complete_npy_path = npy_path.replace('collection', 'complete')
    broken_npy_path = npy_path.replace('collection', 'broken')
    repair_npy_path = npy_path.replace('collection', 'repair')

    os.makedirs(os.path.dirname(complete_npy_path), exist_ok=True)
    os.makedirs(os.path.dirname(broken_npy_path), exist_ok=True)
    os.makedirs(os.path.dirname(repair_npy_path), exist_ok=True)

    if exclude_aug:
        file = npy_path.split('/')[-1]
        if exclude_aug in file:
            n = 1
    for i in range(n):
        broken, piece = break_piece(npy_path, i)
        
        _, ext = os.path.splitext(npy_path)

        serial_name = f"_{i}{ext}" if not exclude_aug else ext
        broken_npy_path_i = broken_npy_path.replace(ext, serial_name)
        repair_npy_path_i = repair_npy_path.replace(ext, serial_name)
        complete_npy_path_i = complete_npy_path.replace(ext, serial_name)
        np.save(broken_npy_path_i, broken)
        np.save(repair_npy_path_i, piece)
        shutil.copy(npy_path, complete_npy_path_i)
        
        save_obj(broken_npy_path_i, broken)
        save_obj(repair_npy_path_i, piece)


def save_obj(npy_path, points):
    if len(points.shape) != 3:
        if points.shape[0] == 3:
            points = points.T
        if points.shape[1] == 3:
            points = volume_from_points(points)
    
    if len(points.shape) != 3:
        raise ValueError(f"Check array dimensions, should be (x,x,x). Shapes (3,x) and (x,3) are supported but this is {points.shape}")
    
    v, f = mcubes.marching_cubes(points, 0)
    obj_path = npy_path.replace('.npy','.obj')

    mcubes.export_obj(v, f, obj_path)
    
def parallel_break_and_save(args):
    return break_and_save(args[0], args[1])


if __name__ == '__main__':
    if not dataset:
        print('Please provide a dataset directory')
        exit()

    # ---- Mass processing ----
    data_dir = ''
    threads = opt.threads
    override = opt.forced
    n_breaks = opt.n_breaks
    parallel = opt.multiprocessing
    directory = os.path.join(data_dir, dataset, 'collection')
    counter = 0
    database = []
    for root, dirs, files in os.walk(directory, topdown=False):
        repair_dir = root.replace('collection', 'repair')
        for filename in files:
            if filename.endswith('.npy'):
                counter += 1
                # if enough ground truth samples exist, this object has already been processed
                if not os.path.exists(os.path.join(repair_dir, f'{filename[:-4]}_{n_breaks}.npy')) or override:
                    datapoint = ''
                    datapoint = os.path.join(root, filename)
                    database.append(datapoint)
    print(f"Found {len(database) }/{counter} files unprocessed")
    

    if parallel:
        n_breaks = [n_breaks] * len(database)
        pool = multiprocessing.Pool(threads)
        try:
            for _ in tqdm(pool.imap_unordered(parallel_break_and_save, zip(database, n_breaks)), total=len(database)):
                pass
        except KeyboardInterrupt:
            exit()
        pool.close()
    else:
        for data in tqdm(database):
            break_and_save(data, n_breaks, total=len(database))

    dataset_dir = os.path.join(data_dir, dataset) if '~' not in dataset else dataset
    for file in [dataset, 'train', 'test']:
        csv_record = os.path.join(dataset_dir, f'__collection_{file}.csv')
        objects = []
        with open(csv_record, 'r') as f:
                line = f.readline()
                while line:
                    objects.append(line)
                    line = f.readline()
        with open(csv_record.replace('__collection_', ''), 'w') as f:
            for line in objects:
                path = line.replace('.npy\n', '')
                path = path.replace('/collection/', '/complete/')
                for i in range(opt.n_breaks):
                    f.write(f'{path}_{i}.npy\n')
