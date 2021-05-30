import argparse
import trimesh
import time
import numpy as np
import os
import shutil
import glob
import sys
from multiprocessing import Pool
from functools import partial
sys.path.append('../')
#from im2mesh.utils.libmesh import check_mesh_contains
import h5py
#from sklearn.neighbors import KDTree
import trimesh
from trimesh.sample import sample_surface
from scipy.spatial import cKDTree
from tqdm import tqdm
import logging
logging.getLogger("trimesh").setLevel(9000)
from sklearn.neighbors import KDTree
import math
from im2mesh.utils.io import export_pointcloud
import random

parser = argparse.ArgumentParser('Sample a watertight mesh.')
parser.add_argument('in_folder', type=str,
                    help='Path to input watertight meshes.')
parser.add_argument('--ext', type=str, default='obj',
                    help='Extensions for meshes.')
parser.add_argument('--n_proc', type=int, default=0,
                    help='Number of processes to use.')

parser.add_argument('--resize', action='store_true',
                    help='When active, resizes the mesh to bounding box.')
parser.add_argument('--bbox_padding', type=float, default=0.,
                    help='Padding for bounding box')
parser.add_argument('--pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--pointcloud_size', type=int, default=250000,
                    help='Size of point cloud.')
parser.add_argument('--partial_pointcloud_folder', type=str,
                    help='Output path for point cloud.')
parser.add_argument('--partial_pointcloud_size', type=int, default=30000,
                    help='Size of point cloud.')
parser.add_argument('--proportion', type=float, default=0.80,
                    help='proportion')

parser.add_argument('--voxels_folder', type=str,
                    help='Output path for voxelization.')
parser.add_argument('--voxels_res', type=int, default=32,
                    help='Resolution for voxelization.')
parser.add_argument('--points_folder', type=str,
                    help='Output path for points.')
parser.add_argument('--points_size', type=int, default=250000,
                    help='Size of points.')
parser.add_argument('--points_uniform_ratio', type=float, default=1.,
                    help='Ratio of points to sample uniformly'
                         'in bounding box.')
parser.add_argument('--points_sigma', type=float, default=0.01,
                    help='Standard deviation of gaussian noise added to points'
                         'samples on the surfaces.')
parser.add_argument('--points_padding', type=float, default=0.1,
                    help='Additional padding applied to the uniformly'
                         'sampled points on both sides (in total).')
parser.add_argument('--mesh_folder', type=str,
                    help='Output path for mesh.')
parser.add_argument('--overwrite', action='store_true',
                    help='Whether to overwrite output.')
parser.add_argument('--float16', action='store_true',
                    help='Whether to use half precision.')
parser.add_argument('--packbits', action='store_true',
                    help='Whether to save truth values as bit array.')
parser.add_argument('--start', type=float, default=0.,
                    help='Starting proportion')
parser.add_argument('--end', type=float, default=1.,
                    help='Ending proportion')
## hole radius
parser.add_argument('--radius', type=float, default=0.15, help='create holes')

lst_dir = './data/Humans/D-FAUST'
train_lst = open(os.path.join(lst_dir, 'train.lst'), 'r').readlines()
val_lst = open(os.path.join(lst_dir, 'val.lst'), 'r').readlines()
test_lst = open(os.path.join(lst_dir, 'test.lst'), 'r').readlines()
test_individual_lst = open(os.path.join(lst_dir, 'test_new_individual.lst'), 'r').readlines()
train_lst = [f.strip() for f in train_lst]
val_lst = [f.strip() for f in val_lst]
test_lst = [f.strip() for f in test_lst]
test_lst2 = [f.strip() for f in test_individual_lst]

def main(args):
    seq_folders = os.listdir(os.path.join(args.in_folder))
    #seq_folders = [folder for folder in seq_folders if folder in train_lst or folder in val_lst or folder in test_lst]
    #seq_folders = [folder for folder in seq_folders if folder in test_lst2]
    seq_folders = [folder for folder in seq_folders if folder in train_lst or folder in val_lst or folder in test_lst or folder in test_lst2]
    print('after select:', len(seq_folders))

    seq_folders = [os.path.join(args.in_folder, folder)
                   for folder in seq_folders]
    seq_folders = [folder for folder in seq_folders if os.path.isdir(folder)]
    # select train val test
    start = int(args.start * len(seq_folders))
    end = int(args.end * len(seq_folders))
    seq_folders = seq_folders[start:end]
    print('after slice:', len(seq_folders))

    if args.n_proc != 0:
        with Pool(args.n_proc) as p:
            p.map(partial(process_path, args=args), seq_folders)
    else:
        for p in seq_folders:
            process_path(p, args)

def process_path(in_path, args):
    modelname = os.path.basename(in_path)
    model_files = glob.glob(os.path.join(in_path, '*.%s' % args.ext))

    # Export various modalities
    export_incomplete_pointcloud(modelname, model_files, args)


def get_loc_scale(mesh, args):
    # Determine bounding box
    if not args.resize:
        # Standard bounding boux
        loc = np.zeros(3)
        scale = 1.
    else:
        bbox = mesh.bounding_box.bounds
        # Compute location and scale
        loc = (bbox[0] + bbox[1]) / 2
        scale = (bbox[1] - bbox[0]).max() / (1 - args.bbox_padding)

    return loc, scale

# Export functions
def export_incomplete_pointcloud(modelname, model_files, args):
    ####
    out_pointcloud_folder = os.path.join('./data/Humans/D-FAUST', modelname, 'pcl_seq')
    out_partial_folder = os.path.join('./data/Humans/D-FAUST', modelname, 'incompcl_seq')
    if os.path.exists(out_partial_folder):
        print('Partial Pointcloud already exist: %s' % out_partial_folder)
    else:
        os.makedirs(out_partial_folder)

    partial_size = args.partial_pointcloud_size
    for it, model_file in enumerate(model_files):
        out_pointcloud_file = os.path.join(out_pointcloud_folder, '%08d.npz' % it)
        out_partial_file = os.path.join(out_partial_folder, '%08d.npz' % it)

        t0 = time.time()
        pointcloud_dict = np.load(out_pointcloud_file)
        print('Read point cloud: %s', out_pointcloud_file)
        points, loc, scale = pointcloud_dict['points'], pointcloud_dict['loc'], pointcloud_dict['scale']
        print('Writing partial pointcloud: %s' % out_partial_file)
        num_points = points.shape[0]

        ## mask
        random.Random(100).shuffle(points)
        seed1, seed2, seed3, seed4, seed5 = points[0, :], points[int(num_points/4), :], points[int(num_points/2), :], \
            points[int(3*num_points/4), :], points[num_points-1, :]
        tree = KDTree(points)
        ind1 = tree.query_radius(seed1[None], r=args.radius)[0]
        ind2 = tree.query_radius(seed2[None], r=args.radius)[0]
        ind3 = tree.query_radius(seed3[None], r=args.radius)[0]
        ind4 = tree.query_radius(seed4[None], r=args.radius)[0]
        #ind5 = tree.query_radius(seed5[None], r=args.radius)[0]
        mask_ind = set(list(ind1)) | set(list(ind2)) | set(list(ind3)) | set(list(ind4)) #| set(list(ind5))
        remain_ind = set(np.arange(num_points)) - mask_ind
        remain_ind = list(remain_ind)

        # Compress
        if args.float16:
            dtype = np.float16
        else:
            dtype = np.float32

        points = points[remain_ind, :].astype(dtype)
        select_idx = np.random.randint(points.shape[0], size=partial_size)
        points = points[select_idx]
        loc = loc.astype(dtype)
        scale = scale.astype(dtype)

        np.savez(out_partial_file, points=points, loc=loc, scale=scale)
        print(out_partial_file, points.shape, 'has written!!!', 'time:', time.time()-t0)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
