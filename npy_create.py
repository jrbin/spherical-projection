# @author:zzb
from __future__ import division

import os
import threading

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import savemat
from scipy.spatial import Delaunay
from scipy.misc import imsave

from util.kitti_util import Calibration
from util.parseTrackletXML import parseXML

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CLASS = {'Car': 1, 'Van': 1, 'Pedestrian': 2, 'Cyclist': 3}
g_pixel_num = {'Car': 0, 'Van': 0, 'Pedestrian': 0, 'Cyclist': 0}
g_instance_num = {'Car': 0, 'Van': 0, 'Pedestrian': 0, 'Cyclist': 0}


DEBUG_MODE = False
# DEBUG_MODE = True
COLOR_MAP = np.array([[0.00,  0.00,  0.00],
                      [0.99,  0.0,  0.0],
                      [0.0,  0.0,  0.99],
                      [0.0,  0.99,  0.0]])

class KittiRaw(object):

    def __init__(self, folder_path):
        self.base_name = os.path.basename(folder_path)
        self.drive_path = folder_path
        self.image_dir = os.path.join(self.drive_path, 'image_02', 'data')
        self.lidar_dir = os.path.join(self.drive_path, 'velodyne_points', 'data')
        self.calib_dir = os.path.join(self.drive_path, 'calib')
        self.num_samples = [idx.rstrip('.bin') for idx in os.listdir(self.lidar_dir)]
        self.parse_xml()

    def __len__(self):
        return len(self.num_samples)

    def get_image(self, idx):
        '''return opencv-format image'''
        img_filename = os.path.join(self.image_dir, idx + '.png')
        img = cv2.imread(img_filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_calibration(self):
        return Calibration(self.calib_dir)

    def get_lidar(self, idx):
        lidar_filename = os.path.join(self.lidar_dir, idx+'.bin')
        scan = np.fromfile(lidar_filename, dtype=np.float32)
        scan = scan.reshape((-1, 4))    # x,y,z,intensity
        return scan

    def get_lable_objects(self, idx):
        if int(idx) in self.objects.keys():
            return self.objects[int(idx)]
        else:
            return None

    def parse_xml(self):
        '''ref: (http://cvlibs.net/datasets/kitti/raw_data.php)
        '''
        tracklets = parseXML(os.path.join(
            self.drive_path, 'tracklet_labels.xml'))
        # loop over tracklets
        self.objects = {}
        for iTracklet, tracklet in enumerate(tracklets):
            # this part is inspired by kitti object development kit matlab code: computeBox3D
            h, w, l = tracklet.size
            object_type = tracklet.objectType
            # only deal with car,van,pedestrian,cyclist
            if object_type not in ('Car', 'Van', 'Pedestrian', 'Cyclist'):
                continue
            trackletBox = np.array([  # in velodyne coordinates around zero point and without orientation yet\
                [-l/2, -l/2,  l/2, l/2, -l/2, -l/2,  l/2, l/2], \
                [w/2, -w/2, -w/2, w/2,  w/2, -w/2, -w/2, w/2], \
                [0.0,  0.0,  0.0, 0.0,    h,     h,   h,   h]])

            # loop over all data in tracklet
            for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber \
                    in tracklet.__iter__():
                # determine if object is in the image; otherwise continue
                if truncation not in (0, 1):
                    # assert(False)
                    continue

                # re-create 3D bounding box in velodyne coordinate system
                # other rotations are 0 in all xml files I checked
                yaw = rotation[2]
                assert np.abs(rotation[:2]).sum(
                ) == 0, 'object rotations other than yaw given!'
                rotMat = np.array([
                    [np.cos(yaw), -np.sin(yaw), 0.0],
                    [np.sin(yaw),  np.cos(yaw), 0.0],
                    [0.0,          0.0, 1.0]])
                cornerPosInVelo = np.dot(
                    rotMat, trackletBox) + np.tile(translation, (8, 1)).T
                cornerPosInVelo = cornerPosInVelo.T

                if absoluteFrameNumber in self.objects.keys():
                    self.objects[absoluteFrameNumber].append(
                        [object_type, cornerPosInVelo])
                else:
                    self.objects[absoluteFrameNumber] = [
                        [object_type, cornerPosInVelo]]



def show_lidar_on_image(pc_velo, img, calib):
    ''' Project LiDAR points to image '''
    img_height, img_width, img_channel = img.shape
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
                                                              calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds, :]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0/depth), :]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i, 0])),
                         int(np.round(imgfov_pts_2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show()
    return img


def add_depth(pc):
    '''Add depth(range) attribute'''
    depth = np.sqrt(pc[:, 0] ** 2 + pc[:, 1] ** 2, pc[:, 2] ** 2)
    return np.concatenate((pc, depth[:, np.newaxis]), axis=1)


def add_label(pc, labels, count_pixel_num=False):
    '''Add label of each point'''
    label = np.zeros((pc.shape[0], 1), dtype=pc.dtype)
    for obj_type, box3d in labels:
        pc_box_ind = extract_pc_in_box3d(pc, box3d)
        label[pc_box_ind] = CLASS[obj_type]
        if count_pixel_num:
            g_pixel_num[obj_type] += np.sum(pc_box_ind)
            g_instance_num[obj_type] += 1
    # print(g_pixel_num)
    # print(g_instance_num)
    return np.concatenate((pc, label), axis=1)


def extract_pc_in_box3d(pc, box3d):
    ''' pc: (N,3), box3d: (8,3) '''
    def in_hull(p, hull):
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
            return hull.find_simplex(p) >= 0
    box3d_roi_inds = in_hull(pc[:, 0:3], box3d)
    return box3d_roi_inds


def add_rgb(pc, image, calib, clip_distance=[0, 500], image_fov_pc=False):
    '''Add rgb information from image'''
    img_height, img_width, img_channel = image.shape
    if image.max() > 1:
        image = image / 255.0
    else:
        image = image.astype(pc.dtype)
    rgb = np.zeros((pc.shape[0], 3))

    _, pts_2d, fov_inds = get_lidar_in_image_fov(
        pc[:, 0:3], calib, 0, 0, img_width, img_height, True, clip_distance=clip_distance)

    for ind, val in enumerate(fov_inds):
        if val:
            rgb[ind] = image[int(pts_2d[ind][1]), int(pts_2d[ind][0]), :]
        else:
            rgb[ind] = np.array([0, 0, 0], dtype=pc.dtype)
    pc = np.concatenate((pc, rgb), axis=1)
    if image_fov_pc:
        return pc[fov_inds, :]
    else:
        return pc


def get_lidar_in_fov_90(pc, fov=(-45, 45), clip_distance=0.0, max_distance=500):
    ''' Filter lidar points, keep those in image FOV '''
    fov_inds = (pc[:, 0] > pc[:, 1]) & (pc[:, 0] > -pc[:, 1])
    fov_inds = fov_inds & (pc[:, 0] > clip_distance) & (
        pc[:, 0] < max_distance)
    return pc[fov_inds, :]


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=[0, 120]):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
        (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance[0]) & (
        pc_velo[:, 0] < clip_distance[1])
    imgfov_pc_velo = pc_velo[fov_inds, :]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def channel_plot(img, c, action=1, rgb=False):
    '''plot channel
    Args:
        img: input image, defaut channel: x,y,z,intensity,depth,label,R,G,B,count_mask
        c: iist, the channel of image to plot
        action: int, 1: image size is 64x512, 2: image size is 375x1242
        rgb: bool, if true plot rgb-img
    '''
    assert (len(img.shape) == 3), 'input shape should be hxwxc'
    assert (c[len(l)-1] <= img.shape[2]), 'c is out of range'
    if action == 1:
        h, w = 64, 512
    else:
        h, w = 375, 1242

    if rgb:
        channel = img[:, :, c]
        if channel.max() < 255:
            channel = channel * 255 / channel.max()
        image = cv2.resize(img.astype(np.uint8), (w, h))
        Image.fromarray(image).show()
    else:
        for i in list(c):
            channel = img[:, :, i]
            if channel.max() < 255:
                channel = channel * 255 / channel.max()
            image = cv2.resize(channel.astype(np.uint8), (w, h))
            Image.fromarray(image).show()


def spherical_projection(pc, height=64, width=512):
    '''spherical projection 
    Args:
        pc: point cloud, dim: N*9
    Returns:
        pj_img: projected spherical iamges, shape: h*w*9
    '''
    pj_img = np.zeros((height, width, pc.shape[1]))
    R = np.sqrt(pc[:, 0]**2+pc[:, 1]**2)
    theta = np.arcsin(pc[:, 2]/pc[:, 4])
    phi = np.arcsin(pc[:, 1]/R)

    ## filter
    # theta = theta[(theta >= -0.4) & (theta < theta.max())]

    idx_h = height - 1 - ((height-1) * (theta - theta.min()) / (theta.max() - theta.min())).astype(np.int32)
    idx_w = width - 1 - ((width - 1) * (phi - phi.min()) / (phi.max() - phi.min())).astype(np.int32)
    
    idx_h = np.round(idx_h[:])
    idx_w = np.round(idx_w[:])

    count_mask = np.zeros((height, width, 1))
    for i in range(idx_h.shape[0]):
        pj_img[idx_h[i], idx_w[i], :] = pc[i, :]
        count_mask[idx_h[i], idx_w[i], ] += 1
    # pj_img = np.concatenate((pj_img, count_mask), 2)
    return pj_img


def knn(mask, image, ignore=None):
    img = image.copy()
    idx_not_zero = np.where(mask != 0)
    idx_not_zero_T = np.transpose(idx_not_zero)

    idx_zero = np.where(mask == 0)
    idx_zero_T = np.transpose(idx_zero)

    tri = Delaunay(idx_not_zero_T)

    idx_new = idx_zero_T[tri.find_simplex(idx_zero_T) > -1]

    for i in range(idx_new.shape[0]):
        t = np.sqrt(np.sum(np.square(idx_not_zero_T - idx_new[i]), axis=1))

        dst = idx_not_zero_T[np.argmin(t)]
        src = idx_new[i]
        img[tuple(src)] = img[tuple(dst)]
    if ignore:
        img[:, :, ignore] = image[:, :, ignore]
    return img


def process(input_folder, output_folder, img_save=False, show_img_and_lidar=False):
    '''this procedure transfer source kitti raw data into numpy format data,
        prepare for training/validation data
    Args:
        input_folder: like '2011_09_26_drive_0001_sync' folder
        output_folder: folder to save .npy file
    '''
    dataset = KittiRaw(input_folder)
    calib = dataset.get_calibration()
    base_name = dataset.base_name

    for idx in dataset.num_samples:
        print('processing: ', idx)

        output_name = base_name[:11] + base_name[17:22] + idx  # '2011_09_26_0000000001'
        output_path = os.path.join(output_folder, output_name+'.npy')

        # if os.path.isfile(output_path):
        #     continue

        pc = dataset.get_lidar(idx)  # get point cloud
        labels = dataset.get_lable_objects(idx)  # get label
        img = dataset.get_image(idx)  # get image

        if not labels:
            continue

        pc = get_lidar_in_fov_90(pc)

        # add extra attributes
        pc = add_depth(pc)
        pc = add_label(pc, labels)
        pc = add_rgb(pc, img, calib, image_fov_pc=False)

        # save point cloud with matlab format
        # savemat(output_name+'.mat',{'a':pc})

        sphere_images = spherical_projection(pc)

        np.save(output_path, sphere_images)

        if show_img_and_lidar:
            Image.fromarray(img).show()
            show_lidar_on_image(pc[:, :3], img, calib)
        if img_save:
            if not os.path.exists(os.path.join(output_folder, 'cache')):
                os.makedirs(os.path.join(output_folder, 'cache'))
            for i in [0, 1, 2, 3, 4, 5]:
                imsave(os.path.join(
                    output_folder, 'cache', output_name+'_%d.jpg' % i), sphere_images[:, :, i])
            imsave(os.path.join(
                output_folder, 'cache', output_name + '_rgb.jpg'), sphere_images[:, :, 6:9])
            print('save', output_name, '.jpg')


def single_thread(folders, output_folder):
    for folder in folders:
        process(folder, output_folder, img_save=True, show_img_and_lidar=False)
    print('done!')

def multi_thead(folders, output_folder):
    threads = []
    files = range(len(folders))
    for idx in files:
        t = threading.Thread(target=process, args=(folders[idx], output_folder))
        threads.append(t)
    for idx in files:
        threads[idx].start()
    for idx in files:
        threads[idx].join()
    print('done!')


if __name__ == '__main__':
    kitti_path = 'D:\\KittiRaw\\all'
    output_folder = 'result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    folders = [os.path.join(kitti_path, folder) for folder in os.listdir(kitti_path)]

    single_thread(folders, output_folder)
