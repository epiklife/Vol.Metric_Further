import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from utils import get_bbox3d_for_blenderobj

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_own_data(basedir, half_res=False, testskip=1):
    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)

    imgs_path = []
    imgs = []
    poses = []
    counts = [0]
    
    # basedir = os.getcwd()
            
    for frame in meta['frames'][::1]:
        fname = os.path.join(basedir, frame['file_path'])
        imgs_path.append(fname)
        imgs.append(imageio.imread(fname))
        poses.append(np.array(frame['transform_matrix']))
    imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    N = imgs.shape[0] # How many imgs
    counts.append(counts[-1] + imgs.shape[0])

    # Train, Test세트의 개수 계산
    num_train = int(N * 0.8)
    num_test = int(N * 0.2)

    # 이미지 인덱스 생성
    shuffled_indices = np.arange(N)
    np.random.seed(52142) 
    np.random.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:num_train]
    test_indices = shuffled_indices[num_train:]

    counts = [0, num_train, num_train + num_test, N]
    i_split = [train_indices, test_indices, test_indices]
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

    bounding_box = get_bbox3d_for_blenderobj(meta, H, W, near=2.0, far=6.0)
        
    return imgs, poses, render_poses, [H, W, focal], i_split, bounding_box, imgs_path
