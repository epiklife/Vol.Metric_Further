import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2

from utils import get_bbox3d_for_llff, get_bbox3d_for_blenderobj

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

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def _make_image(basedir, imgsname, factors=[], resolutions=[]):
    
    # Return if folder already exists
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
            
    if not needtoload:
        return

    sh = cv2.imread(imgsname[0]).shape
    
    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            H = sh[0]//r
            W = sh[1]//r 
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            H = r[0]
            W = r[1]
            
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue
        
        def imread(f):
            if f.endswith('png'):
                return cv2.imread(f)# , format="PNG-PIL", ignoregamma=True
            else:
                return cv2.imread(f)
        
        os.makedirs(imgdir)
        for imgname in imgsname:
            img = imread(imgname)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            imgname = os.path.join(imgdir,imgname[-8:])
            cv2.imwrite(imgname,img)
    
def load_ingp_data(basedir, factor=1, width=None, height=None):

    with open(os.path.join(basedir, 'transforms.json'), 'r') as fp:
        meta = json.load(fp)
    
    poses = []
    imgfiles = []
    i=0
    for frame in meta['frames']:
        poses.append(np.array(frame['transform_matrix']))
        fname = frame['file_path']
        if fname.endswith('JPG') or fname.endswith('jpg') or fname.endswith('png'):
            imgfiles.append(os.path.join(basedir, fname))
        else:
            imgfiles.append(os.path.join(basedir, fname+'.png'))

    sh = imageio.imread(imgfiles[0]).shape

    sfx = ''
    if factor != 1:
        sfx = '_{}'.format(factor)
        _make_image(basedir,imgfiles, factors=[factor])
        factor = factor        
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _make_image(basedir,imgfiles, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _make_image(basedir,imgfiles, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)

    imgfiles = []
    for frame in meta['frames']:
        fname = frame['file_path'].split('/')[-1]

        if fname.endswith('JPG') or fname.endswith('jpg') or fname.endswith('png'):
            imgfiles.append(os.path.join(imgdir, fname))
        else:
            imgfiles.append(os.path.join(imgdir, fname+'.png'))
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, format="PNG-PIL", ignoregamma=True)
        else:
            return imageio.imread(f)
    
    
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    imgs = np.stack(imgs, -1)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, imgs.shape)
    
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    # bounding_box_llff = get_bbox3d_for_llff(poses[:,:3,:4], [H, W, focal], near=0.0, far=1.0)
    bounding_box = get_bbox3d_for_blenderobj(meta, H, W, near=2.0, far=6.0)
    print('bounding_box_llff',get_bbox3d_for_llff(poses[:,:3,:4], [H, W, focal], near=0.0, far=1.0))
    print('bounding_box_blender',bounding_box)
    
    return imgs, poses, render_poses, [H, W, focal], i_test, bounding_box#, i_split