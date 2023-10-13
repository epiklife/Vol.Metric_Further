from glob import glob
import os
from pathlib import Path, PurePosixPath

import numpy as np 
import json
import sys
import math
import cv2
import os
import shutil
from datetime import datetime

def do_system(arg):
    err = os.system(arg)
    if err:
        # comand error
        print("")
        sys.exit(err)

#First step, Run ffmpeg
def ffmpeg(video_path, run_now):
    datas = f"datas_{run_now}"

    VIDEO_PATH = "\"" + video_path + "\""
    OUTPUT_PATH = "\"" + datas + "\""
    fps = 3.0
    do_system(f"mkdir {datas}")
    do_system(f"mkdir {datas}/images")

    do_system(f"ffmpeg -i {VIDEO_PATH} -qscale:v 1 -qmin 1 -vf \"fps={fps}\" {datas}/images/%04d.jpg")

    return datas

def colmap(data_path):
    camera_model = 'OPENCV'
    camera_params = ''
    image_path = "\"" + "./"+ data_path + '/images'+ "\""
    db_PATH = "\"" + "./" + data_path + "/colmap.db" + "\""
    sparse_PATH = "\"" + "./" + data_path + "/colmap_sparse" + "\""    
    do_system(f"colmap feature_extractor --ImageReader.camera_model {camera_model} --ImageReader.camera_params '' --SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 --database_path {db_PATH} --image_path {image_path}")
    do_system(f"colmap sequential_matcher --SiftMatching.guided_matching=true --database_path {db_PATH}")
    do_system(f"mkdir {sparse_PATH}")
    do_system(f"colmap mapper --database_path {db_PATH} --image_path {image_path} --output_path {sparse_PATH}")
    
    sparse_PATH = "\"" + "./" + data_path + "/colmap_sparse/0" + "\""
    text_PATH = "\"" + "./" + data_path + "/text" + "\""
    do_system(f"colmap bundle_adjuster --input_path {sparse_PATH} --output_path {sparse_PATH} --BundleAdjustment.refine_principal_point 1")
    do_system(f"mkdir {text_PATH}")
    do_system(f"colmap model_converter --input_path {sparse_PATH} --output_path {text_PATH} --output_type TXT")
    
    return f"{image_path}"

def json_run(TEXT_FOLDER, AABB_SCALE=32, SKIP_EARLY=0, keep_colmap_coords=False):
    with open(os.path.join(TEXT_FOLDER,"text", "cameras.txt"), "r") as f:
        camera_angle_x = math.pi / 2
        cameras = {}
        
        for line in f:
            if line[0] == "#":
                continue
            els = line.split(" ")
            camera = {}
            camera_id = int(els[0])
            camera["w"] = float(els[2])
            camera["h"] = float(els[3])
            camera["fl_x"] = float(els[4])
            camera["fl_y"] = float(els[4])
            camera["k1"] = 0
            camera["k2"] = 0
            camera["k3"] = 0
            camera["k4"] = 0
            camera["p1"] = 0
            camera["p2"] = 0
            camera["cx"] = camera["w"] / 2
            camera["cy"] = camera["h"] / 2
            camera["is_fisheye"] = False
            
            if els[1] == "SIMPLE_PINHOLE":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
            elif els[1] == "PINHOLE":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
            elif els[1] == "SIMPLE_RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL":
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV":
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["p1"] = float(els[10])
                camera["p2"] = float(els[11])
            elif els[1] == "SIMPLE_RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
            elif els[1] == "RADIAL_FISHEYE":
                camera["is_fisheye"] = True
                camera["cx"] = float(els[5])
                camera["cy"] = float(els[6])
                camera["k1"] = float(els[7])
                camera["k2"] = float(els[8])
            elif els[1] == "OPENCV_FISHEYE":
                camera["is_fisheye"] = True
                camera["fl_y"] = float(els[5])
                camera["cx"] = float(els[6])
                camera["cy"] = float(els[7])
                camera["k1"] = float(els[8])
                camera["k2"] = float(els[9])
                camera["k3"] = float(els[10])
                camera["k4"] = float(els[11])
            else:
                print("Unknown camera model ", els[1])
            camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
            camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
            camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
            camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi
            print(f"camera {camera_id}:\n\tres={camera['w'],camera['h']}\n\tcenter={camera['cx'],camera['cy']}\n\tfocal={camera['fl_x'],camera['fl_y']}\n\tfov={camera['fovx'],camera['fovy']}\n\tk={camera['k1'],camera['k2']} p={camera['p1'],camera['p2']} ")
            cameras[camera_id] = camera

    if len(cameras) == 0:
        print("No cameras found!")
        sys.exit(1)

    with open(os.path.join(TEXT_FOLDER, "text", "images.txt"), "r") as f:
        i = 0
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        if len(cameras) == 1:
            camera = cameras[camera_id]
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": AABB_SCALE,
                "frames": [],
            }
        else:
            out = {
                "frames": [],
                "aabb_scale": AABB_SCALE
            }

        up = np.zeros(3)
        for line in f:
            line = line.strip()
            if line[0] == "#":
                continue
            i = i + 1
            if i < SKIP_EARLY * 2:
                continue
            if i % 2 == 1:
                elems = line.split(" ")
                image_rel = os.path.relpath(TEXT_FOLDER)
                name = str(f"./{image_rel}/images/{'_'.join(elems[9:])}")
                print(name)
                b = sharpness(name)
                print(name, "sharpness=", b)
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                R = qvec2rotmat(-qvec)
                t = tvec.reshape([3, 1])
                m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
                c2w = np.linalg.inv(m)
                if not keep_colmap_coords:
                    c2w[0:3, 2] *= -1
                    c2w[0:3, 1] *= -1
                    c2w = c2w[[1, 0, 2, 3], :]
                    c2w[2, :] *= -1

                    up += c2w[0:3, 1]

                frame = {"file_path": f"./images/{'_'.join(elems[9:])}", "sharpness": b, "transform_matrix": c2w}
                if len(cameras) != 1:
                    frame.update(cameras[int(elems[8])])
                out["frames"].append(frame)

        nframes = len(out["frames"])

        if keep_colmap_coords:
            flip_mat = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ])

            for f in out["frames"]:
                f["transform_matrix"] = np.matmul(f["transform_matrix"], flip_mat)
        else:
            up = up / np.linalg.norm(up)
            print("up vector was", up)
            R = rotmat(up, [0, 0, 1])
            R = np.pad(R, [0, 1])
            R[-1, -1] = 1

            for f in out["frames"]:
                f["transform_matrix"] = np.matmul(R, f["transform_matrix"])

            print("computing center of attention...")
            totw = 0.0
            totp = np.array([0.0, 0.0, 0.0])
            for f in out["frames"]:
                mf = f["transform_matrix"][0:3, :]
                for g in out["frames"]:
                    mg = g["transform_matrix"][0:3, :]
                    p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                    if w > 0.00001:
                        totp += p * w
                        totw += w
            if totw > 0.0:
                totp /= totw
            print(totp)
            for f in out["frames"]:
                f["transform_matrix"][0:3, 3] -= totp

            avglen = 0.
            for f in out["frames"]:
                avglen += np.linalg.norm(f["transform_matrix"][0:3, 3])
            avglen /= nframes
            print("avg camera distance from origin", avglen)
            for f in out["frames"]:
                f["transform_matrix"][0:3, 3] *= 4.0 / avglen

        for f in out["frames"]:
            f["transform_matrix"] = f["transform_matrix"].tolist()
        print(nframes, "frames")
        print(f"writing transforms.json")
        with open(f"{TEXT_FOLDER}/transforms.json", "w") as outfile:
            json.dump(out, outfile, indent=2)

    return f"{TEXT_FOLDER}"

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	fm = variance_of_laplacian(gray)
	return fm

def qvec2rotmat(qvec):
	return np.array([
		[
			1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
			2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
			2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]
		], [
			2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
			1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
			2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]
		], [
			2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
			2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
			1 - 2 * qvec[1]**2 - 2 * qvec[2]**2
		]
	])

def rotmat(a, b):
	a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
	v = np.cross(a, b)
	c = np.dot(a, b)
	# handle exception for the opposite direction input
	if c < -1 + 1e-10:
		return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
	s = np.linalg.norm(v)
	kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
	return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2 + 1e-10))

def closest_point_2_lines(oa, da, ob, db): # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
	da = da / np.linalg.norm(da)
	db = db / np.linalg.norm(db)
	c = np.cross(da, db)
	denom = np.linalg.norm(c)**2
	t = ob - oa
	ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
	tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
	if ta > 0:
		ta = 0
	if tb > 0:
		tb = 0
	return (oa+ta*da+ob+tb*db) * 0.5, denom


    
    
def run(video_path):
	run_now = int(datetime.now().strftime("%y%m%d%H%M%S"))
	
	image_path = ffmpeg(video_path, run_now)
	#print(f"./{image_path}")
	colmap(image_path)
    
	return json_run(image_path)
    
