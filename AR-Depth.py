# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# %%
import cv2
from cv2 import DISOpticalFlow
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import scipy
import scipy.sparse
import scipy.sparse.linalg
import copy
import sys

###
from tqdm import tqdm
###

# %% [markdown]
# # README
# 
# #### Dependencies: 
# - OpenCV 4.0
# - opencv-contrib (`pip install opencv-contrib-python`)
# - NumPy
# - pyquaternion (`pip install pyquaternion`)
# 
# #### First, make sure it runs on the test sequence:
# - Above, press `Cell > Run All`
# - Make sure the 'output' folder is populated with frames, and the depth maps look reasonable
#     
# #### Then, you can input your own data:
# - Replace the values of the three variables below
#   - **input_frames**: the path to the folder which contains your video frames
#   - **input_recon**: the path to the folder which contains your sparse reconstruction. For input format, we use the COLMAP format defined here (https://colmap.github.io/format.html). This folder should contain three files: `points2D.txt`, `images.txt`, and `cameras.txt`. Since the COLMAP format does not natively include information about whether or not a frame is a keyframe (since it's not originally intended for SLAM systems), we interpret this information from the POINTS2D[] in `images.txt`. That is, if a given image has a nonzero number of POINTS2D, we assume it is a keyframe. Note that the COLMAP format must be TXT and not BIN.
#   - **output_folder**: the path to the folder where you want the output saved
# 
# This Python notebook is intended as a reference implementation for research purposes. Note that while this implementation is feature-complete, it is an unoptimized Python port of the version used for experiments in the paper, so performance will be much worse than what is reported. Please refer to the paper for more accurate timing results and details on how to further optimize this implementation. Additionally, while the linear system constraints used here are the same as in the paper, the solver is not the same, and therefore the results are not guaranteed to be identical.
# %% [markdown]
# ## The parameters

# %%
input_frames = "sample_data/frames/"
input_colmap = "sample_data/reconstruction/"
output_folder = "output/"

dump_debug_images = True


# %%
# Algorithm parameters. See the paper for details.

tau_high = 0.1
tau_low = 0.1
tau_flow = 0.2
k_I = 5
k_T = 7
k_F = 31
lambda_d = 1
lambda_t = 0.01
lambda_s = 1

num_solver_iterations = 500

# %% [markdown]
# ## A simplified COLMAP importer

# %%
class Reconstruction:
    def __init__(self):
        self.cameras = {}
        self.views = {}
        self.points3d = {}
        self.min_view_id = -1
        self.max_view_id = -1
        self.image_folder = ""
    
    def ViewIds(self):
        return list(self.views.keys())
    
    def GetNeighboringKeyframes(self, view_id):
        previous_keyframe = -1
        next_keyframe = -1
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                previous_keyframe = idx
                break
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if self.views[idx].IsKeyframe():
                next_keyframe = idx
                break
        if previous_keyframe < 0 or next_keyframe < 0:
            return np.array([])
        return [previous_keyframe, next_keyframe]
    
    def GetReferenceFrames(self, view_id):
        kf = self.GetNeighboringKeyframes(view_id)
        if (len(kf) < 2):
            return []
        dist = np.linalg.norm(self.views[kf[1]].Position() -                              self.views[kf[0]].Position()) / 2
        pos = self.views[view_id].Position()
        ref = []
        for idx in range(view_id + 1, self.max_view_id):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        for idx in range(view_id - 1, self.min_view_id, -1):
            if idx not in self.views:
                continue
            if (np.linalg.norm(pos -                              self.views[idx].Position()) > dist):
                ref.append(idx)
                break
        return ref

    def GetImage(self, view_id):
        return self.views[view_id].GetImage(self.image_folder)
    
    def GetSparseDepthMap(self, frame_id):
        camera = self.cameras[self.views[frame_id].camera_id]
        view = self.views[frame_id]
        view_pos = view.Position()
        depth_map = np.zeros((camera.height, camera.width), dtype=np.float32)
        for point_id, coord in view.points2d.items():
            try:
                pos3d = self.points3d[point_id].position3d
            except KeyError:
                print(point_id, " is not in points3d")
            depth = np.linalg.norm(pos3d - view_pos)
            depth_map[int(coord[1]), int(coord[0])] = depth
        return depth_map
    
    def Print(self):
        print("Found " + str(len(self.views)) + " cameras.")
        for id in self.cameras:
            self.cameras[id].Print()
        print("Found " + str(len(self.views)) + " frames.")
        for id in self.views:
            self.views[id].Print()

class Point:
    def __init__(self):
        self.id = -1
        self.position3d = np.zeros(3, float)
    
            
class Camera:

    def __init__(self):
        self.id = -1
        self.width = 0
        self.height = 0
        self.focal = np.zeros(2,float)
        self.principal = np.zeros(2,float)
        self.model = ""
    
    def Print(self):
        print("Camera " + str(self.id))
        print("-Image size: (" + str(self.width) +             ", " + str(self.height) + ")")
        print("-Focal: " + str(self.focal))
        print("-Model: " + self.model)
        print("")

class View:    
    def __init__(self):
        self.id = -1
        self.orientation = Quaternion()
        self.translation = np.zeros(3, float)
        self.points2d = {}
        self.camera_id = -1
        self.name = ""
    
    def IsKeyframe(self):
        return len(self.points2d) > 0
    
    def Rotation(self):
        return self.orientation.rotation_matrix
    
    def Position(self):
        return self.orientation.rotate(self.translation)
    
    def GetImage(self, image_folder):
        mat = cv2.imread(image_folder + "/" + self.name)
        # Check that we loaded correctly.
        assert mat is not None,             "Image " + self.name + " was not found in "             + image_folder
        return mat
    
    def Print(self):
        print("Frame " + str(self.id) + ": " + self.name)
        print("Rotation: \n" +             str(self.Rotation()))
        print("Position: \n" +             str(self.Position()))
        print("")
        
def ReadColmapCamera(filename):
    file = open(filename, "r")
    line = file.readline()
    cameras = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            cameras[id_value] = Camera()
            cameras[id_value].id = id_value
            cameras[id_value].model = tokens[1]
            # Currently we're assuming that the camera model
            # is in the SIMPLE_RADIAL format
            assert(cameras[id_value].model == "PINHOLE")
            cameras[id_value].width = int(tokens[2])
            cameras[id_value].height = int(tokens[3])
            cameras[id_value].focal[0] = float(tokens[4])
            cameras[id_value].focal[1] = float(tokens[5])
            cameras[id_value].principal[0] = float(tokens[6])
            cameras[id_value].principal[1] = float(tokens[7])
        line = file.readline()
    return cameras

def ReadColmapImages(filename):
    file = open(filename, "r")
    line = file.readline()
    views = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            views[id_value] = View()
            views[id_value].id = id_value
            views[id_value].orientation = Quaternion(float(tokens[1]),                                                      float(tokens[2]),                                                      float(tokens[3]),                                                      float(tokens[4]))
            views[id_value].translation[0] = float(tokens[5])
            views[id_value].translation[1] = float(tokens[6])
            views[id_value].translation[2] = float(tokens[7])
            views[id_value].camera_id = int(tokens[8])
            views[id_value].name = tokens[9]
            line = file.readline()
            tokens = line.split()
            views[id_value].points2d = {}
            for idx in range(0, len(tokens) // 3):
                point_id = int(tokens[idx * 3 + 2])
                coord = np.array([float(tokens[idx * 3 + 0]),                          float(tokens[idx * 3 + 1])])
                views[id_value].points2d[point_id] = coord
            
            # Read the observations...
        line = file.readline()
    return views
           
def ReadColmapPoints(filename):
    file = open(filename, "r")
    line = file.readline()
    points = {}
    while (line):
        if (line[0] != '#'):
            tokens = line.split()
            id_value = int(tokens[0])
            points[id_value] = Point()
            points[id_value].id = id_value
            points[id_value].position3d = np.array([float(tokens[1]),                                         float(tokens[2]),                                         float(tokens[3])])
            
        line = file.readline()
    return points
        
            
    
def ReadColmap(poses_folder, images_folder):
    # Read the cameras (intrinsics)
    recon = Reconstruction()
    recon.image_folder = images_folder
    recon.cameras = ReadColmapCamera(poses_folder + "/cameras.txt")
    recon.views = ReadColmapImages(poses_folder + "/images.txt")
    recon.points3d = ReadColmapPoints(poses_folder + "/points3D.txt")
    recon.min_view_id = min(list(recon.views.keys()))
    recon.max_view_id = max(list(recon.views.keys()))
    print("Number of points: " + str(len(recon.points3d)))
    print("Number of frames: " + str(len(recon.views)))
    #assert len(recon.views) == (recon.max_view_id - recon.min_view_id) + 1, "Min\max: " + str(recon.max_view_id) + " " + str(recon.min_view_id)
    return recon

# %% [markdown]
# ## The densification code

# %%
import flow_color

dis = DISOpticalFlow.create(2)
def GetFlow(image1, image2):
    flow = np.zeros((image1.shape[0], image1.shape[1], 2), np.float32)
    flow = dis.calc(        cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY),        cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY), flow)
    return flow

def AbsoluteMaximum(images):
    assert(len(images) > 0)
    output = images[0]
    for i in range(1,len(images)):
        output[np.abs(images[i]) > np.abs(output)] = images[i][np.abs(images[i]) > np.abs(output)]
    return output

def GetImageGradient(image):
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))
    img_grad_x = AbsoluteMaximum([xr,xg,xb])
    img_grad_y = AbsoluteMaximum([yr,yg,yb])
    
    return img_grad_x, img_grad_y

def GetGradientMagnitude(img_grad_x, img_grad_y):
    img_grad_magnitude = cv2.sqrt((img_grad_x * img_grad_x)                                   + (img_grad_y * img_grad_y))
    return img_grad_magnitude

def GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y):
    x1,x2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,1,0,ksize=5))
    y1,y2 = cv2.split(cv2.Sobel(flow,cv2.CV_64F,0,1,ksize=5))
    flow_grad_x = AbsoluteMaximum([x1,x2])
    flow_grad_y = AbsoluteMaximum([y1,y2])
    flow_gradient_magnitude = cv2.sqrt((flow_grad_x * flow_grad_x)                                    + (flow_grad_y * flow_grad_y))
    reliability = np.zeros((flow.shape[0], flow.shape[1]))

    for x in tqdm(range(0, flow.shape[0]), leave=True, ascii=True, desc="by one image length"):
        for y in range(1, flow.shape[1]):
            magn = (img_grad_x[x,y] * img_grad_x[x,y]) +                 (img_grad_y[x,y] * img_grad_y[x,y])
            gradient_dir = np.array((img_grad_y[x,y], img_grad_x[x,y]))
            if (np.linalg.norm(gradient_dir) == 0):
                reliability[x,y] = 0
                continue
            gradient_dir = gradient_dir / np.linalg.norm(gradient_dir)
            center_pixel = np.array((x,y))
            p0 = center_pixel + gradient_dir
            p1 = center_pixel - gradient_dir
            if p0[0] < 0 or p1[0] < 0 or p0[1] < 0 or p1[1] < 0                 or p0[0] >= flow.shape[0] or p0[1] >= flow.shape[1] or                 p1[0] >= flow.shape[0] or p1[1] >= flow.shape[1]:
                reliability[x,y] = -1000
                continue
            f0 = flow[int(p0[0]), int(p0[1])].dot(gradient_dir)
            f1 = flow[int(p1[0]), int(p1[1])].dot(gradient_dir)
            reliability[x,y] = f1 - f0

    return flow_gradient_magnitude, reliability

def GetSoftEdges(image, flows):
    img_grad_x, img_grad_y = GetImageGradient(image)
    img_grad_magnitude = GetGradientMagnitude(img_grad_x, img_grad_y)
    if (dump_debug_images):
        plt.imsave(output_folder + "/image_gradient_" + recon.views[frame].name,                 img_grad_magnitude)
    flow_gradient_magnitude = np.zeros(img_grad_magnitude.shape)
    
    max_reliability = np.zeros(flow_gradient_magnitude.shape)
    i = 0
    for flow in tqdm(flows, leave=True, ascii=True, desc="by flows in image"):
        magnitude, reliability = GetFlowGradientMagnitude(flow, img_grad_x, img_grad_y)
        if (dump_debug_images):
            plt.imsave(output_folder + "/flow_" + str(i) + "_" + recon.views[frame].name,                     flow_color.computeImg(flow))            
            plt.imsave(output_folder + "/reliability_" + str(i) + "_" + recon.views[frame].name,                     reliability)
        flow_gradient_magnitude[reliability > max_reliability] = magnitude[reliability > max_reliability]
        i += 1
        
    if (dump_debug_images):
        plt.imsave(output_folder + "/flow_gradient_" + recon.views[frame].name,                 flow_gradient_magnitude)
    flow_gradient_magnitude =         cv2.GaussianBlur(flow_gradient_magnitude,(k_F, k_F),0)
    flow_gradient_magnitude *= img_grad_magnitude
    flow_gradient_magnitude /= flow_gradient_magnitude.max()
    return flow_gradient_magnitude
    
def Canny(soft_edges, image):
    image = cv2.GaussianBlur(image, (k_I, k_I), 0)
    xr,xg,xb = cv2.split(cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5))
    yr,yg,yb = cv2.split(cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5))
    img_gradient = cv2.merge((AbsoluteMaximum([xr,xg,xb]),AbsoluteMaximum([yr,yg,yb])))
    
    TG22 = 13573
    
    gx,gy = cv2.split(img_gradient * (2**15))
    mag = cv2.sqrt((gx * gx)                     + (gy * gy))
    seeds = []
    edges = np.zeros(mag.shape)
    for x in tqdm(range(1, img_gradient.shape[0] - 1), leave=True, ascii=True, desc="by rows canny"):
        for y in range(1, img_gradient.shape[1] - 1):
            ax = int(abs(gx[x,y]))
            ay = int(abs(gy[x,y])) << 15
            tg22x = ax * TG22
            m = mag[x,y]
            if (ay < tg22x):
                if (m > mag[x,y-1] and                   m >= mag[x,y+1]):
                    #suppressed[x,y] = m
                    if (m > tau_high and soft_edges[x,y] > tau_flow):
                        seeds.append((x,y))
                        edges[x,y] = 255
                    elif (m > tau_low):
                        edges[x,y] = 1
            else:
                tg67x = tg22x + (ax << 16)
                if (ay > tg67x):
                    if (m > mag[x+1,y] and m >= mag[x-1,y]):
                        if (m > tau_high and soft_edges[x,y] > tau_flow):
                            seeds.append((x,y))
                            edges[x,y] = 255
                        elif (m > tau_low):
                            edges[x,y] = 1
                else:
                    if (int(gx[x,y]) ^ int(gy[x,y]) < 0):
                        if (m > mag[x-1,y+1] and m >= mag[x+1,y-1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
                    else:
                        if (m > mag[x-1,y-1] and m > mag[x+1,y+1]):
                            if (m > tau_high and soft_edges[x,y] > tau_flow):
                                seeds.append((x,y))
                                edges[x,y] = 255
                            elif (m > tau_low):
                                edges[x,y] = 1
    w = img_gradient.shape[0] - 1
    h = img_gradient.shape[1] - 1
    if (dump_debug_images):
        plt.imsave(output_folder + "/edge_seeds_" + recon.views[frame].name,             edges == 255)
        plt.imsave(output_folder + "/edge_all_possible_" + recon.views[frame].name,             edges == 1)
    while len(seeds) > 0:
        (x,y) = seeds.pop()
        
        if (x < w and y < h and edges[x+1,y+1] == 1):
            edges[x+1,y+1] = 255
            seeds.append((x+1,y+1))
        if (x > 0 and y < h and edges[x-1,y+1] == 1):
            edges[x-1,y+1] = 255
            seeds.append((x-1,y+1))
        if (y < h and edges[x,y+1] == 1):
            edges[x,y+1] = 255
            seeds.append((x,y+1))
        if (x < w and y > 0 and edges[x+1,y-1] == 1):
            edges[x+1,y-1] = 255
            seeds.append((x+1,y-1))
        if (x > 0 and y > 0 and edges[x-1,y-1] == 1):
            edges[x-1,y-1] = 255
            seeds.append((x-1,y-1))
        if (y > 0 and edges[x,y-1] == 1):
            edges[x,y-1] = 255
            seeds.append((x,y-1))
        if (x < w and edges[x+1,y] == 1):
            edges[x+1,y] = 255
            seeds.append((x+1,y))
        if (x > 0 and edges[x-1,y] == 1):
            edges[x-1,y] = 255
            seeds.append((x-1,y))
    edges[edges == 1] = 0
    return edges
    
def GetInitialization(sparse_points, last_depth_map):
    initialization = sparse_points.copy()
    if last_depth_map.size > 0:
        initialization[last_depth_map > 0] = 1.0 / last_depth_map[last_depth_map > 0]
    
    w = edges.shape[0]
    h = edges.shape[1]
    last_known = -1
    first_known = -1
    for col in tqdm(range(0,w), leave=True, ascii=True, desc="by rows initialization"):
        for row in range(0,h):
            if (sparse_points[col, row] > 0):
                last_known = 1.0 / sparse_points[col, row]
            elif (initialization[col, row] > 0):
                last_known = initialization[col, row]
            if (first_known < 0):
                first_known = last_known
            initialization[col, row] = last_known
    initialization[initialization < 0] = first_known
    
    return initialization
    
    
def DensifyFrame(sparse_points, hard_edges, soft_edges, last_depth_map):
    w = sparse_points.shape[0]
    h = sparse_points.shape[1]
    num_pixels = w * h
    A = scipy.sparse.dok_matrix((num_pixels * 3, num_pixels), dtype=np.float32)
    A[A > 0] = 0
    A[A < 0] = 0
    b = np.zeros(num_pixels * 3, dtype=np.float32)
    x0 = np.zeros(num_pixels, dtype=np.float32)
    num_entries = 0
    
    smoothness = np.maximum(1 - soft_edges, 0)
    smoothness_x = np.zeros((w,h), dtype=np.float32)
    smoothness_y = np.zeros((w,h), dtype=np.float32)
    initialization = GetInitialization(sparse_points, last_depth_map)
                             
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_initialization_" + recon.views[frame].name,                 initialization)
        plt.imsave(output_folder + "/sparse_points_" + recon.views[frame].name,                 sparse_points)
        plt.imsave(output_folder + "/soft_edges_" + recon.views[frame].name,                 soft_edges)
        plt.imsave(output_folder + "/hard_edges_" + recon.views[frame].name,                 hard_edges)
    
    for row in tqdm(range(1,h - 1), leave=True, ascii=True, desc="by rows densify"):
        for col in range(1,w - 1):
            x0[col + row * w] = initialization[col, row]
            # Add the data constraints
            if (sparse_points[col, row] > 0.00):
                A[num_entries, col + row * w] = lambda_d
                b[num_entries] = (1.0 / sparse_points[col, row]) * lambda_d
                num_entries += 1
            elif (last_depth_map.size > 0 and last_depth_map[col, row] > 0):
                A[num_entries, col + row * w] = lambda_t
                b[num_entries] = (1.0 / last_depth_map[col, row]) * lambda_t
                num_entries += 1
    
            # Add the smoothness constraints
            smoothness_weight = lambda_s * min(smoothness[col, row],                                                smoothness[col - 1, row])
            if (hard_edges[col, row] == hard_edges[col - 1, row]):
                smoothness_x[col,row] = smoothness_weight
                A[num_entries, (col - 1) + row * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
            
            smoothness_weight = lambda_s * min(smoothness[col,row],                                                smoothness[col, row - 1])
            if (hard_edges[col,row] == hard_edges[col, row - 1]):
                smoothness_y[col,row] = smoothness_weight
                A[num_entries, col + (row - 1) * w] = smoothness_weight
                A[num_entries, col + row * w] = -smoothness_weight
                b[num_entries] = 0
                num_entries += 1
    
    
    # Solve the system
    if (dump_debug_images):
        plt.imsave(output_folder + "/solver_smoothness_x_" + recon.views[frame].name,                 smoothness_x)
        plt.imsave(output_folder + "/solver_smoothness_y_" + recon.views[frame].name,                 smoothness_y)

    [x,info] = scipy.sparse.linalg.cg(A.transpose() * A,                                       A.transpose() * b, x0, 1e-05, num_solver_iterations)
    # if info < 0:
    #     print("====> Error! Illegal input!")
    # elif info > 0:
    #     print("====> Ran " + str(info) + " solver iterations.")
    # else:
    #     print("====> Solver converged!")
    
    depth = np.zeros(sparse_points.shape, dtype=np.float32)

    # Copy back the pixels
    for row in range(0,h):
        for col in range(0,w):
            depth[col,row] = 1.0 / x[col + row * w]

    return depth

def TemporalMedian(depth_maps):
    lists = {}
    depth_map = depth_maps[0].copy()
    h = depth_map.shape[0]
    w = depth_map.shape[1]
    for row in range(0,h):
        for col in range(0,w):
            values = []
            for img in depth_maps:
                if (img[row,col] > 0):
                    values.append(img[row, col])
            if len(values) > 0:
                depth_map[row,col] = np.median(np.array(values))
            else:
                depth_map[row,col] = 0
    return depth_map


# %%
recon = ReadColmap(input_colmap, input_frames)

last_depths = []
last_depth = np.array([])

# Use the first two keyframes as initialization for the system. 
# The depth maps for these initialization frames will not be saved.
skip_frames =     recon.GetNeighboringKeyframes(recon.GetNeighboringKeyframes(recon.ViewIds()[15])[0])[1]

# print("Using the first " + str(skip_frames) + " frames to initialize (these won't be saved).")

for frame in tqdm(recon.ViewIds(), leave=True, ascii=True, desc="by frames"):
    reference_frames = recon.GetReferenceFrames(frame)
    if (len(reference_frames) == 0):
        # print("==> Skipping frame " + recon.views[frame].name +               ", No prior keyframes.")
        continue
    print("==> Processing frame " + recon.views[frame].name)

    base_img = recon.GetImage(frame)
    flows = []
    for ref in reference_frames:
        ref_img = recon.GetImage(ref) 
        flows.append(GetFlow(base_img, ref_img))
    soft_edges = GetSoftEdges(base_img, flows)
    edges = Canny(soft_edges, base_img)
    
    last_keyframe = frame
    if not recon.views [frame].IsKeyframe():
        neighboring_keyframes = recon.GetNeighboringKeyframes(frame)
        assert(len(neighboring_keyframes) > 1)
        last_keyframe = neighboring_keyframes[0]
    depth = DensifyFrame(recon.GetSparseDepthMap(last_keyframe), edges,                          soft_edges, last_depth)
    last_depths.append(depth)
    if (len(last_depths) > k_T):
        last_depths.pop(0)
    filtered_depth = TemporalMedian(last_depths)

    # Skip the first 20 frames, to make sure the depths have converged.
    # if (frame >= skip_frames):
    if (True):
        plt.imsave(output_folder + "/" + recon.views[frame].name,                    filtered_depth) 
        print("===> Depth saved to output as " + recon.views[frame].name)
    last_depth = depth


# %%



