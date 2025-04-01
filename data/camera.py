#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
from dataset_readers import ProjectionType

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = np.tan((fovY / 2))
    tanHalfFovX = np.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    z_sign = 1.0

    # P = torch.zeros(4, 4)

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)
    P = np.array([
       [2.0 * znear / (right - left),     0.0,                          (right + left) / (right - left), 0.0 ],
       [0.0,                              2.0 * znear / (top - bottom), (top + bottom) / (top - bottom), 0.0 ],
       [0.0,                              0.0,                          z_sign * zfar / (zfar - znear),  -(zfar * znear) / (zfar - znear) ],
       [0.0,                              0.0,                          z_sign,                          0.0 ]
    ])
    return P

class Camera:
    def __init__(self, colmap_id, R, T, fovx, fovy, image, gt_alpha_mask,
                 image_name, uid, cx, cy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 model=ProjectionType.PERSPECTIVE, distortion_params=None,
                 exposure=1, iso=100, aperature=1):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.fovx = fovx
        self.fovy = fovy
        self.cx = cx
        self.cy = cy
        self.image_name = image_name
        self.model = model
        self.distortion_params = distortion_params
        self.glo_vector = None
        self.exposure = exposure
        self.iso = iso
        self.aperature = aperature

        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= np.ones((1, self.image_height, self.image_width))

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale
        self.update()

    def update(self):
        self.world_view_transform = np.array(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.fovx, fovY=self.fovy).transpose(0,1)
        self.camera_center = np.linalg.inv(self.world_view_transform)[3, :3]
