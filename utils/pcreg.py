"""
The class uses the Kinect 1.0 calibration parameters to return a registred point cloud to RGB data.

Returns:pointclaud,rgb: In 3D projected pointclud and a coresponding colors of the Kinect RGB camera.
More on: https://github.com/fvilmos/kinect_point_cloud
"""
import numpy as np
import cv2

class PointCloudRegistration():
    '''
    Register RGB image on a Point Cloud 
    '''
    # depth cam paramaters
    DepthCamParams = {
        "fx": 5.8818670481438744e+02,
        "fy": 5.8724220649505514e+02,
        "cx": 3.1076280589210484e+02,
        "cy": 2.2887144980135292e+02,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "a": -0.0030711,
        "b": 3.3309495,
    }

    # RGB cam parameters
    RGBCamParams = {
        "fx": 5.2161910696979987e+02,
        "fy": 5.2132946256749767e+02,
        "cx": 3.1755491910920682e+02,
        "cy": 2.5921654718027673e+02,
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "k3": 0.0,
        "rot": np.array([[9.9996518012567637e-01, 2.6765126468950343e-03, -7.9041012313000904e-03],
                         [-2.7409311281316700e-03, 9.9996302803027592e-01, -8.1504520778013286e-03],
                         [7.8819942130445332e-03, 8.1718328771890631e-03, 9.9993554558014031e-01]]),
        "trans": np.array([[-2.5558943178152542e-02, 1.0109636268061706e-04, 2.0318321729487039e-03]])
    }

    def __init__(self):
        pass

    def depth_cam_mat(self):
        '''
        Returns camera matrix, including the transformation values of depth to meters
        :return: camera (intrisec) matrix
        '''

        mat = np.array([[1 / self.DepthCamParams['fx'], 0, 0, -self.DepthCamParams['cx'] / self.DepthCamParams['fx']],
                        [0, 1 / self.DepthCamParams['fy'], 0, -self.DepthCamParams['cy'] / self.DepthCamParams['fy']],
                        [0, 0, 0, 1],
                        [0, 0, self.DepthCamParams['a'], self.DepthCamParams['b']]])

        return mat

    def get_registred_depth_rgb(self, depth, img):
        '''
        Returns the registred pointclaud and image with transforming the cameras position in world coordinate system
        :return: registred point cloud and image
        '''

        if depth is not None and img is not None:
            h, w = img.shape[:2]

            depth = np.array(depth, dtype=np.float32)

            # project points to 3D space
            points = cv2.reprojectImageTo3D(depth, self.depth_cam_mat())

            # transform coordinates to RGB camera coordinates
            points = np.dot(points, self.RGBCamParams['rot'].T)
            points = np.add(points, self.RGBCamParams['trans'])

            # handle invalid values
            points[depth >= depth.max()] = 0.0

            points = points.reshape(-1, w, 3)

            # project 3D points back to image plain
            with np.errstate(divide='ignore', invalid='ignore'):
                x = np.array((points[:, :, 0] * (self.RGBCamParams['fx'] / points[:, :, 2]) + self.RGBCamParams['cx']),
                        dtype=np.int).clip(0, w - 1)
                y = np.array((points[:, :, 1] * (self.RGBCamParams['fy'] / points[:, :, 2]) + self.RGBCamParams['cy']),
                        dtype=np.int).clip(0, h - 1)

        return points, img[y, x]
