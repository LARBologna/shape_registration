#!/usr/bin/env python
import numpy as np
import cv2 as cv
import rospy
import roslib
import math
import tf2_ros
import collections
from tf.transformations import *
import geometry_msgs.msg
import transforms3d
from std_msgs.msg import String
from collections import defaultdict
from io import StringIO
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as imga
from math import sqrt,asin,acos,cos,sin
from numpy import linalg as norm



red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
thickness = 2

myiterationsCount = 600 #number of Ransac iterations
myreprojectionError = 10.0 #max allowed distance to consider it an inliers
myconfidence = 0.90 # RANSACs succesfull confidence
pnp_flag = cv.SOLVEPNP_ITERATIVE

#WORLD REFERENCE COORDINATES OF THE PHYSICAL OBJECTS
glassdoor_wrf_coord = np.array([[0.05, 0.13, 0.04],  #left top
                                 [0.57, 0.65, 0.04]]) #right bottom
detergentbox_wrf_coord = np.array([[-0.30, 0.0, -0.01], # left top
                                   [0.0, 0.11, 0.04], # right bottom 
                                   [-0.30, 0.11, 0.03]]) # left bottom
userinterface_wrf_coord = np.array([[0.0,0.0,0.0], #left top
                                   [0.30, 0.11, 0.04]]) #bottom right
roundbutton_wrf_coord = np.array([[-0.0475, 0.01, 0.02],#left top
                                  [0.0475, 0.10, 0.04]]) #bottom right


objectPoints = np.empty((4,3), dtype = np.float32)
#objectPoints[0] = userinterface_wrf_coord[0]
#objectPoints[1] = detergentbox_wrf_coord[0]
#objectPoints[2] = detergentbox_wrf_coord[1]
objectPoints[0] = [0.0,0.0,0.0] #round center
objectPoints[1] = [0.15,0.0,0.0] #ui center
objectPoints[2] = [0.0,0.32,0.0] #glassdoor center
objectPoints[3] = [-0.15,0.0,0.0] # detergent center





def findTheta(v1,v2):
    dotProd_nom = np.dot(v1,v2)
    norm_v1 = norm.norm(v1)
    norm_v2 = norm.norm(v2)
    dotProd_denom = np.dot(norm_v1,norm_v2)
    cosTheta = dotProd_nom/dotProd_denom
    return np.arccos(cosTheta)*180/math.pi

def findBaseProjections(tf_transformation, CRF_projections):
    quaternions = np.array([tf_transformation.transform.rotation.w, tf_transformation.transform.rotation.x, tf_transformation.transform.rotation.y, tf_transformation.transform.rotation.z], dtype = np.float32)
    ROT = transforms3d.quaternions.quat2mat(quaternions)
    T_vec = np.array([tf_transformation.transform.translation.x,tf_transformation.transform.translation.y,tf_transformation.transform.translation.z],dtype = np.float32)
    TrMat = np.empty((4,4), dtype = np.float32)
    TrMat[:3, :3] = ROT
    TrMat[0, 3] = T_vec[0]
    TrMat[1, 3] = T_vec[1]
    TrMat[2, 3] = T_vec[2]
    TrMat[3, :] = [0,0,0,1]
    Projection_Base = np.dot(TrMat,CRF_projections)
    return Projection_Base

def findBaseTransformation(tf_transformation, CRF_transformation):
    quaternions = np.array([tf_transformation.transform.rotation.w, tf_transformation.transform.rotation.x, tf_transformation.transform.rotation.y, tf_transformation.transform.rotation.z], dtype = np.float32)
    ROT = transforms3d.quaternions.quat2mat(quaternions)
    T_vec = np.array([tf_transformation.transform.translation.x,tf_transformation.transform.translation.y,tf_transformation.transform.translation.z],dtype = np.float32)
    TrMat = np.empty((4,4), dtype = np.float32)
    TrMat[:3, :3] = ROT
    TrMat[0, 3] = T_vec[0]
    TrMat[1, 3] = T_vec[1]
    TrMat[2, 3] = T_vec[2]
    TrMat[3, :] = [0,0,0,1]
    Projection_Base = np.dot(TrMat,CRF_transformation)
    return Projection_Base
