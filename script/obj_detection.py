#!/usr/bin/env python
import numpy as np
import cv2 as cv
import os
# TO USE CPU
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import sys
import six
import tf2_ros
import tensorflow as tf
import rospy
import roslib
import message_filters
import math
import collections
import tf_conversions
import pyquaternion
import transforms3d
import globalVars
from globalVars import green,red,blue

from std_msgs.msg import String
from collections import defaultdict
from io import StringIO
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from numpy import linalg as norm
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from mpl_toolkits import mplot3d
import open3d as o3d
from geometry_msgs.msg import Transform, TransformStamped

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

#Global Variables
bridge = CvBridge()
rgbImg = np.zeros([480,640],dtype = np.uint8)
resizedRGBImg = np.zeros([600,600],dtype=np.uint8)
grayImg = np.zeros([480,640],dtype = np.uint8)
depthImg = np.zeros([480,640], dtype = np.float32)
enhancedDepthImg = np.zeros([480,640], dtype= np.uint8)

im_height, im_width = rgbImg.shape[:2]
font = cv.FONT_HERSHEY_SIMPLEX
fx = 0
fy = 0
cx = im_width/2
cy = im_height/2


def rgbFrameCallback(data):
    global rgbImg
    global detected_edges
    global thresholdImg
    global grayImg
    try:
        rgbImg = bridge.imgmsg_to_cv2(data,"bgr8")
        grayImg = cv.cvtColor(rgbImg, cv.COLOR_BGR2GRAY)
    except CvBridgeError as e:
        print(e)

def depthFrameCallback(data):
    global depthImg
    global enhancedDepthImg
    try:
        depthImg = bridge.imgmsg_to_cv2(data,"32FC1")
        #Enhancement
        cv_image_array = np.array(depthImg, dtype = np.float32)
        #normalization
        enhancedDepthImg = cv.normalize(cv_image_array,cv_image_array,0,1,cv.NORM_MINMAX)
        
    except CvBridgeError as e:
        print(e)

def setCameraIntrinsics(p_cameraInfoTopic):
    global im_width, im_height, fx, fy, cx, cy
    camera_info_msg  = rospy.wait_for_message(p_cameraInfoTopic, CameraInfo)
    camera_info_A = np.array(camera_info_msg.K).reshape(3,3)
    camera_info_D = np.array(camera_info_msg.D)
    im_width = camera_info_msg.width
    im_height  = camera_info_msg.height
    fx = camera_info_A[0,0]
    fy = camera_info_A[1,1]
    cx = camera_info_A[0,2]
    cy = camera_info_A[1,2]
    return camera_info_A,camera_info_D

# Load the Tensorflow model into memory.
def load_graph(PATH_TO_CKPT):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        sess = tf.compat.v1.Session(graph=detection_graph)
    return detection_graph,sess

class PoseEstimation:
    def __init__(self):
        self.knob_umin_vmin = []
        self.knob_umax_vmax = []
        self.det_umin_vmin = []
        self.det_umax_vmax = []
        self.user_umin_vmin = []
        self.user_umax_vmax = []
        self.glassdoor_umin_vmin = []
        self.glassdoor_umax_vmax = []
        self.center_glassdoor_XYZ = []
        self.center_knob_XYZ = []


        self.tfBufferHeight = tf2_ros.Buffer()
        self.tf_listenerHeight = tf2_ros.TransformListener(self.tfBufferHeight)
        #for PnP Estimation
        self.imagePoints = np.empty((4,2), dtype = np.float32)
        self.T_washOrigin_CRF = np.empty((4,4),dtype = np.float32)
        self.T_washOrigin_Depth = np.empty((4,4),dtype = np.float32)
        self.T_wrt_mobilebase = np.empty((4,4), dtype = np.float32)
        self.T_knob_wrt_machine_corner = np.empty((4,4), dtype = np.float32)

    def detectObject(self, detection_graph,sess,category_index):
        rospy.set_param('bool_human', False)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # Number of objects detected
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        resizedRGBImg = cv.resize(rgbImg,(600,600),interpolation=cv.INTER_NEAREST)
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        np_image_data = np.asarray(resizedRGBImg)
        frame_expanded = np.expand_dims(np_image_data, axis=0)
        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                                 feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(rgbImg,
                                                            np.squeeze(boxes),
                                                            np.squeeze(classes).astype(np.int32),
                                                            np.squeeze(scores),
                                                            category_index,
                                                            use_normalized_coordinates=True,
                                                            line_thickness=3,
                                                            min_score_thresh=0.60)



        sqz_boxes = np.squeeze(boxes)
        sqz_classes = np.squeeze(classes).astype(np.int32)
        sqz_scores = np.squeeze(scores)

        min_score_thresh = 0.6
        max_boxes_to_draw = 4

        text = 'Knob detected'
        kernel = np.ones((5,5),np.uint8)
        #Taking bounding box corners
        for i in range(min(max_boxes_to_draw,sqz_boxes.shape[0])):
            # walks through all boxes obtained from detection
            if  sqz_scores is None or sqz_scores[i] > min_score_thresh:
                if np.squeeze(sqz_classes)[i] in six.viewkeys(category_index):
                    object_class_name = category_index[np.squeeze(classes).astype(np.int32)[i]]['name'] # gets the name of the boxes from label_map
                    # if Knob is detected
                    if object_class_name == 'Knob':
                        rospy.set_param('bool_knob', True)
                        self.knob_umin_vmin = [int(sqz_boxes[i][1]*im_width), int(sqz_boxes[i][0]*im_height)]
                        self.knob_umax_vmax = [int(sqz_boxes[i][3]*im_width), int(sqz_boxes[i][2]*im_height)]
                    if object_class_name == 'Detergent Box':
                        self.det_umin_vmin = [int(sqz_boxes[i][1]*im_width), int(sqz_boxes[i][0]*im_height)]
                        self.det_umax_vmax = [int(sqz_boxes[i][3]*im_width), int(sqz_boxes[i][2]*im_height)]
                    if object_class_name == 'User Interface':
                        self.user_umin_vmin = [int(sqz_boxes[i][1]*im_width), int(sqz_boxes[i][0]*im_height)]
                        self.user_umax_vmax = [int(sqz_boxes[i][3]*im_width), int(sqz_boxes[i][2]*im_height)]
                    if object_class_name == 'Glassdoor':
                        self.glassdoor_umin_vmin = [int(sqz_boxes[i][1]*im_width), int(sqz_boxes[i][0]*im_height)]
                        self.glassdoor_umax_vmax = [int(sqz_boxes[i][3]*im_width), int(sqz_boxes[i][2]*im_height)]


    def depthCompute(self,point_):
        pointXYZ = []
        if len(point_)>0:
            for i in range(len(point_)):
                distance  = depthImg[point_[1],point_[0]+20]
                pointXYZ.append([((point_[0]-cx)*distance)/fx,((point_[1]-cy)*distance)/fy, distance])
                cv.circle(enhancedDepthImg,(point_[0]+20,point_[1]), 10, 255)
        else:
            print("Could not find object")
        return pointXYZ

    def pnpEstimate(self,cameraMatrix,distCoeffs):
        retval, rvec, tvec = cv.solvePnP(globalVars.objectPoints, self.imagePoints, cameraMatrix, distCoeffs)
        ROT, jacobian_rot = cv.Rodrigues(rvec)
        self.T_washOrigin_CRF[:3,:3] = ROT
        self.T_washOrigin_CRF[0,3] = tvec[0]
        self.T_washOrigin_CRF[1,3] = tvec[1]
        self.T_washOrigin_CRF[2,3] = tvec[2]
        self.T_washOrigin_CRF[3,:] = [0,0,0,1]

    def applyTransformation(self, str_to, str_from, input_matrix):
        output_array = np.empty((4,4), dtype = np.float32)
        try:
            transForHeight = self.tfBufferHeight.lookup_transform(str_to,str_from,rospy.Time())
            output_array = globalVars.findBaseTransformation(transForHeight, input_matrix)
        except tf2_ros.LookupException as e:
            print(e)
        return output_array




if __name__ == '__main__':
    count = 0
    rospy.init_node('wash_machine', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    # Topics
    rgbTopic = "/camera/rgb/image_raw"
    depthTopic = "/camera/depth_registered/image_raw"
    cameraInfoTopic = "/camera/rgb/camera_info"
    # ROS SUBSCRIBERS AND PUBLISHERS
    rgbImg_sub = rospy.Subscriber(rgbTopic, Image, rgbFrameCallback, queue_size=10)
    depthImg_sub = rospy.Subscriber(depthTopic, Image, depthFrameCallback, queue_size = 10)
    T_matrix_pub = rospy.Publisher('transformation_wrt_base', Transform, queue_size = 1)
    T_matrix_depth_pub = rospy.Publisher('transformation_wrt_depth_optical_frame', Transform, queue_size=1)
    # Fetching camera intrinsic parameters from the robot
    camera_info_A, camera_info_D = setCameraIntrinsics(cameraInfoTopic)
    # Load the graph file of the network
    abs_drc = os.path.abspath(__file__)
    path_to_scripts_ind = abs_drc.find('script')
    path_to_project = abs_drc[0:path_to_scripts_ind]
    PATH_TO_CKPT = os.path.join(path_to_project, 'Data/object_det_database/wash_detect_2.pb')
    PATH_TO_LABELS = os.path.join(path_to_project, 'Data/object_det_database/object-detection.pbtxt')
    NUM_CLASSES = 4
    graph,sess = load_graph(PATH_TO_CKPT)
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    # Object Pose Instance
    objectPose = PoseEstimation()
    # ROS Message instances
    tr_msg = TransformStamped()
    tr_msg_depth_opt_frame = Transform()
    # Creating tf2 broadcaster
    br = tf2_ros.TransformBroadcaster()
    
    while not rospy.is_shutdown():
        # ROS loop started to spin  
        try:
            # Find center point of 4 bounding boxes
            objectPose.detectObject(graph,sess,category_index)
            center_detergent_box = [int((objectPose.det_umin_vmin[0] + objectPose.det_umax_vmax[0])/2),int((objectPose.det_umin_vmin[1] + objectPose.det_umax_vmax[1])/2)]
            center_ui = [int((objectPose.user_umin_vmin[0] + objectPose.user_umax_vmax[0])/2),int((objectPose.user_umin_vmin[1] + objectPose.user_umax_vmax[1])/2)]
            center_knob = [int((objectPose.knob_umin_vmin[0] + objectPose.knob_umax_vmax[0])/2),int((objectPose.knob_umin_vmin[1] + objectPose.knob_umax_vmax[1])/2)]
            center_glassdoor = [int((objectPose.glassdoor_umin_vmin[0] + objectPose.glassdoor_umax_vmax[0])/2),int((objectPose.glassdoor_umin_vmin[1] + objectPose.glassdoor_umax_vmax[1])/2)]

            # Set 4 image points
            objectPose.imagePoints[0] = center_knob
            objectPose.imagePoints[1] = center_ui
            objectPose.imagePoints[2] = center_glassdoor
            #objectPose.imagePoints[3] = center_detergent_box
            if objectPose.det_umin_vmin[1] < objectPose.user_umin_vmin[1]:
                globalVars.objectPoints[3] = [-0.15,0.05,0.0] #[-0.30, -0.05, 0]
                objectPose.imagePoints[3] = [center_detergent_box[0], objectPose.det_umax_vmax[1]] #objectPose.det_umin_vmin
            else:
                globalVars.objectPoints[3] = [0.30, -0.05, 0]
                objectPose.imagePoints[3] = [objectPose.user_umax_vmax[0],objectPose.user_umin_vmin[1]]

            cv.circle(rgbImg,tuple(objectPose.imagePoints[0]), 5, 255,-1)
            cv.circle(rgbImg,tuple(objectPose.imagePoints[1]), 5, 255,-1)
            cv.circle(rgbImg,tuple(objectPose.imagePoints[2]), 5, 255,-1)
            cv.circle(rgbImg,tuple(objectPose.imagePoints[3]), 5,  255,-1)
            # PnP Estimation
            objectPose.pnpEstimate(camera_info_A,camera_info_D)
        except:
             print("Error occured")
        
        try:                     
            # Compute Depths
            objectPose.center_glassdoor_XYZ = objectPose.depthCompute(center_glassdoor)
            objectPose.center_knob_XYZ = objectPose.depthCompute(center_knob)            
            # To fix origin wrt rgb_optial_frame to knob center
            objectPose.T_washOrigin_CRF[0,3] = objectPose.center_knob_XYZ[0][0]
            objectPose.T_washOrigin_CRF[1,3] = objectPose.center_knob_XYZ[0][1]
            objectPose.T_washOrigin_CRF[2,3] = objectPose.center_knob_XYZ[0][2]
            
            # Apply transformation from depth_optical_frame to base_link
            objectPose.T_wrt_mobilebase = objectPose.applyTransformation('base','camera_rgb_optical_frame',objectPose.T_washOrigin_CRF)           
            # Transformation from base_link to optical frame in order to show axes
            objectPose.T_washOrigin_CRF = objectPose.applyTransformation('camera_rgb_optical_frame','base',objectPose.T_wrt_mobilebase)
            # Show coordinate frame from result of last transformation
            rot_vec,_ =  cv.Rodrigues(objectPose.T_washOrigin_CRF[:3,:3])
            tr_vec = objectPose.T_washOrigin_CRF[:3,3]
            cv.drawFrameAxes(rgbImg, camera_info_A, camera_info_D, rot_vec, tr_vec, 0.1)
            # Transformation from rgb_optical_frame to depth_optical_frame
            objectPose.T_washOrigin_Depth = objectPose.applyTransformation('camera_depth_optical_frame', 'camera_rgb_optical_frame', objectPose.T_washOrigin_CRF)
            # Transformation of origin from WASH MACHINE ORIGIN located in knob center to the down part of the wash machine - necessary for pcl_registration
            objectPose.T_knob_wrt_machine_corner[0,3] = 0 
            objectPose.T_knob_wrt_machine_corner[1,3] = 0
            objectPose.T_knob_wrt_machine_corner[2,3] = 0         
            
                       
            objectPose.T_knob_wrt_machine_corner[:3,:3] = np.identity(3,dtype=np.float32)
            objectPose.T_knob_wrt_machine_corner[3,:] = [0,0,0,1]
            # Multiplication matrices
            objectPose.T_washOrigin_Depth = np.dot(objectPose.T_knob_wrt_machine_corner,objectPose.T_washOrigin_Depth)
            
            ##### Publishers ###########################################################
            # Structuring message for transformation matrix between wash machine origin and base link
            tr_msg.header.stamp = rospy.Time.now()
            tr_msg.header.frame_id = "base"
            tr_msg.child_frame_id = "wash_machine" 
            tr_msg.transform.translation.x = objectPose.T_wrt_mobilebase[0,3]
            tr_msg.transform.translation.y = objectPose.T_wrt_mobilebase[1,3]
            tr_msg.transform.translation.z = objectPose.T_wrt_mobilebase[2,3]
            q_orig = transforms3d.quaternions.mat2quat(objectPose.T_wrt_mobilebase[:3,:3])
            euler_orig = transforms3d.euler.quat2euler(q_orig, axes='sxyz')            
            print(euler_orig)
            tr_msg.transform.rotation.x = q_orig[1]
            tr_msg.transform.rotation.y = q_orig[2]
            tr_msg.transform.rotation.z = q_orig[3]
            tr_msg.transform.rotation.w = q_orig[0]

            # Structuring message for transformation matrix between wash machine origin and depth optical frame
            tr_msg_depth_opt_frame.translation.x = objectPose.T_washOrigin_Depth[0,3]
            tr_msg_depth_opt_frame.translation.y = objectPose.T_washOrigin_Depth[1,3]
            tr_msg_depth_opt_frame.translation.z = objectPose.T_washOrigin_Depth[2,3]
            q_orig_depth = transforms3d.quaternions.mat2quat(objectPose.T_washOrigin_Depth[:3,:3]) 
            tr_msg_depth_opt_frame.rotation.x = q_orig_depth[1]
            tr_msg_depth_opt_frame.rotation.y = q_orig_depth[2]
            tr_msg_depth_opt_frame.rotation.z = q_orig_depth[3]
            tr_msg_depth_opt_frame.rotation.w = q_orig_depth[0]            

            print(tr_msg.transform)
            T_matrix_pub.publish(tr_msg.transform)
            # Publishing wash mash origin located in below wrt Depth Optical Frame
            T_matrix_depth_pub.publish(tr_msg_depth_opt_frame) 
            br.sendTransform(tr_msg)
        except:
            pass

        # Visualization
        cv.namedWindow("rgb")
        cv.imshow("rgb", rgbImg)
        cv.namedWindow("depth image")
        cv.imshow("depth image", enhancedDepthImg)
        cv.waitKey(3)

        # Deleting dynamic memories
        del objectPose.center_glassdoor_XYZ[:]
        del objectPose.center_knob_XYZ[:]
        del objectPose.user_umin_vmin[:]
        del objectPose.user_umax_vmax[:]
        del objectPose.knob_umin_vmin[:]
        del objectPose.knob_umax_vmax[:]
        del objectPose.det_umax_vmax[:]
        del objectPose.det_umin_vmin[:]
        del objectPose.glassdoor_umin_vmin[:]
        del objectPose.glassdoor_umax_vmax[:]
        
        rate.sleep()
    cv.release()
    cv.destroyAllWindows()



