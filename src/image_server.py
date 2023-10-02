#!/usr/bin/env python3

import sys
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
import laser_geometry.laser_geometry as lg
import sensor_msgs_py.point_cloud2 as pc2
from itertools import groupby
from operator import itemgetter
import message_filters

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 
from matplotlib import pyplot as plt

#Correr el punto del: ros2 launch stereo_image_proc stereo_image_proc.launch.py
#Correr el rosbag
#Correr este con python3 XXXXX.py

class ImageServer(Node):
    
    def __init__(self):
        super().__init__('publish_cylinders_server')
        self.bridge = CvBridge()
        self.bf = cv2.BFMatcher()
        self.stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
        self.projectionMatrixLeft = [0,0,0] #Supongo que de la calibracion viene esto nvm
        self.projectionMatrixRight = [0,0,0]
    
    def extract_SIFT(frame_msg):
        current_frame = self.bridge.imgmsg_to_cv2(frame_msg)
        cv2.imshow("camera", current_frame)  
      
        # Converting image to grayscale
        gray= cv2.cvtColor(current_frame,cv2.COLOR_BGR2GRAY)
         
        # Applying SIFT detector
        sift = cv2.xfreatures2d.SIFT_create()
        kp, descr = sift.detectAndCompute(current_frame, None) #Son los keypoint
         
        # Marking the keypoint on the image using circles
        frame_sift=cv2.drawKeypoints(gray,
                                     kp,
                                     current_frame,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return (frame_sift,kp, descr)
        
    def feature_matching(frame_cv_left, kpntL, descrL,frame_cv_right,kpntR, descrR):
        matches = bf.knnMatch(descrL,descrR,k=2)
        matches_image = cv2.drawMatchesKnn(frame_cv_left,kpntL,frame_cv_right,kpntR,matches, None,
                                           matchColor=(255,0,0), matchesMask=None,
                                           singlePointColor=(0,255,0), flags=0)
        cv2.imshow("Matches L and R", matches_image)
        
        ratio_thresh = 0.9
        good_matches = []
        for m,n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        cv2.waitKey(0)
        return good_matches
    
    def triangulacion(kpntL,kpntR, matches):
        ptsL = np.float32([kpntL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ptsR = np.float32([kpntR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        triagulitos = cv2.triangulatePoint(self.projectionMatrixLeft,self.projectionMatrixRight,
                                           ptsL,ptsR)
        #Messu publica pts como un PointCLoud2 asi va a RVIZ2
        triagulitos /= triagulitos[3]
        pts = triagulitos.T[:,:3]
        return pts
        
    def findHomograficaOEsencial(kpntL,kpntR, matches):
        ptsL = np.float32([kpntL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        ptsR = np.float32([kpntR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        esencial, _ = cv2.findHomography(ptsL,ptsR, cv2.RANSAC, ransacReprojThreshold = 2.0)
        return esencial
        
    def stereo_matches_disparity(frame_L,frame_R):
        disparity = self.stereo.compute(frameL,frameR)
        plt.imshow(disparity, 'gray')
        plt.show()
        return disparity
    
    def a(self):
        # Creamos subscribers a las cámaras izq y der
        self.left_rect = message_filters.Subscriber(self, Image, "/left/image_rect")
        self.right_rect = message_filters.Subscriber(self, Image, "/right/image_rect")
        # Creamos el objeto ts del tipo TimeSynchronizer encargado de sincronizar los mensajes recibidos.
        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)
        # Registramos un callback para procesar los mensajes sincronizados.
        ts.registerCallback(self.callback)

    def callback(self, left_msg, right_msg):
        self.get_logger().info('Receiving video frame')
        ##############
        frame_SIFT_left, kpntL, descrL = self.extract_SIFT(left_msg)
        cv2.imshow("cameraLeft", frame_SIFT_left)
        cv2.imwrite('current_frame_left_sift.jpg', current_frame_left_sift)
        
        frame_SIFT_right, kpntR, descrR = self.extract_SIFT(right_msg)
        cv2.imshow("cameraRight", frame_SIFT_right)
        cv2.imwrite('current_frame_right_sift.jpg', frame_SIFT_right)
        ##############
        good_matches = self.feature_matching(frame_cv_left, kpntL, descrL,frame_cv_right,kpntR, descrR)
        ##############
        pts = self.triangulacion(kpntL,kpntR, matches)
        ##############
        esencialMatrix = self.findHomograficaOEsencial(kpntL,kpntR, matches)
        ##############
        current_frame_L = self.bridge.imgmsg_to_cv2(left_msg)
        current_frame_R = self.bridge.imgmsg_to_cv2(right_msg)
        disparityMap = self.stereo_matches_disparity(current_frame_L,current_frame_R)
        ##############
        cv2.waitKey(1)
        pass

def main(args=sys.argv):
    rclpy.init(args=args)
    node = ImageServer()
    node.get_logger().info('Image server started')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
