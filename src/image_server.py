#!/usr/bin/env python3

import sys
import os
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

import numpy as np

RESULT_PATH = "/catkin_ws/src/result/"

class ImageServer(Node):
    
    def __init__(self):
        super().__init__('publish_cylinders_server')
        self.bridge = CvBridge()
        self.bfMatcher = cv2.BFMatcher()
        self.stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
        self.projectionMatrixLeft = [0,0,0] #Supongo que de la calibracion viene esto nvm
        self.projectionMatrixRight = [0,0,0]

        self.syncImages()

    def syncImages(self):
        left_rect = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/left/image_rect"))
        right_rect = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/right/image_rect"))
        ts = message_filters.TimeSynchronizer([left_rect, right_rect], 10)
        ts.registerCallback(self.callback)

    def callback(self, left_msg, right_msg):
        self.get_logger().info('Receiving synced video frame')

        frameSiftLeft, keypointsLeft, descriptorLeft = self.extract_SIFT(left_msg)
        cv2.imwrite(os.path.join(RESULT_PATH, 'current_frame_left_sift.jpg'), frameSiftLeft)

        frameSiftRight, keypointsRight, descriptorRight = self.extract_SIFT(right_msg)
        cv2.imwrite(os.path.join(RESULT_PATH, 'current_frame_right_sift.jpg'), frameSiftRight)

        good_matches = self.feature_matching(frameSiftLeft, keypointsLeft, descriptorLeft, frameSiftRight,
                                             keypointsRight, descriptorRight)
        normalizedPoints = self.triangulation(keypointsLeft, keypointsRight, good_matches)
        esencialMatrix = self.findHomograficaOEsencial(keypointsLeft, keypointsRight, good_matches)
        currentFrameLeft = self.bridge.imgmsg_to_cv2(left_msg)
        currentFrameRight = self.bridge.imgmsg_to_cv2(right_msg)
        disparityMap = self.stereoMatchesDisparity(currentFrameLeft, currentFrameRight)

    def extract_SIFT(self, frame_msg):
        currentFrame = self.bridge.imgmsg_to_cv2(frame_msg)
        cv2.imwrite(os.path.join(RESULT_PATH, 'sift_frame.jpg'), currentFrame)
         
        sift = cv2.SIFT_create()
        keyPoints, descriptors = sift.detectAndCompute(currentFrame, None)
         
        frame_sift=cv2.drawKeypoints(currentFrame, keyPoints, currentFrame, 
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame_sift, keyPoints, descriptors
        
    def feature_matching(self, frameCvLeft, keypointsLeft, descriptorLeft, frameCvRight, keypointsRight, descriptorRight):
        matches = self.bfMatcher.knnMatch(descriptorLeft, descriptorRight, k=2)
        matches_image = cv2.drawMatchesKnn(frameCvLeft, keypointsLeft, frameCvRight, keypointsRight, matches, None,
                                           matchColor=(255,0,0), matchesMask=None, singlePointColor=(0,255,0), flags=0)
        cv2.imwrite(os.path.join(RESULT_PATH, 'matches.jpg'), matches_image)
        
        ratio_thresh = 0.9
        good_matches = []
        for m,n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        return good_matches
    
    def triangulation(self, keypointsLeft, keypointsRight, matches):
        pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pointsRight = np.float32([keypointsRight[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        triangulatedPoints = cv2.triangulatePoints(self.projectionMatrixLeft, self.projectionMatrixRight, pointsLeft, pointsRight)
        # TODO Messu publica pts como un PointCLoud2 asi va a RVIZ2
        triangulatedPoints /= triangulatedPoints[3]
        normalizedPoints = triangulatedPoints.T[:,:3]
        return normalizedPoints
        
    def findHomograficaOEsencial(self, kpntL,kpntR, matches):
        pointsLeft = np.float32([kpntL[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pointsRight = np.float32([kpntR[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        esencial, _ = cv2.findHomography(pointsLeft, pointsRight, cv2.RANSAC, ransacReprojThreshold = 2.0)
        return esencial
        
    def stereoMatchesDisparity(self, frameL, frameR):
        disparity = self.stereo.compute(frameL,frameR)
        cv2.imwrite(os.path.join(RESULT_PATH, 'gray.jpg'), disparity)
        return disparity

def main(args=sys.argv):
    rclpy.init(args=args)
    node = ImageServer()
    node.get_logger().info('Image server started')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
