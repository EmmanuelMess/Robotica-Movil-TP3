#!/usr/bin/env python3

import sys
import os

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header

import sensor_msgs_py.point_cloud2 as pc2

from sensor_msgs.msg import CameraInfo, Image, PointCloud2

import message_filters

from cv_bridge import CvBridge

import cv2

import numpy as np

RESULT_PATH = "/catkin_ws/src/result/"

class ImageServer(Node):
    
    def __init__(self):
        super().__init__('publish_cylinders_server')
        self.bridge = CvBridge()
        self.bfMatcher = cv2.BFMatcher()
        self.stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
        self.projectionMatrixLeft = None
        self.projectionMatrixRight = None

        self.pointCloudPoints = self.create_publisher(PointCloud2, "/points", 2)

        self.leftCameraInfoSubscriber = self.create_subscription(CameraInfo, '/left/camera_info', self.projectionMatrixLeftCallback, 1)
        self.rightCameraInfoSubscriber = self.create_subscription(CameraInfo, '/right/camera_info', self.projectionMatrixRightCallback, 1)

        self.leftRectSubscriber = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/left/image_rect"))
        self.rightRectSubscriber = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/right/image_rect"))
        timeSync = message_filters.TimeSynchronizer([self.leftRectSubscriber, self.rightRectSubscriber], 10)
        timeSync.registerCallback(self.callback)

    def projectionMatrixLeftCallback(self, message: CameraInfo):
        self.get_logger().info('Receiving left camera info')
        self.projectionMatrixLeft = message.p.reshape((3, 4))

    def projectionMatrixRightCallback(self, message: CameraInfo):
        self.get_logger().info('Receiving right camera info')
        self.projectionMatrixRight = message.p.reshape((3, 4))

    def callback(self, leftMessage, rightMessage):
        if self.projectionMatrixLeft is None or self.projectionMatrixRight is None:
            return

        self.get_logger().info('Receiving synced video frame')

        leftImage = self.bridge.imgmsg_to_cv2(leftMessage)
        rightImage = self.bridge.imgmsg_to_cv2(rightMessage)

        cv2.imwrite(os.path.join(RESULT_PATH, 'leftImage.jpg'), leftImage)
        cv2.imwrite(os.path.join(RESULT_PATH, 'rightImage.jpg'), rightImage)

        frameSiftLeft, keypointsLeft, descriptorLeft = self.extractSift(leftImage)
        frameSiftRight, keypointsRight, descriptorRight = self.extractSift(rightImage)

        cv2.imwrite(os.path.join(RESULT_PATH, 'frameSiftLeft.jpg'), frameSiftLeft)
        cv2.imwrite(os.path.join(RESULT_PATH, 'frameSiftRight.jpg'), frameSiftRight)

        good_matches = self.feature_matching(frameSiftLeft, keypointsLeft, descriptorLeft, frameSiftRight,
                                             keypointsRight, descriptorRight)
        normalizedPoints = self.triangulation(keypointsLeft, keypointsRight, good_matches)

        pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        pointsRight = np.float32([keypointsRight[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        esencialMatrix, _ = cv2.findHomography(pointsLeft, pointsRight, cv2.RANSAC, ransacReprojThreshold=2.0)

        disparityMap = self.stereo.compute(leftImage, rightImage)
        cv2.imwrite(os.path.join(RESULT_PATH, 'disparity.jpg'), disparityMap)

    def extractSift(self, currentFrame):
        sift = cv2.SIFT_create()
        keyPoints, descriptors = sift.detectAndCompute(currentFrame, None)
         
        frameSift = cv2.drawKeypoints(currentFrame, keyPoints, currentFrame,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frameSift, keyPoints, descriptors
        
    def feature_matching(self, frameCvLeft, keypointsLeft, descriptorLeft, frameCvRight, keypointsRight, descriptorRight):
        matches = self.bfMatcher.knnMatch(descriptorLeft, descriptorRight, k=2)
        matchesImage = cv2.drawMatchesKnn(frameCvLeft, keypointsLeft, frameCvRight, keypointsRight, matches, None,
                                           matchColor=(255,0,0), matchesMask=None, singlePointColor=(0,255,0), flags=0)
        cv2.imwrite(os.path.join(RESULT_PATH, 'matches.jpg'), matchesImage)
        
        ratioThreshold = 0.9
        goodMatches = []
        for m,n in matches:
            if m.distance < ratioThreshold * n.distance:
                goodMatches.append(m)
        return goodMatches
    
    def triangulation(self, keypointsLeft, keypointsRight, matches):
        pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pointsRight = np.float32([keypointsRight[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        triangulatedPoints = cv2.triangulatePoints(self.projectionMatrixLeft, self.projectionMatrixRight, pointsLeft, pointsRight)
        triangulatedPoints /= triangulatedPoints[3]
        normalizedPoints = triangulatedPoints.T[:,:3]

        cloud = pc2.create_cloud_xyz32(Header(), normalizedPoints)

        self.pointCloudPoints.publish(cloud)

        return normalizedPoints


def main(args=sys.argv):
    rclpy.init(args=args)
    node = ImageServer()
    node.get_logger().info('Image server started')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
