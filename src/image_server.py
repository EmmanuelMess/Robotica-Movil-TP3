#!/usr/bin/env python3

import sys
import os

from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header

import sensor_msgs_py.point_cloud2 as pc2

from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from geometry_msgs.msg import Pose, Point, Quaternion, PoseStamped

import message_filters

from cv_bridge import CvBridge

import cv2

import numpy as np

import math

import yaml

RESULT_PATH = "/catkin_ws/src/result/"
CAMERA_CALIBRATION_DATA_PATH = "/catkin_ws/calibrationdata/"
TARA_DISTANCE = 0.060
DEBUG = True

class ImageServer(Node):
    
    def __init__(self):
        super().__init__('publish_cylinders_server')
        self.image_index = 0
        self.bridge = CvBridge()
        self.bfMatcher = cv2.BFMatcher()
        self.stereo = cv2.StereoBM_create(numDisparities = 16, blockSize = 15)
        self.projectionMatrixLeft = None
        self.intrisicsMatrixRight = None
        self.distortionCoeffLeft = None
        self.projectionMatrixRight = None
        self.distortionCoeffRight = None
        self.intrisicsMatrixRight = None

        self.keypointsLeftAccumulated = []
        self.descriptorLeftAccumulated = []
        self.imagesLeft = []

        self.tAccumulated = Point()
        self.rotAccumulated = R.from_matrix(np.matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).as_quat()
        
        # Calibration Data. Ex 4
        self.projectionMatrixLeft, self.intrisicsMatrixLeft, self.distortionCoeffLeft = self.extractCal(CAMERA_CALIBRATION_DATA_PATH+"left.yaml")
        self.projectionMatrixRight, self.intrisicsMatrixRight, self.distortionCoeffRight = self.extractCal(CAMERA_CALIBRATION_DATA_PATH+"right.yaml")

        self.pointCloudPublisher = self.create_publisher(PointCloud2, "/points", 2)
        self.densePointCloudPublisher = self.create_publisher(PointCloud2, "/points_dense", 2)

        self.cameraPoseLeft = self.create_publisher(PoseStamped, "/pose_camera_left", 2)
        self.cameraPoseRight = self.create_publisher(PoseStamped, "/pose_camera_right", 2)

        # Synch of images and rectify. Ex 5 and Ex 6 
        self.leftRectSubscriber = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/left/image_rect"))
        self.rightRectSubscriber = message_filters.Subscriber(self, Image, os.path.join(RESULT_PATH, "/right/image_rect"))
        timeSync = message_filters.TimeSynchronizer([self.leftRectSubscriber, self.rightRectSubscriber], 10)
        timeSync.registerCallback(self.callback)

    def callback(self, leftMessage, rightMessage):
        if self.projectionMatrixLeft is None or self.projectionMatrixRight is None:
            return

        self.get_logger().info('Receiving synced video frame')

        # Transform image_msg to image, we're starting the matching process
        leftImage = self.bridge.imgmsg_to_cv2(leftMessage)
        rightImage = self.bridge.imgmsg_to_cv2(rightMessage)

        if DEBUG:
            self.get_logger().info('Saving video frame of both cameras')
            cv2.imwrite(os.path.join(RESULT_PATH, 'leftImage.jpg'), leftImage)
            cv2.imwrite(os.path.join(RESULT_PATH, 'rightImage.jpg'), rightImage)

        # Extract the keypoints and descriptors using SIFT. Ex 7
        frameSiftLeft, keypointsLeft, descriptorLeft = self.extractSift(leftImage)
        frameSiftRight, keypointsRight, descriptorRight = self.extractSift(rightImage)

        if DEBUG:
            # Saving the SIFT FRAMES and KPs
            self.saveSIFTFrameAndDrawKeypoints(frameSiftLeft, leftImage, keypointsLeft, 'narrow_stereo_left')
            self.saveSIFTFrameAndDrawKeypoints(frameSiftRight, rightImage, keypointsRight, 'narrow_stereo_right')
            
        # Compute matches between both images. Ex 8
        goodMatches = self.featureMatching(descriptorLeft, descriptorRight)

        if DEBUG:
            # Store all matches (good matches)
            matchesImage = cv2.drawMatchesKnn(leftImage, keypointsLeft, rightImage, keypointsRight, goodMatches,
                                              None,
                                              matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(0, 255, 0),
                                              flags=0)
            cv2.imwrite(os.path.join(RESULT_PATH, 'matches','allMatches','matches_image' + str(self.image_index) + '.jpg'), matchesImage)
            
            # Store all matches with distance less than 30 
            matchesFiltered = list(filter(lambda match: match[0].distance <= 30, goodMatches))
            matchesImageDistance = cv2.drawMatchesKnn(leftImage, keypointsLeft, rightImage, keypointsRight, matchesFiltered,
                                                      None,
                                                      matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(0, 255, 0),
                                                      flags=0)
            cv2.imwrite(os.path.join(RESULT_PATH, 'matches','matchesFilteredByDistance','matches_image' + str(self.image_index) + '.jpg'), matchesImageDistance)

        # Triangulation. Ex 9
        normalizedPoints, pointsLeft, pointsRight = self.triangulation(keypointsLeft, keypointsRight, goodMatches)
        
        # RANSAC. Ex 10
        # Homography
        esencialMatrix, _ = cv2.findHomography(pointsLeft, pointsRight, cv2.RANSAC, ransacReprojThreshold=2.0)
        self.transformLeftKPsInRightFrame(pointsLeft, esencialMatrix, rightImage)

        # DisparityMap. Ex 11
        disparityMap = self.stereo.compute(leftImage, rightImage)

        if DEBUG:
            cv2.imwrite(os.path.join(RESULT_PATH, 'disparityMaps', 'disparityMap' + str(self.image_index) + '.jpg'), disparityMap)

        # Dense Reconstruction. Ex 12
        self.reconstruct3d(np.array([leftImage.shape[1], leftImage.shape[0]]), disparityMap)

        # Pose estimation. Ex 13
        self.monocularPose(esencialMatrix, leftImage, descriptorLeft, keypointsLeft)
        
        self.image_index = self.image_index + 1

    def transformLeftKPsInRightFrame(self, pointLeft, homography, right_frame):
        rightKps = cv2.perspectiveTransform(pointLeft, homography).reshape(-1, 2)
        frame_with_keypoint_transform = cv2.cvtColor(right_frame, cv2.COLOR_GRAY2RGB)
        for kp in rightKps:
            kp_pos = (round(kp[0]), round(kp[1]))
            frame_with_keypoint_transform = cv2.circle(frame_with_keypoint_transform, kp_pos, 3, (0, 255, 0))
        cv2.imwrite(os.path.join(RESULT_PATH, 'keypoints transform L->R', 'framekp' + '_id_' + str(self.image_index) + '.jpg'), frame_with_keypoint_transform)

    def saveSIFTFrameAndDrawKeypoints(self, siftFrame, image_raw, image_keypoint, camera_name):
        # Save SIFT Frame
        cv2.imwrite(os.path.join(RESULT_PATH, 'SIFT Frames', 'frameSift_'+ camera_name + '_id_' + str(self.image_index) + '.jpg'), siftFrame)
        # Save Keypoints
        frame_with_keypoint = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2RGB)
        for kp in image_keypoint:
            kp_pos = (round(kp.pt[0]), round(kp.pt[1]))
            frame_with_keypoint = cv2.circle(frame_with_keypoint, kp_pos, 3, (255, 0, 0))
        cv2.imwrite(os.path.join(RESULT_PATH, 'keypoints', 'framekp_'+ camera_name + '_id_' + str(self.image_index) + '.jpg'), frame_with_keypoint)


    def extractCal(self, cal_camera_file):
        with open(cal_camera_file, 'r') as stream:
            cal_camera_data = yaml.safe_load(stream)
        
        camera_name = cal_camera_data['camera_name']
        self.get_logger().info('Extracting data cal. of ' + str(camera_name))
        
        proj_matrix_raw = np.array(cal_camera_data['projection_matrix']['data'])
        proj_matrix = proj_matrix_raw.reshape(cal_camera_data['projection_matrix']['rows'],
                                              cal_camera_data['projection_matrix']['cols'])
        self.get_logger().info('Projection Matrix extracted')
        
        camera_matrix_raw = np.array(cal_camera_data['camera_matrix']['data'])
        camera_matrix = camera_matrix_raw.reshape(cal_camera_data['camera_matrix']['rows'],
                                                  cal_camera_data['camera_matrix']['cols'])
        self.get_logger().info('Intrinsic Matrix extracted')

        dist_coeff = np.array(cal_camera_data['distortion_coefficients']['data'])
        self.get_logger().info('Distortion Coeff. extracted')
                
        return proj_matrix, camera_matrix, dist_coeff


    def extractSift(self, currentFrame):
        sift = cv2.SIFT_create()
        keyPoints, descriptors = sift.detectAndCompute(currentFrame, None)
         
        frameSift = cv2.drawKeypoints(currentFrame, keyPoints, currentFrame,
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frameSift, keyPoints, descriptors
        
    def featureMatching(self, descriptorLeft, descriptorRight):
        matches = self.bfMatcher.knnMatch(descriptorLeft, descriptorRight, k=2)

        ratioThreshold = 0.99
        goodMatches = []
        for match in matches:
            m, n = match
            if m.distance < ratioThreshold * n.distance:
                goodMatches.append(match)

        return goodMatches

    def triangulation(self, keypointsLeft, keypointsRight, matches):
        self.get_logger().info('Publishing triangulation')

        pointsLeft = np.float32([keypointsLeft[m.queryIdx].pt for (m,_) in matches]).reshape(-1, 1, 2)
        pointsRight = np.float32([keypointsRight[m.trainIdx].pt for (m,_) in matches]).reshape(-1, 1, 2)
        
        triangulatedPoints = cv2.triangulatePoints(self.projectionMatrixLeft, self.projectionMatrixRight, pointsLeft, pointsRight)
        triangulatedPoints /= triangulatedPoints[3]
        normalizedPoints = triangulatedPoints.T[:,:3]

        pointCloudHeader = Header()
        pointCloudHeader.frame_id = "map"

        cloud = pc2.create_cloud_xyz32(pointCloudHeader, normalizedPoints)

        self.pointCloudPublisher.publish(cloud)

        return normalizedPoints, pointsLeft, pointsRight

    def reconstruct3d(self, imageSize, disparityMap):
        R = np.array([0, 0, 0])
        T = np.array([TARA_DISTANCE, 0, 0])

        self.get_logger().info('Publishing 3D reconstruction')

        _, _, _, _, Q, _, _ = cv2.stereoRectify(self.intrisicsMatrixLeft, self.distortionCoeffLeft,
                                                self.intrisicsMatrixRight, self.distortionCoeffRight, imageSize, R, T)
        spacialRepresentation = cv2.reprojectImageTo3D(disparityMap, Q)

        densePointCloudHeader = Header()
        densePointCloudHeader.frame_id = "map"

        denseCloud = pc2.create_cloud_xyz32(densePointCloudHeader, spacialRepresentation)
        self.densePointCloudPublisher.publish(denseCloud)

    def monocularPose(self, essentialMatrix, imageLeftNew, descriptorLeftNew, keypointsLeftNew):
        if not self.keypointsLeftAccumulated:
            self.imagesLeft.append(imageLeftNew)
            self.descriptorLeftAccumulated.append(descriptorLeftNew)
            self.keypointsLeftAccumulated.append(keypointsLeftNew)
            return

        self.get_logger().info('Publishing 3D reconstruction')

        keypointsLeftOld = self.keypointsLeftAccumulated[-1]
        descriptorLeftOld = self.descriptorLeftAccumulated[-1]

        goodMatchesTemporal = self.featureMatching(descriptorLeftOld, descriptorLeftNew)
        goodPointsOld = np.float32([keypointsLeftOld[m.queryIdx].pt for (m, _) in goodMatchesTemporal]).reshape(-1, 1, 2)
        goodPointsNew = np.float32([keypointsLeftNew[m.trainIdx].pt for (m, _) in goodMatchesTemporal]).reshape(-1, 1, 2)

        _, rotLeft, tLeft, mask = cv2.recoverPose(essentialMatrix, goodPointsOld, goodPointsNew, self.intrisicsMatrixLeft)

        if DEBUG:
            imageLeftOld = self.imagesLeft[-1]

            matchesImageTemporal = cv2.drawMatchesKnn(imageLeftOld, keypointsLeftOld, imageLeftNew, keypointsLeftNew, goodMatchesTemporal,
                                              None,
                                              matchColor=(255, 0, 0), matchesMask=None, singlePointColor=(0, 255, 0),
                                              flags=0)
            cv2.imwrite(os.path.join(RESULT_PATH, 'matchesImageTemporal' + '_id_' + str(self.image_index) + '.jpg'), matchesImageTemporal)

            self.imagesLeft.append(imageLeftNew)

        self.descriptorLeftAccumulated.append(descriptorLeftNew)
        self.keypointsLeftAccumulated.append(keypointsLeftNew)

        self.tAccumulated.x += float(tLeft[0]) * TARA_DISTANCE
        self.tAccumulated.y += float(tLeft[1]) * TARA_DISTANCE
        self.tAccumulated.z += float(tLeft[2]) * TARA_DISTANCE

        self.rotAccumulated *= R.from_matrix(rotLeft).as_quat()

        leftPose = PoseStamped()
        leftPose.header = Header()
        leftPose.header.frame_id = "map"
        leftPose.pose = Pose()
        leftPose.pose.position = self.tAccumulated
        leftPose.pose.orientation = Quaternion()
        leftPose.pose.orientation.x = self.rotAccumulated[0]
        leftPose.pose.orientation.y = self.rotAccumulated[1]
        leftPose.pose.orientation.z = self.rotAccumulated[2]
        leftPose.pose.orientation.w = self.rotAccumulated[3]
        self.cameraPoseLeft.publish(leftPose)

        self.get_logger().info('New position: ' + str(self.tAccumulated))


def main(args=sys.argv):
    rclpy.init(args=args)
    node = ImageServer()
    node.get_logger().info('Image server started')
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
