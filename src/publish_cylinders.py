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

class ImageServer(Node):

    def __init__(self):
        super().__init__('publish_cylinders_server')

    def a(self):
        # Creamos subscribers a las cámaras izq y der
        self.left_rect = message_filters.Subscriber(self, Image, "/left/image_rect")
        self.right_rect = message_filters.Subscriber(self, Image, "/right/image_rect")
        # Creamos el objeto ts del tipo TimeSynchronizer encargado de sincronizar los mensajes recibidos.
        ts = message_filters.TimeSynchronizer([self.left_rect, self.right_rect], 10)
        # Registramos un callback para procesar los mensajes sincronizados.
        ts.registerCallback(self.callback)

    def callback(self, left_msg, right_msg):
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
