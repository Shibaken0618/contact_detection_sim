"""
ROS2 Contact Detection Node
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped
import joblib
from typing import List, Optional
from cd_ml_classifier import ContactClassifier


MODEL_PATH = "models/contact_detector.pth"
SCALER_PATH = "models/scaler.pkl"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class ContactDetectorNode(Node):
    def __init__(self):
        super().__init__('contact_detector')
        self.model_path = MODEL_PATH
        self.scaler_path = SCALER_PATH
        self.model = None
        self.scaler = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.contact_threshold = 0.5
        self.confidence_threshold = 0.7
        self.num_joints = 7
        self.last_joint_state = None
        self.contact_history = []
        self.history_size = 5

        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.contact_pub = self.create_publisher(Bool, '/contact_detected', 10)
        self.confidence_pub = self.create_publisher(Float32, '/contact_confidence', 10)
        self.wrench_pub = self.create_publisher(WrenchStamped, '/estimated_wrench', 10)

        self.timer = self.create_timer(0.01, self.process_contact_detection)
        self.load_model()
        self.get_logger().info('Contact Detection Node Initialized')

def main(args=None):
    pass


if __name__ == "__main__":
    main()