"""
ROS2 Contact Detection Node
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import rclpy
import rclpy.executors
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

    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                self.get_logger().error("Model file not found.")
                self.get_logger().error("Please train the model first.")
                return False
            
            self.scaler = joblib(self.scaler_path)
            input_size = 7*3
            self.model = ContactClassifier(input_size).to(self.device)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            self.get_logger().info("Model loaded successfully.")
            return True

        except Exception as e:
            self.get_logger().error("Failed to load model")
            return False


    def joint_state_callback(self, msg: JointState):
        self.last_joint_state = msg
    

    def predict_contact(self, positions: np.ndarray, velocities: np.ndarray, torques: np.ndarray) -> float:
        features = np.concatenate([positions, velocities, torques])
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(self.device)
            prediction = self.model(features_tensor).squeeze().cpu().numpy()

        return float(prediction)


    def publish_contact_state(self, contact_detected: bool, confidence: float):
        contact_msg = Bool()
        contact_msg.data = contact_detected
        self.contact_pub.publish(contact_msg)
        confidence_msg = Float32()
        confidence_msg.data = confidence
        self.confidence_pub.publish(confidence_msg)
        wrench_msg = WrenchStamped()
        wrench_msg.header.stamp = self.get_clock().now().to_msg()
        wrench_msg.header.frame_id = "base_link"

        estimated_force = confidence * 10.0 if contact_detected else 0.0
        wrench_msg.wrench.force.z = estimated_force

        self.wrench_pub.publish(wrench_msg)

    
    def process_contact_detection(self):
        if self.last_joint_state is None or self.model is None:
            return 
        
        try:
            joint_pos = np.array(self.last_joint_state.position[:self.num_joints])
            joint_vel = np.array(self.last_joint_state.velocity[:self.num_joint]) if len(self.last_joint_state.velocity) > 0 else np.zeros(self.num_joints)
            joint_tor = np.array(self.last_joint_state.effort[:self.num_joints]) if len(self.last_joint_state.effort) > 0 else np.zeros(self.num_joints)

            if len(joint_pos) != self.num_joints:
                self.get_logger().warn(f"Expected {self.num_joints} joints, got {len(joint_pos)}")
                return 
            
            confidence = self.predict_contact(joint_pos, joint_vel, joint_tor)

            self.contact_history.append(confidence)
            if len(self.contact_history) > self.history_size:
                self.contact_history.pop(0)
            
            avg_confidence = np.mean(self.contact_history)
            contact_detected = avg_confidence > self.contact_threshold
            self.publish_contact_state(contact_detected, avg_confidence)
            if contact_detected and avg_confidence > self.confidence_threshold:
                self.get_logger.info(f"Contact detected with confidence {avg_confidence:.3f}.")

        except Exception as e:
            self.get_logger().error("Error in contact detection.")


    def get_contact_stats(self) -> dict:
        if len(self.contact_history) == 0:
            return {"avg_confidence": 0.0, "contact_rate": 0.0}

        avg_confidence = np.mean(self.contact_history)
        contact_rate = np.mean([c > self.contact_threshold for c in self.contact_history])

        return {
            "avg_confidence": avg_confidence,
            "contact_rate": contact_rate,
            "history_size": len(self.contact_history)
        }



class ContactDetectorSimNode(Node):
    def __init__(self):
        super().__init__('contact_detector_sim')
        


def main(args=None):
    rclpy.init(args=args)
    detector_node = ContactDetectorNode()
    # add sim node

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(detector_node)
    # add sim node ot executor here
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        detector_node.destroy_node()
        # dont forget to destroy sim node
        rclpy.shutdown()


if __name__ == "__main__":
    main()