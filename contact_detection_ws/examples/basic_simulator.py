#!/usr/bin/env python3
"""
Basic Contact Detection Simulator using PyBullet
This simulator creates a robot arm, moves it around, and detects contacts.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class JointState:
    """Data class to store joint state information"""
    positions: List[float]
    velocities: List[float]
    applied_torques: List[float]
    timestamp: float
    contact_detected: bool = False

class ContactDetectionSimulator:
    def __init__(self, gui=True, robot_urdf="kuka_iiwa/model.urdf"):
        """Initialize the contact detection simulator"""
        self.gui = gui
        self.robot_urdf = robot_urdf
        self.robot_id = None
        self.plane_id = None
        self.physics_client = None
        self.num_joints = 0
        self.joint_data = []
        
        # Contact detection parameters
        self.contact_threshold = 0.1  # Force threshold for contact detection
        
    def setup_simulation(self):
        """Set up the PyBullet simulation environment"""
        # Connect to PyBullet
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
            
        # Set up physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1./240.)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Add some objects for contact
        self._add_contact_objects()
        
        # Load robot
        self.robot_id = p.loadURDF(self.robot_urdf, [0, 0, 0])
        self.num_joints = p.getNumJoints(self.robot_id)
        
        print(f"Simulation setup complete. Robot has {self.num_joints} joints.")
        
    def _add_contact_objects(self):
        """Add objects in the environment for the robot to potentially contact"""
        # Add a box
        box_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[1, 0, 0, 1])
        self.box_id = p.createMultiBody(baseMass=1.0, 
                                       baseCollisionShapeIndex=box_collision,
                                       baseVisualShapeIndex=box_visual,
                                       basePosition=[0.5, 0.3, 0.5])
        
        # Add a sphere
        sphere_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.1)
        sphere_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[0, 1, 0, 1])
        self.sphere_id = p.createMultiBody(baseMass=0.5,
                                          baseCollisionShapeIndex=sphere_collision,
                                          baseVisualShapeIndex=sphere_visual,
                                          basePosition=[0.3, -0.4, 0.3])
        
    def detect_contact(self) -> bool:
        """Detect if the robot is in contact with any object"""
        # Check contacts with all objects
        contact_points = p.getContactPoints(self.robot_id)
        
        for contact in contact_points:
            # contact[9] is the contact normal force
            if len(contact) > 9 and abs(contact[9]) > self.contact_threshold:
                return True
        return False
    
    def get_joint_states(self) -> JointState:
        """Get current joint states of the robot"""
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        
        positions = [state[0] for state in joint_states]
        velocities = [state[1] for state in joint_states]
        applied_torques = [state[3] for state in joint_states]
        
        contact_detected = self.detect_contact()
        
        return JointState(
            positions=positions,
            velocities=velocities,
            applied_torques=applied_torques,
            timestamp=time.time(),
            contact_detected=contact_detected
        )
    
    def move_robot_random(self):
        """Move robot with random joint angles"""
        for joint in range(self.num_joints):
            if joint < 7:  # Only move the first 7 joints (arm joints)
                angle = np.random.uniform(-np.pi/2, np.pi/2)
                p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL, angle)
    
    def move_robot_trajectory(self, step: int):
        """Move robot following a predefined trajectory"""
        # Create a sinusoidal trajectory that will cause contacts
        for joint in range(min(7, self.num_joints)):
            # Different frequency for each joint to create complex motion
            freq = 0.01 + joint * 0.002
            amplitude = 0.8 + joint * 0.1
            
            angle = amplitude * np.sin(step * freq + joint * 0.5)
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL, angle)
    
    def collect_data(self, num_samples: int = 1000, data_file: str = "joint_data.csv"):
        """Collect joint state data with contact labels"""
        print(f"Collecting {num_samples} samples...")
        
        # Prepare CSV file
        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", data_file)
        
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = [f'joint_{i}_pos' for i in range(self.num_joints)] + \
                        [f'joint_{i}_vel' for i in range(self.num_joints)] + \
                        [f'joint_{i}_torque' for i in range(self.num_joints)] + \
                        ['contact_detected']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for step in range(num_samples):
                # Move robot
                if step % 100 < 50:  # Alternate between trajectory and random motion
                    self.move_robot_trajectory(step)
                else:
                    self.move_robot_random()
                
                # Step simulation
                p.stepSimulation()
                
                # Collect data every few steps
                if step % 5 == 0:
                    joint_state = self.get_joint_states()
                    
                    # Prepare data row
                    row = {}
                    for i in range(self.num_joints):
                        row[f'joint_{i}_pos'] = joint_state.positions[i]
                        row[f'joint_{i}_vel'] = joint_state.velocities[i]
                        row[f'joint_{i}_torque'] = joint_state.applied_torques[i]
                    
                    row['contact_detected'] = int(joint_state.contact_detected)
                    
                    writer.writerow(row)
                    
                    # Print progress
                    if step % 100 == 0:
                        contact_status = "CONTACT" if joint_state.contact_detected else "NO CONTACT"
                        print(f"Step {step}: {contact_status}")
                
                # Small delay for visualization
                if self.gui:
                    time.sleep(1./240.)
        
        print(f"Data collection complete! Saved to {filepath}")
    
    def run_demo(self, duration: int = 10):
        """Run a demo of the contact detection system"""
        print(f"Running demo for {duration} seconds...")
        
        start_time = time.time()
        step = 0
        
        while time.time() - start_time < duration:
            # Move robot
            self.move_robot_trajectory(step)
            
            # Step simulation
            p.stepSimulation()
            
            # Check for contact
            joint_state = self.get_joint_states()
            
            # Print contact status
            if step % 60 == 0:  # Print every 60 steps (~0.25 seconds)
                contact_status = "CONTACT DETECTED!" if joint_state.contact_detected else "No contact"
                print(f"Step {step}: {contact_status}")
            
            step += 1
            
            if self.gui:
                time.sleep(1./240.)
        
        print("Demo complete!")
    
    def cleanup(self):
        """Clean up the simulation"""
        if self.physics_client is not None:
            p.disconnect()

def main():
    """Main function to run the simulator"""
    print("Starting Contact Detection Simulator...")
    
    # Create simulator
    simulator = ContactDetectionSimulator(gui=True)
    
    try:
        # Setup simulation
        simulator.setup_simulation()
        
        # Run demo
        simulator.run_demo(duration=15)
        
        # Collect training data
        print("\nCollecting training data...")
        simulator.collect_data(num_samples=500)
        
    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        simulator.cleanup()

if __name__ == "__main__":
    main()
