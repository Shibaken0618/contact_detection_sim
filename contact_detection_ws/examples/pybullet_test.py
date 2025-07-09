#!/usr/bin/env python3
"""
Simple PyBullet test script to verify installation and basic functionality.
This creates a simple scene with a robot arm and demonstrates basic physics.
"""

import pybullet as p
import pybullet_data
import time
import numpy as np

def test_pybullet():
    """Test basic PyBullet functionality with a simple robot arm"""
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)  # Use p.DIRECT for headless
    
    # Set up the simulation
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load ground plane
    plane_id = p.loadURDF("plane.urdf")
    
    # Load a simple robot (we'll use Kuka for now, later we'll create our own)
    robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])
    
    # Get number of joints
    num_joints = p.getNumJoints(robot_id)
    print(f"Robot has {num_joints} joints")
    
    # Print joint info
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        print(f"Joint {i}: {joint_info[1].decode('utf-8')}, Type: {joint_info[2]}")
    
    # Simple animation - move joints
    print("Starting simulation...")
    for step in range(1000):
        # Set joint positions (simple sinusoidal motion)
        for joint in range(7):  # Kuka has 7 joints
            angle = 0.5 * np.sin(step * 0.01 + joint * 0.5)
            p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, angle)
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1./240.)  # 240 Hz
        
        # Every 100 steps, print joint states
        if step % 100 == 0:
            joint_states = p.getJointStates(robot_id, range(7))
            positions = [state[0] for state in joint_states]
            print(f"Step {step}: Joint positions: {[f'{pos:.3f}' for pos in positions]}")
    
    print("Test completed successfully!")
    p.disconnect()

if __name__ == "__main__":
    test_pybullet()
