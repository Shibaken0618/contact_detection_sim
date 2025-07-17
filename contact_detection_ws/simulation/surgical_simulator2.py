"""
Attempted realistic surgical simulator
"""

import os
import csv
import time
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pybullet as p
import pybullet_data


@dataclass
class SurgicalContact:
    contact_type: str
    force_magnitude: float
    contact_position: List[float]
    contact_normal: List[float]
    timestamp: float


class SurgicalContactSimulator:
    def __init__(self, gui=True):
        self.gui = gui
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.table_id = None
        self.end_effector_id = None

        self.instruments = {}
        self.anatomy_objects = {}
        self.constraints = {}

        self.contact_threshold = 0.05  # Increased threshold for more reliable detection
        self.contact_history = []

        self.scenario_type = "tissue_manipulation"

    def setup_surgical_environment(self):
        """Setup the surgical environment with improved stability"""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0/240.)

        # Improved physics parameters for stability
        p.setPhysicsEngineParameter(
            enableConeFriction=1,
            numSolverIterations=200,
            numSubSteps=4,
            contactBreakingThreshold=0.001,
            erp=0.8,
            contactERP=0.8,
            frictionERP=0.2,
            enableFileCaching=0
        )

        # Load basic environment
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

        # Set robot to a stable initial position
        num_joints = p.getNumJoints(self.robot_id)
        initial_angles = [0, -0.3, 0, -1.2, 0, 0.8, 0] 
        for i in range(min(7, num_joints)):
            p.resetJointState(self.robot_id, i, initial_angles[i] if i < len(initial_angles) else 0.0)

        self._setup_surgical_workspace()
        self._add_anatomical_structures()
        self._add_surgical_instruments()
        self._attach_end_effector()

        # Print all object IDs after setup for debugging
        print(f"[DEBUG] Final object IDs after setup:")
        print(f"  robot_id: {self.robot_id}")
        print(f"  end_effector_id: {self.end_effector_id}")
        print(f"  plane_id: {self.plane_id}")
        print(f"  table_id: {self.table_id}")
        print(f"  anatomy_objects: {self.anatomy_objects}")
        print(f"  instruments: {self.instruments}")

        # Let physics settle
        for _ in range(100):
            p.stepSimulation()

        print("Improved surgical environment setup complete.")

    def _setup_surgical_workspace(self):
        """Create a stable surgical table"""
        # Larger, more stable table
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.05])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.05], rgbaColor=[0.9, 0.9, 0.9, 1])
        self.table_id = p.createMultiBody(
            baseMass=0.0,  # Fixed table
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.6, 0, 0.05]
        )

        # Add table legs for visual realism (optional)
        for x_offset in [-0.4, 0.4]:
            for y_offset in [-0.3, 0.3]:
                leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.4])
                leg_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.4], rgbaColor=[0.7, 0.7, 0.7, 1])
                p.createMultiBody(
                    baseMass=0.0,
                    baseCollisionShapeIndex=leg_collision,
                    baseVisualShapeIndex=leg_visual,
                    basePosition=[0.6 + x_offset, y_offset, -0.35]
                )

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.8,
                cameraYaw=30,
                cameraPitch=-25,
                cameraTargetPosition=[0.6, 0, 0.2]
            )


    def _add_anatomical_structures(self):
        """Add anatomical structures with improved stability and proper positioning"""
        # Tissue - positioned where robot can easily reach
        tissue_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.06, 0.025])
        tissue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.06, 0.025], rgbaColor=[0.8, 0.4, 0.4, 0.8])
        self.anatomy_objects['tissue'] = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=tissue_collision,
            baseVisualShapeIndex=tissue_visual,
            basePosition=[0.6, 0, 0.125]  # Slightly lower, on table surface
        )

        # Better physics properties for tissue
        p.changeDynamics(self.anatomy_objects['tissue'], -1, 
                        lateralFriction=2.0,
                        rollingFriction=0.5, 
                        restitution=0.2, 
                        linearDamping=0.8,
                        angularDamping=0.8, 
                        contactStiffness=1000, 
                        contactDamping=100)
        
        # Organ - positioned for easy access
        organ_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        organ_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.6, 0.3, 0.3, 0.8])
        self.anatomy_objects['organ'] = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=organ_collision,
            baseVisualShapeIndex=organ_visual,
            basePosition=[0.45, 0.15, 0.14]  # Closer to robot reach
        )

        p.changeDynamics(self.anatomy_objects['organ'], -1, 
                        lateralFriction=1.5, 
                        restitution=0.2, 
                        linearDamping=0.7, 
                        angularDamping=0.7, 
                        contactStiffness=800, 
                        contactDamping=80)

        # Bone - positioned for drilling scenario
        bone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.03, height=0.08)
        bone_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=0.08, rgbaColor=[0.9, 0.9, 0.8, 1])
        bone_position = [0.75, -0.1, 0.14]  # Positioned for drilling access
        self.anatomy_objects['bone'] = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=bone_collision,
            baseVisualShapeIndex=bone_visual,
            basePosition=bone_position
        )

        p.changeDynamics(self.anatomy_objects['bone'], -1,
                        lateralFriction=1.0, 
                        restitution=0.1, 
                        linearDamping=0.9, 
                        angularDamping=0.9, 
                        contactStiffness=2000, 
                        contactDamping=200)

        print(f"[DEBUG] Bone created at position: {bone_position}, collision shape: cylinder, radius=0.03, height=0.08, id={self.anatomy_objects['bone']}")

        # Add lighter constraints that allow contact
        self._add_stability_constraints()
        
        # Print anatomy object IDs for debug
        print("[DEBUG] Anatomy object IDs:")
        for name, obj_id in self.anatomy_objects.items():
            print(f"  {name}: {obj_id}")
        
        # Let objects settle
        for _ in range(100):
            p.stepSimulation()
            
        print("Anatomical structures added with improved positioning for contact")

    def _add_stability_constraints(self):
        # Very light tissue constraint - allows movement for contact
        tissue_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['tissue'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.075],
            childFramePosition=[0, 0, -0.025]
        )
        p.changeConstraint(tissue_constraint, maxForce=20.0)  # Much lighter
        self.constraints['tissue'] = tissue_constraint
        
        # Light organ constraint
        organ_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['organ'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.15, 0.15, 0.09],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(organ_constraint, maxForce=10.0)  # Light constraint
        self.constraints['organ'] = organ_constraint

        # Light bone constraint
        bone_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['bone'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.15, -0.1, 0.09],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(bone_constraint, maxForce=25.0)  # Light constraint
        self.constraints['bone'] = bone_constraint


    def _add_surgical_instruments(self):
        """Add surgical instruments positioned away from anatomy"""
        # Grasper
        grasper_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.12)
        grasper_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.005, length=0.12, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.instruments['grasper'] = p.createMultiBody(
            baseMass=0.02,
            baseCollisionShapeIndex=grasper_collision,
            baseVisualShapeIndex=grasper_visual,
            basePosition=[0.3, 0.3, 0.15]
        )

        # Scissors
        scissors_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04])
        scissors_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04], rgbaColor=[0.7, 0.7, 0.9, 1])
        self.instruments['scissors'] = p.createMultiBody(
            baseMass=0.025,
            baseCollisionShapeIndex=scissors_collision,
            baseVisualShapeIndex=scissors_visual,
            basePosition=[0.3, -0.3, 0.13]
        )

        # Scalpel
        scalpel_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05])
        scalpel_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05], rgbaColor=[0.9, 0.9, 0.7, 1])
        self.instruments['scalpel'] = p.createMultiBody(
            baseMass=0.015,
            baseCollisionShapeIndex=scalpel_collision,
            baseVisualShapeIndex=scalpel_visual,
            basePosition=[0.9, 0, 0.12]
        )

        for instrument_id in self.instruments.values():
            p.changeDynamics(instrument_id, -1, 
                           lateralFriction=0.8, restitution=0.1, 
                           linearDamping=0.5, angularDamping=0.5)

        print("Surgical instruments added to workspace")
        # Print all instrument IDs for debug
        print("[DEBUG] Instrument IDs:")
        for name, obj_id in self.instruments.items():
            print(f"  {name}: {obj_id}")
        print("End effector attached.")
        print(f"[DEBUG] End effector ID: {self.end_effector_id}")
        # Print anatomy object IDs for clarity
        print("[DEBUG] Anatomy Object IDs:")
        for name, obj_id in self.anatomy_objects.items():
            print(f"  {name}: {obj_id}")

    def _attach_end_effector(self):
        """Attach end effector with improved design"""
        num_joints = p.getNumJoints(self.robot_id)
        end_effector_link = num_joints - 1

        # Larger probe for better contact detection
        probe_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
        probe_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[0.5, 0.5, 0.8, 1])

        end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
        end_effector_pos = end_effector_state[0]
        probe_pos = [end_effector_pos[0], end_effector_pos[1], end_effector_pos[2] - 0.05]

        self.end_effector_id = p.createMultiBody(
            baseMass=0.01,
            baseCollisionShapeIndex=probe_collision,
            baseVisualShapeIndex=probe_visual,
            basePosition=probe_pos
        )

        # Better physics properties for end effector
        p.changeDynamics(self.end_effector_id, -1, 
                        lateralFriction=1.0, 
                        restitution=0.1, 
                        linearDamping=0.5, 
                        angularDamping=0.5)

        self.end_effector_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=end_effector_link,
            childBodyUniqueId=self.end_effector_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -0.05],
            childFramePosition=[0, 0, 0]
        )

        print(f"[DEBUG] End effector attached with ID: {self.end_effector_id}")

    def detect_surgical_contact(self) -> List[SurgicalContact]:
        """Improved contact detection with better filtering"""
        contacts = []

        # Get all contact points in the simulation
        all_contacts = p.getContactPoints()
        
        # Filter for contacts involving robot or end effector
        robot_contacts = []
        for contact in all_contacts:
            if len(contact) < 10:
                continue
                
            bodyA, bodyB = contact[1], contact[2]
            
            # Check if either body is the robot or end effector
            if (bodyA == self.robot_id or bodyA == self.end_effector_id or 
                bodyB == self.robot_id or bodyB == self.end_effector_id):
                robot_contacts.append(contact)

        print(f"[DEBUG] Found {len(robot_contacts)} robot-related contacts out of {len(all_contacts)} total contacts")

        for contact in robot_contacts:
            bodyA, bodyB = contact[1], contact[2]
            linkA, linkB = contact[3], contact[4]
            contact_pos = contact[5]
            contact_normal = contact[7]
            normal_force = contact[9]

            # Determine the 'other' body (not robot/end effector)
            other_body = None
            if bodyA == self.robot_id or bodyA == self.end_effector_id:
                other_body = bodyB
            elif bodyB == self.robot_id or bodyB == self.end_effector_id:
                other_body = bodyA
            else:
                continue  # Skip if neither is robot/end effector

            contact_type = self._classify_contact(other_body)

            print(f"[DEBUG] Contact: robot/ee with body {other_body}, type={contact_type}, force={normal_force:.3f}")
            
            # Only report significant contacts
            if abs(normal_force) > self.contact_threshold:
                surgical_contact = SurgicalContact(
                    contact_type=contact_type,
                    force_magnitude=abs(normal_force),
                    contact_position=list(contact_pos),
                    contact_normal=list(contact_normal),
                    timestamp=time.time()
                )
                contacts.append(surgical_contact)

        return contacts

    def _classify_contact(self, body_id) -> str:
        """Classify the type of contact with improved logic"""
        # Print all known IDs for debugging
        print(f"[DEBUG] Classifying body_id: {body_id}")
        print(f"[DEBUG] Known anatomy IDs: tissue={self.anatomy_objects.get('tissue')}, organ={self.anatomy_objects.get('organ')}, bone={self.anatomy_objects.get('bone')}")
        print(f"[DEBUG] Known other IDs: table={self.table_id}, plane={self.plane_id}")
        
        # Check anatomy objects
        if body_id == self.anatomy_objects.get('tissue'):
            return 'tissue'
        elif body_id == self.anatomy_objects.get('organ'):
            return 'organ'  
        elif body_id == self.anatomy_objects.get('bone'):
            return 'bone'
        elif body_id == self.table_id:
            return 'table'
        elif body_id == self.plane_id:
            return 'ground'
        elif body_id in self.instruments.values():
            return 'instrument'
        else:
            # Additional debug info for unknown contacts
            print(f"[DEBUG] Unknown body_id {body_id} - not matching any known objects")
            return 'unknown'

    def get_end_effector_position(self):
        """Get end effector position"""
        if self.end_effector_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.end_effector_id)
            return pos
        else:
            num_joints = p.getNumJoints(self.robot_id)
            link_state = p.getLinkState(self.robot_id, num_joints-1)
            return link_state[0]

    def execute_surgical_scenario(self, scenario_type: str = "tissue_manipulation"):
        """Execute surgical scenario with improved trajectories"""
        self.scenario_type = scenario_type

        if scenario_type == "tissue_manipulation":
            return self._tissue_manipulation_scenario()
        elif scenario_type == "organ_examination":
            return self._organ_examination_scenario()
        elif scenario_type == "bone_drilling":
            return self._bone_drilling_scenario()
        else:
            return self._default_scenario()


    def _tissue_manipulation_scenario(self):
        """Improved tissue manipulation with direct contact approach"""
        target_positions = [
            [0.4, 0, 0.5],      # start position
            [0.6, 0, 0.35],     # move over tissue
            [0.6, 0, 0.25],     # approach tissue
            [0.6, 0, 0.20],     # closer to tissue
            [0.6, 0, 0.17],     # contact tissue
            [0.6, 0, 0.15],     # press into tissue
            [0.6, 0.03, 0.15],  # manipulation motion
            [0.6, -0.03, 0.15], # continue manipulation
            [0.6, 0, 0.14],     # deeper press
            [0.6, 0.02, 0.14],  # more manipulation
            [0.6, -0.02, 0.14], # continue
            [0.6, 0, 0.15],     # lift slightly
            [0.6, 0, 0.25],     # retract
            [0.6, 0, 0.4],      # final retract
        ]

        return self._execute_waypoint_motion(target_positions, speed_factor=0.4)


    def _organ_examination_scenario(self):
        """Improved organ examination with proper approach"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.45, 0.15, 0.35], # move over organ
            [0.45, 0.15, 0.25], # approach organ
            [0.45, 0.15, 0.20], # closer to organ
            [0.45, 0.15, 0.18], # contact organ
            [0.45, 0.15, 0.16], # press into organ
            [0.45, 0.13, 0.16], # examination motion
            [0.45, 0.17, 0.16], # continue examination
            [0.45, 0.15, 0.15], # deeper examination
            [0.45, 0.14, 0.15], # precise movement
            [0.45, 0.16, 0.15], # continue
            [0.45, 0.15, 0.18], # lift slightly
            [0.45, 0.15, 0.25], # retract
            [0.45, 0.15, 0.4],  # final retract
        ]

        return self._execute_waypoint_motion(target_positions, speed_factor=0.4)


    def _bone_drilling_scenario(self):
        """Improved bone drilling with direct contact"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.75, -0.1, 0.35], # move over bone
            [0.75, -0.1, 0.25], # approach bone
            [0.75, -0.1, 0.20], # closer to bone
            [0.75, -0.1, 0.18], # contact bone
            [0.75, -0.1, 0.16], # press into bone
            [0.75, -0.08, 0.16], # drilling motion
            [0.75, -0.12, 0.16], # continue drilling
            [0.75, -0.1, 0.15], # deeper drilling
            [0.75, -0.09, 0.15], # precise drilling
            [0.75, -0.11, 0.15], # continue
            [0.75, -0.1, 0.18], # lift slightly
            [0.75, -0.1, 0.25], # retract
            [0.75, -0.1, 0.4],  # final retract
        ]

        result = self._execute_waypoint_motion(target_positions, speed_factor=0.3)
        # Debug: print if any bone contact detected in this scenario
        if any(d.get('contact_type') == 'bone' and d.get('contact_detected') for d in result):
            print("[DEBUG] Bone contact detected during bone drilling scenario.")
        else:
            print("[DEBUG] No bone contact detected during bone drilling scenario.")
        return result


    def _default_scenario(self):
        return self._tissue_manipulation_scenario()


    def _execute_waypoint_motion(self, waypoints: List[List[float]], speed_factor: float = 1.0):
        """Execute waypoint motion with improved control"""
        num_joints = p.getNumJoints(self.robot_id)
        joint_data = []

        for waypoint_idx, waypoint in enumerate(waypoints):
            print(f"Moving to waypoint {waypoint_idx + 1}/{len(waypoints)}: {waypoint}")

            try:
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    num_joints - 1,
                    waypoint,
                    maxNumIterations=500,
                    residualThreshold=0.0001,
                    jointDamping=[0.05] * num_joints
                )
            except Exception as e:
                print(f"IK failed for waypoint {waypoint}: {e}")
                continue

            steps = int(150 * speed_factor)  # More steps for smoother motion
            for step in range(steps):
                for joint_idx in range(min(7, len(joint_angles))):
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        joint_angles[joint_idx],
                        maxVelocity=0.1,  # Slower for more control
                        force=30.0
                    )

                p.stepSimulation()

                # Collect data more frequently
                if step % 3 == 0:
                    joint_states = p.getJointStates(self.robot_id, range(min(7, num_joints)))
                    contacts = self.detect_surgical_contact()

                    positions = [state[0] for state in joint_states]
                    velocities = [state[1] for state in joint_states]
                    torques = [state[3] for state in joint_states]

                    # Pad to 7 joints
                    while len(positions) < 7:
                        positions.append(0.0)
                        velocities.append(0.0)
                        torques.append(0.0)

                    contact_detected = len(contacts) > 0
                    contact_force = max([c.force_magnitude for c in contacts]) if contacts else 0.0
                    contact_type = contacts[0].contact_type if contacts else 'none'

                    joint_data.append({
                        'positions': positions[:7],
                        'velocities': velocities[:7],
                        'torques': torques[:7],
                        'contact_detected': contact_detected,
                        'contact_force': contact_force,
                        'contact_type': contact_type,
                        'timestamp': time.time()
                    })

                    if contact_detected:
                        ee_pos = self.get_end_effector_position()
                        print(f"  -> Contact: {contact_type}, Force: {contact_force:.3f}, EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

                if self.gui:
                    time.sleep(1.0 / 240.0)

        return joint_data

    def collect_surgical_data(self, num_scenarios: int = 9, data_file: str = "surgical_data.csv"):
        """Collect surgical data with improved scenarios"""
        print(f"Collecting surgical data with {num_scenarios} scenarios...")

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", data_file)

        all_data = []
        scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]

        for scenario_idx in range(num_scenarios):
            scenario = scenarios[scenario_idx % len(scenarios)]
            print(f"\n--- Executing {scenario} scenario {scenario_idx + 1}/{num_scenarios} ---")

            # Reset any displaced objects before each scenario
            self._reset_anatomy_positions()

            scenario_data = self.execute_surgical_scenario(scenario)

            for data_point in scenario_data:
                data_point['scenario'] = scenario
                all_data.append(data_point)

        # Save data
        with open(filepath, 'w', newline='') as csvfile:
            if not all_data:
                print("No data collected")
                return

            fieldnames = []
            for i in range(7):
                fieldnames.extend([f'joint_{i}_pos', f'joint_{i}_vel', f'joint_{i}_torque'])
            fieldnames.extend(['contact_detected', 'contact_force', 'contact_type', 'scenario'])

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for data_point in all_data:
                row = {}
                for i in range(7):
                    row[f'joint_{i}_pos'] = data_point['positions'][i]
                    row[f'joint_{i}_vel'] = data_point['velocities'][i]
                    row[f'joint_{i}_torque'] = data_point['torques'][i]

                row['contact_detected'] = int(data_point['contact_detected'])
                row['contact_force'] = data_point['contact_force']
                row['contact_type'] = data_point['contact_type']
                row['scenario'] = data_point['scenario']

                writer.writerow(row)

        print(f'Surgical data collection complete! Saved {len(all_data)} samples to {filepath}')

        # Summary stats
        contact_data = [d for d in all_data if d['contact_detected']]
        print(f"\nData Summary:")
        print(f"  Total samples: {len(all_data)}")
        print(f"  Contact samples: {len(contact_data)} ({len(contact_data)/len(all_data)*100:.1f}%)")
        
        contact_types = {}
        for d in contact_data:
            contact_types[d['contact_type']] = contact_types.get(d['contact_type'], 0) + 1
        
        print(f"  Contact types: {contact_types}")


    def _reset_anatomy_positions(self):
        """Reset anatomical objects to their original positions"""
        if 'tissue' in self.anatomy_objects:
            p.resetBasePositionAndOrientation(self.anatomy_objects['tissue'], [0.6, 0, 0.125], [0, 0, 0, 1])
        if 'organ' in self.anatomy_objects:
            p.resetBasePositionAndOrientation(self.anatomy_objects['organ'], [0.45, 0.15, 0.14], [0, 0, 0, 1])
        if 'bone' in self.anatomy_objects:
            p.resetBasePositionAndOrientation(self.anatomy_objects['bone'], [0.75, -0.1, 0.14], [0, 0, 0, 1])
        
        # Let physics settle
        for _ in range(50):
            p.stepSimulation()


    def run_surgical_demo(self, duration: int = 60):
        """Run surgical demo with improved scenarios"""
        print(f"Running surgical demo for {duration} seconds...")
        scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]
        scenario_duration = duration // len(scenarios)

        for scenario in scenarios:
            print(f"\n--- {scenario.upper()} SCENARIO ---")
            
            # Reset positions before each scenario
            self._reset_anatomy_positions()
            
            scenario_data = self.execute_surgical_scenario(scenario)

            contacts = [d for d in scenario_data if d['contact_detected']]
            if contacts:
                avg_force = np.mean([c['contact_force'] for c in contacts])
                contact_types = {}
                for c in contacts:
                    contact_types[c['contact_type']] = contact_types.get(c['contact_type'], 0) + 1
                
                print(f"  Contacts detected: {len(contacts)}")
                print(f"  Average force: {avg_force:.3f}")
                print(f"  Contact types: {contact_types}")
            else:
                print("  No contacts detected in this scenario")

        print("\nSurgical demo complete!")


    def cleanup(self):
        """Clean up the simulation"""
        if hasattr(self, 'end_effector_constraint'):
            try:
                p.removeConstraint(self.end_effector_constraint)
            except:
                pass

        for constraint_id in self.constraints.values():
            try:
                p.removeConstraint(constraint_id)
            except:
                pass

        if self.physics_client is not None:
            p.disconnect()


def main():
    print("Starting Fixed Surgical Contact Detection Simulator...")
    simulator = SurgicalContactSimulator(gui=True)

    try:
        simulator.setup_surgical_environment()
        simulator.run_surgical_demo(duration=60)

        print("\nCollecting surgical training data...")
        simulator.collect_surgical_data(num_scenarios=18)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    main()