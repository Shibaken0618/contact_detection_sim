"""
A more realistic surgical contact detection simulator - FIXED VERSION
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

        self.contact_threshold = 0.015  # Reduced threshold for better detection
        self.contact_history = []

        self.scenario_type = "tissue_manipulation"

    def setup_surgical_environment(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0/240.)

        # Enhanced physics parameters
        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setPhysicsEngineParameter(numSubSteps=4)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.001)

        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

        # Initialize robot to a better starting position
        num_joints = p.getNumJoints(self.robot_id)
        initial_angles = [0, -0.5, 0, -1.5, 0, 1.0, 0]  # Better starting pose
        for i in range(min(7, num_joints)):
            p.resetJointState(self.robot_id, i, initial_angles[i] if i < len(initial_angles) else 0.0)

        self._setup_surgical_workspace()
        self._add_anatomical_structures()
        self._add_surgical_instruments()
        self._attach_end_effector()

        print("Surgical environment setup complete.")

    def _setup_surgical_workspace(self):
        # Larger, more stable table
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.4, 0.03])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.4, 0.03], rgbaColor=[0.9, 0.9, 0.9, 1])
        self.table_id = p.createMultiBody(
            baseMass=0.0,  # Static table
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.6, 0, 0.06]
        )

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-20,
                cameraTargetPosition=[0.6, 0, 0.2]
            )

    def _add_anatomical_structures(self):
        # FIXED: Tissue with soft constraint - can be moved but returns to position
        tissue_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.06, 0.02])
        tissue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.06, 0.02], rgbaColor=[0.8, 0.4, 0.4, 0.9])
        self.anatomy_objects['tissue'] = p.createMultiBody(
            baseMass=0.1,  # Reduced mass for softer behavior
            baseCollisionShapeIndex=tissue_collision,
            baseVisualShapeIndex=tissue_visual,
            basePosition=[0.6, 0, 0.11],
            baseOrientation=[0, 0, 0, 1]
        )

        # Softer tissue dynamics with stronger restoring force
        p.changeDynamics(self.anatomy_objects['tissue'], -1, 
                        lateralFriction=1.8, rollingFriction=0.8, restitution=0.3, 
                        linearDamping=0.95, angularDamping=0.95, 
                        contactStiffness=200, contactDamping=25)  # Reduced for softer contact
        
        # Organ positioned for better contact
        organ_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.04)
        organ_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.04, rgbaColor=[0.6, 0.3, 0.3, 0.9])
        self.anatomy_objects['organ'] = p.createMultiBody(
            baseMass=0.04,  # Slightly reduced mass
            baseCollisionShapeIndex=organ_collision,
            baseVisualShapeIndex=organ_visual,
            basePosition=[0.5, 0.15, 0.13]
        )

        p.changeDynamics(self.anatomy_objects['organ'], -1, 
                        lateralFriction=1.3, restitution=0.15, 
                        linearDamping=0.85, angularDamping=0.85, 
                        contactStiffness=400, contactDamping=40)  # Reduced for softer contact

        # FIXED: Bone repositioned to avoid contact with tissue
        bone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.025, height=0.08)
        bone_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.025, length=0.08, rgbaColor=[0.9, 0.9, 0.8, 1])
        self.anatomy_objects['bone'] = p.createMultiBody(
            baseMass=0.25,  # Reduced mass for better response
            baseCollisionShapeIndex=bone_collision,
            baseVisualShapeIndex=bone_visual,
            basePosition=[0.45, -0.15, 0.13]
        )

        p.changeDynamics(self.anatomy_objects['bone'], -1,
                        lateralFriction=0.9, restitution=0.25, 
                        linearDamping=0.8, angularDamping=0.8, 
                        contactStiffness=2500, contactDamping=250)  # Reduced for softer contact

        # Add constraints to prevent objects from moving
        self._add_stability_constraints()
        print("Anatomical structures added and anchored to table")

    def _add_stability_constraints(self):
        """Add constraints to keep anatomical objects stable but allow tissue deformation"""
        # FIXED: Stronger spring constraint for tissue with position restoration
        tissue_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['tissue'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.05],
            childFramePosition=[0, 0, -0.02]
        )
        p.changeConstraint(tissue_constraint, maxForce=15.0)  # Stronger restoring force
        self.constraints['tissue'] = tissue_constraint

        # Additional angular constraint to prevent tissue rotation
        tissue_angular_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['tissue'],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.05],
            childFramePosition=[0, 0, -0.02]
        )
        p.changeConstraint(tissue_angular_constraint, maxForce=8.0)  # Moderate angular restraint
        self.constraints['tissue_angular'] = tissue_angular_constraint

        # Stronger constraint for organ with better restoration
        organ_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['organ'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.1, 0.15, 0.07],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(organ_constraint, maxForce=6.0)  # Stronger restoring force
        self.constraints['organ'] = organ_constraint

        # FIXED: Stronger bone constraint with position restoration
        bone_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['bone'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.15, -0.15, 0.07],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(bone_constraint, maxForce=25.0)  # Much stronger restoring force
        self.constraints['bone'] = bone_constraint

        # Additional angular constraint for bone to prevent rotation
        bone_angular_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['bone'],
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.15, -0.15, 0.07],
            childFramePosition=[0, 0, 0]
        )
        p.changeConstraint(bone_angular_constraint, maxForce=15.0)  # Strong angular restraint
        self.constraints['bone_angular'] = bone_angular_constraint


    def _add_surgical_instruments(self):
        # FIXED: Make instruments more visible and position them properly
        # Grasper - visible on table
        grasper_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.12)
        grasper_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.005, length=0.12, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.instruments['grasper'] = p.createMultiBody(
            baseMass=0.02,
            baseCollisionShapeIndex=grasper_collision,
            baseVisualShapeIndex=grasper_visual,
            basePosition=[0.4, 0.2, 0.15]
        )

        # Scissors - visible on table
        scissors_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04])
        scissors_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04], rgbaColor=[0.7, 0.7, 0.9, 1])
        self.instruments['scissors'] = p.createMultiBody(
            baseMass=0.025,
            baseCollisionShapeIndex=scissors_collision,
            baseVisualShapeIndex=scissors_visual,
            basePosition=[0.4, -0.2, 0.13]
        )

        # Scalpel - new instrument
        scalpel_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05])
        scalpel_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05], rgbaColor=[0.9, 0.9, 0.7, 1])
        self.instruments['scalpel'] = p.createMultiBody(
            baseMass=0.015,
            baseCollisionShapeIndex=scalpel_collision,
            baseVisualShapeIndex=scalpel_visual,
            basePosition=[0.8, 0, 0.12]
        )

        # Set proper dynamics for all instruments
        for instrument_id in self.instruments.values():
            p.changeDynamics(instrument_id, -1, 
                           lateralFriction=0.8, restitution=0.1, 
                           linearDamping=0.5, angularDamping=0.5)

        print("Surgical instruments added to workspace")

    def _attach_end_effector(self):
        num_joints = p.getNumJoints(self.robot_id)
        end_effector_link = num_joints - 1
        
        # FIXED: Smaller, more precise probe
        probe_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.003, height=0.05)
        probe_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.003, length=0.05, rgbaColor=[0.5, 0.5, 0.8, 1])

        end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
        end_effector_pos = end_effector_state[0]
        probe_pos = [end_effector_pos[0], end_effector_pos[1], end_effector_pos[2] - 0.03]

        self.end_effector_id = p.createMultiBody(
            baseMass=0.001,
            baseCollisionShapeIndex=probe_collision,
            baseVisualShapeIndex=probe_visual,
            basePosition=probe_pos
        )

        # More rigid attachment
        self.end_effector_constraint = p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=end_effector_link,
            childBodyUniqueId=self.end_effector_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, -0.03],
            childFramePosition=[0, 0, 0]
        )

        print("End effector attached to robot")

    def detect_surgical_contact(self) -> List[SurgicalContact]:
        contacts = []

        contact_sources = [self.robot_id]
        if self.end_effector_id is not None:
            contact_sources.append(self.end_effector_id)

        for source_id in contact_sources:
            contact_points = p.getContactPoints(source_id)

            for contact in contact_points:
                if len(contact) < 10:
                    continue

                bodyA, bodyB = contact[1], contact[2]
                contact_pos = contact[5]
                contact_normal = contact[7]
                normal_force = contact[9]

                other_body = bodyB if bodyA == source_id else bodyA
                contact_type = self._classify_contact(other_body)

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
        if body_id == self.anatomy_objects.get('tissue'):
            return 'tissue'
        elif body_id == self.anatomy_objects.get('organ'):
            return 'organ'
        elif body_id == self.anatomy_objects.get('bone'):
            return 'bone'
        elif body_id in self.instruments.values():
            return 'instrument'
        elif body_id == self.table_id:
            return 'table'
        elif body_id == self.plane_id:
            return 'ground'
        else:
            return 'unknown'

    def get_end_effector_position(self):
        if self.end_effector_id is not None:
            pos, _ = p.getBasePositionAndOrientation(self.end_effector_id)
            return pos
        else:
            num_joints = p.getNumJoints(self.robot_id)
            link_state = p.getLinkState(self.robot_id, num_joints-1)
            return link_state[0]

    def execute_surgical_scenario(self, scenario_type: str = "tissue_manipulation"):
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
        # FIXED: Better waypoints for tissue interaction
        target_positions = [
            [0.4, 0, 0.4],      # start position
            [0.6, 0, 0.3],      # move over table
            [0.6, 0, 0.18],     # approach tissue
            [0.6, 0, 0.13],     # contact tissue
            [0.6, 0.03, 0.13],  # gentle manipulation
            [0.6, -0.03, 0.13], # continue manipulation
            [0.6, 0, 0.13],     # return to center
            [0.6, 0, 0.18],     # lift slightly
            [0.6, 0.02, 0.13],  # more contact
            [0.6, 0, 0.3],      # retract
        ]

        return self._execute_waypoint_motion(target_positions, speed_factor=0.6)

    def _organ_examination_scenario(self):
        # FIXED: New scenario to contact the organ
        target_positions = [
            [0.4, 0, 0.4],      # start
            [0.5, 0.15, 0.3],   # move over organ
            [0.5, 0.15, 0.2],   # approach organ
            [0.5, 0.15, 0.17],  # contact organ
            [0.5, 0.13, 0.17],  # examine
            [0.5, 0.17, 0.17],  # continue examination
            [0.5, 0.15, 0.17],  # center
            [0.5, 0.15, 0.25],  # retract
        ]

        return self._execute_waypoint_motion(target_positions, speed_factor=0.4)

    def _bone_drilling_scenario(self):
        # FIXED: Updated waypoints for bone position away from tissue
        target_positions = [
            [0.4, 0, 0.4],        # start position
            [0.45, -0.15, 0.35],  # move over bone (updated to new position)
            [0.45, -0.15, 0.25],  # approach bone
            [0.45, -0.15, 0.20],  # get closer
            [0.45, -0.15, 0.17],  # contact bone (adjusted for new position)
            [0.45, -0.13, 0.17],  # drilling motion 1
            [0.45, -0.17, 0.17],  # drilling motion 2
            [0.45, -0.15, 0.17],  # back to center
            [0.43, -0.15, 0.17],  # side drilling
            [0.47, -0.15, 0.17],  # other side drilling
            [0.45, -0.15, 0.17],  # center again
            [0.45, -0.15, 0.16],  # press down more for contact
            [0.45, -0.15, 0.25],  # retract
        ]

        print(f"Bone drilling scenario - bone position: [0.45, -0.15, 0.13]")
        return self._execute_waypoint_motion(target_positions, speed_factor=0.3)


    def _default_scenario(self):
        return self._tissue_manipulation_scenario()

    def _execute_waypoint_motion(self, waypoints: List[List[float]], speed_factor: float = 1.0):
        num_joints = p.getNumJoints(self.robot_id)
        joint_data = []

        for waypoint_idx, waypoint in enumerate(waypoints):
            print(f"Moving to waypoint {waypoint_idx + 1}/{len(waypoints)}: {waypoint}")

            try:
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    num_joints - 1,
                    waypoint,
                    maxNumIterations=300,
                    residualThreshold=0.001,
                    jointDamping=[0.1] * num_joints
                )
            except:
                print(f"IK failed for waypoint {waypoint}")
                continue

            steps = int(120 * speed_factor)
            for step in range(steps):
                for joint_idx in range(min(7, len(joint_angles))):
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        joint_angles[joint_idx],
                        maxVelocity=0.2,  # Slower for more precise control
                        force=50.0
                    )

                p.stepSimulation()

                if step % 5 == 0:  # More frequent data collection
                    joint_states = p.getJointStates(self.robot_id, range(min(7, num_joints)))
                    contacts = self.detect_surgical_contact()

                    positions = [state[0] for state in joint_states]
                    velocities = [state[1] for state in joint_states]
                    torques = [state[3] for state in joint_states]

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

    def collect_surgical_data(self, num_scenarios: int = 5, data_file: str = "surgical_data.csv"):
        print(f"Collecting surgical data with {num_scenarios} scenarios...")

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", data_file)

        all_data = []
        # FIXED: Include all three scenarios
        scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]

        for scenario_idx in range(num_scenarios):
            scenario = scenarios[scenario_idx % len(scenarios)]
            print(f"\n--- Executing {scenario} scenario {scenario_idx + 1}/{num_scenarios} ---")

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

        # Print summary statistics
        contact_data = [d for d in all_data if d['contact_detected']]
        print(f"\nData Summary:")
        print(f"  Total samples: {len(all_data)}")
        print(f"  Contact samples: {len(contact_data)} ({len(contact_data)/len(all_data)*100:.1f}%)")
        
        contact_types = {}
        for d in contact_data:
            contact_types[d['contact_type']] = contact_types.get(d['contact_type'], 0) + 1
        
        print(f"  Contact types: {contact_types}")

    def run_surgical_demo(self, duration: int = 30):
        print(f"Running surgical demo for {duration} seconds...")
        scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]
        scenario_duration = duration // len(scenarios)

        for scenario in scenarios:
            print(f"\n--- {scenario.upper()} SCENARIO ---")
            
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
        if hasattr(self, 'end_effector_constraint'):
            try:
                p.removeConstraint(self.end_effector_constraint)
            except:
                pass

        # Clean up all constraints including new angular constraints
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
        simulator.run_surgical_demo(duration=45)

        print("\nCollecting surgical training data...")
        simulator.collect_surgical_data(num_scenarios=18)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        simulator.cleanup()


if __name__ == "__main__":
    main()