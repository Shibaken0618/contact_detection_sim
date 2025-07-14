"""
A more realistic surgical contact detection simulator
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

        self.instruments = {}
        self.anatomy_objects = {}
        self.constraints = {}

        self.contact_threshold = 0.05
        self.contact_history = []

        self.scenario_type = "tissue_manipulation"  # suturing, cutting, etc


    def setup_surgical_environment(self):
        if self.gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0/240.)

        p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(numSolverIterations=10)
        p.setPhysicsEngineParameter(numSubSteps=1)

        self.plane_id = p.loadURDF("plane.urdf")

        self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

        num_joints = p.getNumJoints(self.robot_id)
        for i in range(min(7, num_joints)):
            p.resetJointState(self.robot_id, i, 0.0)

        self._setup_surgical_workspace()
        self._add_anatomical_structures()
        self._add_surgical_instruments()

        print("Surgical environment setup complete.")


    def _setup_surgical_workspace(self):
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.02])
        table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.3, 0.02], rgbaColor=[0.9, 0.9, 0.9, 1])
        self.table_id = p.createMultiBody(
            baseMass = 0.0,
            baseCollisionShapeIndex = table_collision,
            baseVisualShapeIndex = table_visual,
            basePosition = [0.4, 0, 0.08]
        )

        if self.gui:
            p.resetDebugVisualizerCamera(
                cameraDistance = 0.8,
                cameraYaw = 45,
                cameraPitch = 30,
                cameraTargetPosition = [0.4, 0, 0.2]
            )


    def _add_anatomical_structures(self):
        tissue_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.05, 0.02])
        tissue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.05, 0.02], rgbaColor=[0.8, 0.4, 0.4, 0.9])
        self.anatomy_objects['tissue'] = p.createMultiBody(
            baseMass = 0.1,
            baseCollisionShapeIndex = tissue_collision,
            baseVisualShapeIndex = tissue_visual,
            basePosition=[0.4, 0, 0.12]
        )

        p.changeDynamics(self.anatomy_objects['tissue'], -1, 
                         lateralFriction=0.9, rollingFriction=0.1, restitution=0.1, 
                         linearDamping=0.8, angularDamping=0.8, contactStiffness=1000, contactDamping=100)
        
        organ_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.03)
        organ_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.03, rgbaColor=[0.6, 0.3, 0.3, 0.9])
        self.anatomy_objects['organ'] = p.createMultiBody(
            baseMass = 0.05,
            baseCollisionShapeIndex = organ_collision,
            baseVisualShapeIndex = organ_visual,
            basePosition = [0.3, 0.15, 0.13]
        )

        p.changeDynamics(self.anatomy_objects['organ'], -1, 
                         lateralFriction=0.8, spinningFriction=0.1, rollingFriction=0.1, restitution=0.1, 
                         linearDamping=0.9, angularDamping=0.9, contactStiffness=800, contactDamping=80)

        bone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.08)
        bone_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.08, rgbaColor=[0.9, 0.9, 0.8, 1])
        self.anatomy_objects['bone'] = p.createMultiBody(
            baseMass = 0.2,
            baseCollisionShapeIndex = bone_collision,
            baseVisualShapeIndex = bone_visual,
            basePosition = [0.5, -0.1, 0.14]
        )

        p.changeDynamics(self.anatomy_objects['bone'], -1,
                         lateralFriction=0.6, spinningFriction=0.2, rollingFriction=0.2, restitution=0.2,
                         linearDamping=0.5, angularDamping=0.5, contactStiffness=2000, contactDamping=200)
    
        self._add_stability_constraints()

    def _add_stability_constraints(self):
        for obj_name, obj_id in self.anatomy_objects.items():
            constraint = p.createConstraint(
                parentBodyUniqueId = self.table_id, parentLinkIndex=-1, childBodyUniqueId=obj_id, childLinkIndex=-1,
                jointType=p.JOINT_POINT2POINT, jointAxis=[0,0,0], parentFramePosition=[0, 0, 0.5], childFramePosition=[0,0,0]
            )

            p.changeConstraint(constraint, maxForce=10.0)
            self.constraints[obj_name] = constraint


    def _add_surgical_instruments(self):
        grasper_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.002, height=0.05)
        grasper_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.002, length=0.05, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.instruments['grasper'] = p.createMultiBody(
            baseMass = 0.005,
            baseCollisionShapeIndex = grasper_collision,
            baseVisualShapeIndex = grasper_visual,
            basePosition = [0.45, 0.1, 0.2]
        )

        scissors_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.001, 0.01, 0.025])
        scissors_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.001, 0.01, 0.025], rgbaColor=[0.7, 0.7, 0.9, 1])
        self.instruments['scissors'] = p.createMultiBody(
            baseMass = 0.01,
            baseCollisionShapeIndex = scissors_collision,
            baseVisualShapeIndex = scissors_visual,
            basePosition = [0.35, 0.2, 0.2]
        )

        for instrument_id in self.instruments.values():
            p.changeDynamics(instrument_id, -1, lateralFriction=0.7, spinningFriction=0.1,
                             rollingFriction=0.1, restitution=0.1, linearDamping=0.3, angularDamping=0.3)


    def detect_surgical_contact(self) -> List[SurgicalContact]:
        contacts = []

        contact_points = p.getContactPoints(self.robot_id)
        for contact in contact_points:
            if len(contact) < 10:
                continue

            bodyA, bodyB = contact[1], contact[2]
            linkA, linkB = contact[3], contact[4]
            contact_pos = contact[5]
            contact_normal = contact[7]
            normal_force = contact[9]

            contact_type = self._classify_contact(bodyB)

            if abs(normal_force) > self.contact_threshold:
                surgical_contact = SurgicalContact(
                    contact_type = contact_type,
                    force_magnitude = abs(normal_force),
                    contact_position = list(contact_pos),
                    contact_normal = list(contact_normal),
                    timestamp = time.time()
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
        else:
            return 'unknown'
        

    def execute_surgical_scenario(self, scenario_type: str="tissue_manipulation"):
        self.scenario_type = scenario_type

        if scenario_type == "tissue_manipulaiton":
            return self._tissue_manipulation_scenario()
        elif scenario_type == "suturing":
            return self._suturing_scenario()
        elif scenario_type == "cutting":
            return self._cutting_scenario()
        else:
            return self._default_scenario()
        

    def _tissue_manipulation_scenario(self):
        target_positions = [
            [0.3, 0, 0.3],   # approach
            [0.4, 0, 0.15],   # contact tissue
            [0.4, 0.03, 0.15],   # manipulate
            [0.4, -0.03, 0.15],   # continue manipulation
            [0.3, 0, 0.3],   # retract
        ]

        return self._execute_waypoint_motion(target_positions)
    

    def _suturing_scenario(self):
        suture_points = []
        for i in range(5):
            x = 0.35 + i * 0.015
            y = 0.015 * math.sin(i*math.pi / 2)
            z = 0.14 + 0.005 * math.cos(i * math.pi / 2)
            suture_points.append([x, y, z])

        return self._execute_waypoint_motion(suture_points, speed_factor=0.5)
    

    def _cutting_scenario(self):
        cut_points = []
        for i in range(8):
            x = 0.35 + i * 0.008
            y = 0
            z = 0.14
            cut_points.append([x, y, z])
        
        return self._execute_waypoint_motion(cut_points, speed_factor=0.3)
    

    def _default_scenario(self):
        return self._tissue_manipulation_scenario()

    def _execute_waypoint_motion(self, waypoints: List[List[float]], speed_factor: float = 1.0):
        num_joints = p.getNumJoints(self.robot_id)
        joint_data = []

        for waypoint_idx, waypoint in enumerate(waypoints):
            try:
                joint_angles = p.calculateInverseKinematics(
                    self.robot_id,
                    num_joints - 1,
                    waypoint,
                    maxNumIterations=100,
                    residualThreshold=0.01
                )
            
            except:
                print(f"IK failed for waypoint {waypoint}")
                continue
            
            steps = int(60 * speed_factor)
            for step in range(steps):
                for joint_idx in range(min(7, len(joint_angles))):
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint_idx,
                        p.POSITION_CONTROL,
                        joint_angles[joint_idx],
                        maxVelocity=0.5,
                        force=50.0
                    )
                
                p.stepSimulation()

                if step % 10 == 0:
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
                        'positions' : positions[:7],
                        'velocities' : velocities[:7],
                        'torques' : torques[:7],
                        'contact_detected' : contact_detected,
                        'contact_force' : contact_force,
                        'contact_type' : contact_type,
                        'timestamp' : time.time()
                    })

                if self.gui:
                    time.sleep(1.0 / 240.0)
        
        return joint_data
        

    def collect_surgical_data(self, num_scenarios: int = 5, data_file: str = "surgical_data.csv"):
        print(f"Collecting surgical data with {num_scenarios} scenarios...")

        os.makedirs("data", exist_ok=True)
        filepath = os.path.join("data", data_file)

        all_data = []
        scenarios = ["tissue_manipulation", "suturing", "cutting"]

        for scenario_idx in range(num_scenarios):
            scenario = scenarios[scenario_idx % len(scenarios)]
            print(f"Executing {scenario} scenario {scenario_idx + 1} / {num_scenarios}")

            scenario_data = self.execute_surgical_scenario(scenario)

            for data_point in scenario_data:
                data_point['scenario'] = scenario
                all_data.append(data_point)

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
        
        print(f'Surgical data collection done, saved {len(all_data)} samples to {filepath}')


    def run_surgical_demo(self, duration: int = 30):
        print(f"Running surgical demo for {duration} seconds...")
        scenarios = ["tissue_manipulation", "suturing", "cutting"]
        scenario_duration = duration // len(scenarios)

        for scenario in scenarios:
            print(f"\n--{scenario.upper()} SCENARIO ---")
            start_time = time.time()

            while time.time() - start_time < scenario_duration:
                scenario_data = self.execute_surgical_scenario(scenario)

                contacts = [d for d in scenario_data if d['contact_detected']]
                if contacts:
                    avg_force = np.mean([c['contact_force'] for c in contacts])
                    contact_types = set([c['contact_type'] for c in contacts])
                    print(f"Contacts: {len(contacts)} Avg Force: {avg_force:.3f}, Types: {contact_types}")

        print("\n Surgical demo complete")


    def cleanup(self):
        for constraint_id in self.constraints.values():
            try:
                p.removeConstraint(constraint_id)
            except:
                pass
        if self.physics_client is not None:
            p.disconnect()


def main():
    print("Starting Surgical Contact Detection Simulator...")
    simulator = SurgicalContactSimulator(gui=True)

    try:
        simulator.setup_surgical_environment()
        simulator.run_surgical_demo(duration=20)

        print("\nCollecting surgical training data ...")
        simulator.collect_surgical_data(num_scenarios=10)

    except KeyboardInterrupt:
        print("\nStopping simulation...")
    
    finally:
        simulator.cleanup()



if __name__ == "__main__":
    main()