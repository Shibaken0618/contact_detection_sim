"""
Fixed Surgical Simulator with improved bone contact detection and stable tissue positioning
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

        self.contact_threshold = 0.02  # Reduced threshold for better detection
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
            numSolverIterations=300,  # Increased for better stability
            numSubSteps=6,
            contactBreakingThreshold=0.0005,
            erp=0.9,
            contactERP=0.9,
            frictionERP=0.3,
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

        # Let physics settle
        for _ in range(200):
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

        # Add table legs for visual realism
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
        """Add anatomical structures with improved stability and positioning"""
        # FIXED: Tissue positioned properly on the table surface
        tissue_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.08, 0.03])
        tissue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.08, 0.03], rgbaColor=[0.8, 0.4, 0.4, 0.8])
        tissue_height = 0.10 + 0.03  # Table height + tissue half-height
        self.anatomy_objects['tissue'] = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=tissue_collision,
            baseVisualShapeIndex=tissue_visual,
            basePosition=[0.6, 0, tissue_height]  # Fixed positioning
        )

        # Better physics properties for tissue
        p.changeDynamics(self.anatomy_objects['tissue'], -1, 
                         lateralFriction=4.0,
                         rollingFriction=1.5, 
                         restitution=0.05, 
                         linearDamping=0.98,
                         angularDamping=0.98, 
                         contactStiffness=3000, 
                         contactDamping=300)
        
        # Organ - larger and more accessible
        organ_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.08)
        organ_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.08, rgbaColor=[0.6, 0.3, 0.3, 0.8])
        organ_height = 0.10 + 0.08  # Table height + organ radius
        self.anatomy_objects['organ'] = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=organ_collision,
            baseVisualShapeIndex=organ_visual,
            basePosition=[0.45, 0.25, organ_height]
        )

        p.changeDynamics(self.anatomy_objects['organ'], -1, 
                         lateralFriction=3.0, 
                         restitution=0.05, 
                         linearDamping=0.95, 
                         angularDamping=0.95, 
                         contactStiffness=2000, 
                         contactDamping=200)

        # FIXED: Bone positioned properly and made more accessible
        bone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=0.15)
        bone_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=0.15, rgbaColor=[0.9, 0.9, 0.8, 1])
        bone_height = 0.10 + 0.075  # Table height + bone half-height
        bone_position = [0.75, -0.2, bone_height]  # More accessible position
        self.anatomy_objects['bone'] = p.createMultiBody(
            baseMass=0.8,  # Heavier for stability
            baseCollisionShapeIndex=bone_collision,
            baseVisualShapeIndex=bone_visual,
            basePosition=bone_position
        )

        p.changeDynamics(self.anatomy_objects['bone'], -1,
                         lateralFriction=2.0, 
                         restitution=0.05, 
                         linearDamping=0.9, 
                         angularDamping=0.9, 
                         contactStiffness=8000, 
                         contactDamping=800)

        # Add more anatomical structures for increased contact variety
        # Vessel - thin tubular structure
        vessel_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.2)
        vessel_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.2, rgbaColor=[0.7, 0.2, 0.2, 0.9])
        vessel_height = 0.10 + 0.1
        self.anatomy_objects['vessel'] = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=vessel_collision,
            baseVisualShapeIndex=vessel_visual,
            basePosition=[0.4, -0.15, vessel_height]
        )

        p.changeDynamics(self.anatomy_objects['vessel'], -1,
                         lateralFriction=2.5, 
                         restitution=0.1, 
                         linearDamping=0.9, 
                         angularDamping=0.9, 
                         contactStiffness=1500, 
                         contactDamping=150)

        # Nerve - very thin structure
        nerve_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.008, height=0.15)
        nerve_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.008, length=0.15, rgbaColor=[0.9, 0.9, 0.3, 0.8])
        nerve_height = 0.10 + 0.075
        self.anatomy_objects['nerve'] = p.createMultiBody(
            baseMass=0.02,
            baseCollisionShapeIndex=nerve_collision,
            baseVisualShapeIndex=nerve_visual,
            basePosition=[0.8, 0.15, nerve_height]
        )

        p.changeDynamics(self.anatomy_objects['nerve'], -1,
                         lateralFriction=1.5, 
                         restitution=0.2, 
                         linearDamping=0.8, 
                         angularDamping=0.8, 
                         contactStiffness=800, 
                         contactDamping=80)

        # Muscle - larger soft tissue
        muscle_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.15, 0.04])
        muscle_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.15, 0.04], rgbaColor=[0.6, 0.2, 0.2, 0.7])
        muscle_height = 0.10 + 0.04
        self.anatomy_objects['muscle'] = p.createMultiBody(
            baseMass=0.4,
            baseCollisionShapeIndex=muscle_collision,
            baseVisualShapeIndex=muscle_visual,
            basePosition=[0.3, 0.1, muscle_height]
        )

        p.changeDynamics(self.anatomy_objects['muscle'], -1,
                         lateralFriction=3.5, 
                         restitution=0.05, 
                         linearDamping=0.95, 
                         angularDamping=0.95, 
                         contactStiffness=2500, 
                         contactDamping=250)

        # Add improved constraints
        self._add_stability_constraints()
        
        print("[DEBUG] Anatomy object IDs:")
        for name, obj_id in self.anatomy_objects.items():
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            print(f"  {name}: {obj_id} at position {pos}")
        
        # Let objects settle
        for _ in range(300):
            p.stepSimulation()
            
        print("Anatomical structures added with improved stability and positioning")

    def _add_stability_constraints(self):
        """Add improved constraints with proper positioning"""
        # Tissue constraint - positioned on table surface
        tissue_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['tissue'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.08],  # Just above table surface
            childFramePosition=[0, 0, -0.03]   # Bottom of tissue
        )
        p.changeConstraint(tissue_constraint, maxForce=50.0)
        self.constraints['tissue'] = tissue_constraint
        
        # Organ constraint
        organ_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['organ'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.15, 0.25, 0.08],
            childFramePosition=[0, 0, -0.08]
        )
        p.changeConstraint(organ_constraint, maxForce=30.0)
        self.constraints['organ'] = organ_constraint

        # Bone constraint - positioned properly on table
        bone_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['bone'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.15, -0.2, 0.08],
            childFramePosition=[0, 0, -0.075]
        )
        p.changeConstraint(bone_constraint, maxForce=80.0)
        self.constraints['bone'] = bone_constraint

        # Add constraints for new anatomical structures
        vessel_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['vessel'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.2, -0.15, 0.08],
            childFramePosition=[0, 0, -0.1]
        )
        p.changeConstraint(vessel_constraint, maxForce=20.0)
        self.constraints['vessel'] = vessel_constraint

        nerve_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['nerve'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0.2, 0.15, 0.08],
            childFramePosition=[0, 0, -0.075]
        )
        p.changeConstraint(nerve_constraint, maxForce=15.0)
        self.constraints['nerve'] = nerve_constraint

        muscle_constraint = p.createConstraint(
            parentBodyUniqueId=self.table_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.anatomy_objects['muscle'],
            childLinkIndex=-1,
            jointType=p.JOINT_POINT2POINT,
            jointAxis=[0, 0, 0],
            parentFramePosition=[-0.3, 0.1, 0.08],
            childFramePosition=[0, 0, -0.04]
        )
        p.changeConstraint(muscle_constraint, maxForce=40.0)
        self.constraints['muscle'] = muscle_constraint

    def _add_surgical_instruments(self):
        """Add surgical instruments positioned away from anatomy"""
        # Grasper
        grasper_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.12)
        grasper_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.005, length=0.12, rgbaColor=[0.8, 0.8, 0.8, 1])
        self.instruments['grasper'] = p.createMultiBody(
            baseMass=0.02,
            baseCollisionShapeIndex=grasper_collision,
            baseVisualShapeIndex=grasper_visual,
            basePosition=[0.3, 0.4, 0.15]
        )

        # Scissors
        scissors_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04])
        scissors_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04], rgbaColor=[0.7, 0.7, 0.9, 1])
        self.instruments['scissors'] = p.createMultiBody(
            baseMass=0.025,
            baseCollisionShapeIndex=scissors_collision,
            baseVisualShapeIndex=scissors_visual,
            basePosition=[0.3, -0.4, 0.13]
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
        print("[DEBUG] Instrument IDs:")
        for name, obj_id in self.instruments.items():
            print(f"  {name}: {obj_id}")

    def _attach_end_effector(self):
        """Attach end effector with improved design"""
        num_joints = p.getNumJoints(self.robot_id)
        end_effector_link = num_joints - 1
    
        # Larger probe for better contact detection
        probe_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.012)
        probe_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.012, rgbaColor=[0.5, 0.5, 0.8, 1])

        end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
        end_effector_pos = end_effector_state[0]
        probe_pos = [end_effector_pos[0], end_effector_pos[1], end_effector_pos[2] - 0.05]

        self.end_effector_id = p.createMultiBody(
            baseMass=0.008,
            baseCollisionShapeIndex=probe_collision,
            baseVisualShapeIndex=probe_visual,
            basePosition=probe_pos
        )

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

        print("Improved end effector attached to robot")
        print(f"[DEBUG] End effector ID: {self.end_effector_id}")

    def detect_surgical_contact(self) -> List[SurgicalContact]:
        """Improved contact detection with better filtering"""
        contacts = []

        # Check contacts from end effector (primary) and robot
        contact_sources = []
        if self.end_effector_id is not None:
            contact_sources.append(self.end_effector_id)
        contact_sources.append(self.robot_id)

        for source_id in contact_sources:
            contact_points = p.getContactPoints(source_id)

            for contact in contact_points:
                if len(contact) < 10:
                    continue

                bodyA, bodyB = contact[1], contact[2]
                contact_pos = contact[5]
                contact_normal = contact[7]
                normal_force = contact[9]

                # Determine which is the 'other' body (not the robot/end effector)
                if bodyA == source_id:
                    other_body = bodyB
                else:
                    other_body = bodyA

                contact_type = self._classify_contact(other_body)

                # Only report significant contacts and exclude table/ground
                if abs(normal_force) > self.contact_threshold and contact_type not in ['table', 'ground', 'unknown']:
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
        """Classify the type of contact with expanded anatomy types"""
        for anatomy_name, anatomy_id in self.anatomy_objects.items():
            if body_id == anatomy_id:
                return anatomy_name
        
        for instrument_name, instrument_id in self.instruments.items():
            if body_id == instrument_id:
                return 'instrument'
        
        if body_id == self.table_id:
            return 'table'
        elif body_id == self.plane_id:
            return 'ground'
        else:
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
        elif scenario_type == "vessel_cauterization":
            return self._vessel_cauterization_scenario()
        elif scenario_type == "nerve_monitoring":
            return self._nerve_monitoring_scenario()
        elif scenario_type == "muscle_retraction":
            return self._muscle_retraction_scenario()
        else:
            return self._default_scenario()

    def _tissue_manipulation_scenario(self):
        """Improved tissue manipulation"""
        target_positions = [
            [0.4, 0, 0.5],      # start position
            [0.6, 0, 0.4],      # move over table
            [0.6, 0, 0.25],     # approach tissue
            [0.6, 0, 0.18],     # near tissue
            [0.6, 0, 0.16],     # contact tissue
            [0.6, 0.02, 0.16],  # gentle manipulation
            [0.6, -0.02, 0.16], # continue manipulation
            [0.6, 0, 0.15],     # press down
            [0.6, 0.03, 0.15],  # more manipulation
            [0.6, 0, 0.16],     # return to center
            [0.6, 0, 0.25],     # lift
            [0.6, 0, 0.4],      # retract
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.6)

    def _organ_examination_scenario(self):
        """Improved organ examination"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.45, 0.25, 0.4],  # move over organ
            [0.45, 0.25, 0.3],  # approach organ
            [0.45, 0.25, 0.22], # near organ
            [0.45, 0.25, 0.20], # contact organ
            [0.45, 0.23, 0.20], # examine
            [0.45, 0.27, 0.20], # continue examination
            [0.45, 0.25, 0.19], # gentle press
            [0.45, 0.25, 0.30], # retract
            [0.45, 0.25, 0.4],  # lift
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.5)

    def _bone_drilling_scenario(self):
        """Improved bone drilling with guaranteed contact"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.75, -0.2, 0.4],  # move over bone
            [0.75, -0.2, 0.3],  # approach bone
            [0.75, -0.2, 0.22], # near bone
            [0.75, -0.2, 0.19], # contact bone
            [0.75, -0.18, 0.19], # drilling motion
            [0.75, -0.22, 0.19], # continue drilling
            [0.75, -0.2, 0.18], # press down more
            [0.75, -0.2, 0.17], # deeper contact
            [0.75, -0.2, 0.22], # retract
            [0.75, -0.2, 0.4],  # lift
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.4)

    def _vessel_cauterization_scenario(self):
        """New vessel cauterization scenario"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.4, -0.15, 0.4],  # move over vessel
            [0.4, -0.15, 0.25], # approach vessel
            [0.4, -0.15, 0.20], # near vessel
            [0.4, -0.15, 0.18], # contact vessel
            [0.4, -0.13, 0.18], # cauterize along vessel
            [0.4, -0.17, 0.18], # continue cauterization
            [0.4, -0.15, 0.17], # press slightly
            [0.4, -0.15, 0.25], # retract
            [0.4, -0.15, 0.4],  # lift
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.5)

    def _nerve_monitoring_scenario(self):
        """New nerve monitoring scenario"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.8, 0.15, 0.4],   # move over nerve
            [0.8, 0.15, 0.25],  # approach nerve
            [0.8, 0.15, 0.20],  # near nerve
            [0.8, 0.15, 0.18],  # contact nerve
            [0.8, 0.13, 0.18],  # monitor along nerve
            [0.8, 0.17, 0.18],  # continue monitoring
            [0.8, 0.15, 0.17],  # gentle contact
            [0.8, 0.15, 0.25],  # retract
            [0.8, 0.15, 0.4],   # lift
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.6)

    def _muscle_retraction_scenario(self):
        """New muscle retraction scenario"""
        target_positions = [
            [0.4, 0, 0.5],      # start
            [0.3, 0.1, 0.4],    # move over muscle
            [0.3, 0.1, 0.25],   # approach muscle
            [0.3, 0.1, 0.18],   # near muscle
            [0.3, 0.1, 0.16],   # contact muscle
            [0.3, 0.08, 0.16],  # retract muscle
            [0.3, 0.12, 0.16],  # continue retraction
            [0.3, 0.1, 0.15],   # press down
            [0.3, 0.1, 0.25],   # retract
            [0.3, 0.1, 0.4],    # lift
        ]
        return self._execute_waypoint_motion(target_positions, speed_factor=0.5)

    def _default_scenario(self):
        return self._tissue_manipulation_scenario()

    def _execute_waypoint_motion(self, waypoints: List[List[float]], speed_factor: float = 1.0):
        """Execute waypoint motion with improved control"""
        num_joints = p.getNumJoints(self.robot_

