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

# """
# Attempted realistic surgical simulator
# """

# import os
# import csv
# import time
# import math
# import numpy as np
# from dataclasses import dataclass
# from typing import List, Tuple, Optional
# import pybullet as p
# import pybullet_data


# @dataclass
# class SurgicalContact:
#     contact_type: str
#     force_magnitude: float
#     contact_position: List[float]
#     contact_normal: List[float]
#     timestamp: float


# class SurgicalContactSimulator:
#     def __init__(self, gui=True):
#         self.gui = gui
#         self.physics_client = None
#         self.robot_id = None
#         self.plane_id = None
#         self.table_id = None
#         self.end_effector_id = None

#         self.instruments = {}
#         self.anatomy_objects = {}
#         self.constraints = {}

#         self.contact_threshold = 0.05  # Increased threshold for more reliable detection
#         self.contact_history = []

#         self.scenario_type = "tissue_manipulation"

#     def setup_surgical_environment(self):
#         """Setup the surgical environment with improved stability"""
#         if self.gui:
#             self.physics_client = p.connect(p.GUI)
#         else:
#             self.physics_client = p.connect(p.DIRECT)
        
#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         p.setGravity(0, 0, -9.81)
#         p.setTimeStep(1.0/240.)

#         # Improved physics parameters for stability
#         p.setPhysicsEngineParameter(
#             enableConeFriction=1,
#             numSolverIterations=200,  # Increased for stability
#             numSubSteps=4,
#             contactBreakingThreshold=0.001,
#             erp=0.8,  # Error reduction parameter
#             contactERP=0.8,
#             frictionERP=0.2,
#             enableFileCaching=0
#         )

#         # Load basic environment
#         self.plane_id = p.loadURDF("plane.urdf")
#         self.robot_id = p.loadURDF("kuka_iiwa/model.urdf", [0, 0, 0])

#         # Set robot to a stable initial position
#         num_joints = p.getNumJoints(self.robot_id)
#         initial_angles = [0, -0.3, 0, -1.2, 0, 0.8, 0] 
#         for i in range(min(7, num_joints)):
#             p.resetJointState(self.robot_id, i, initial_angles[i] if i < len(initial_angles) else 0.0)

#         self._setup_surgical_workspace()
#         self._add_anatomical_structures()
#         self._add_surgical_instruments()
#         self._attach_end_effector()

#         # Let physics settle
#         for _ in range(100):
#             p.stepSimulation()

#         print("Improved surgical environment setup complete.")

#     def _setup_surgical_workspace(self):
#         """Create a stable surgical table"""
#         # Larger, more stable table
#         table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.05])
#         table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.5, 0.05], rgbaColor=[0.9, 0.9, 0.9, 1])
#         self.table_id = p.createMultiBody(
#             baseMass=0.0,  # Fixed table
#             baseCollisionShapeIndex=table_collision,
#             baseVisualShapeIndex=table_visual,
#             basePosition=[0.6, 0, 0.05]
#         )

#         # Add table legs for visual realism (optional)
#         for x_offset in [-0.4, 0.4]:
#             for y_offset in [-0.3, 0.3]:
#                 leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.4])
#                 leg_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.4], rgbaColor=[0.7, 0.7, 0.7, 1])
#                 p.createMultiBody(
#                     baseMass=0.0,
#                     baseCollisionShapeIndex=leg_collision,
#                     baseVisualShapeIndex=leg_visual,
#                     basePosition=[0.6 + x_offset, y_offset, -0.35]
#                 )

#         if self.gui:
#             p.resetDebugVisualizerCamera(
#                 cameraDistance=1.8,
#                 cameraYaw=30,
#                 cameraPitch=-25,
#                 cameraTargetPosition=[0.6, 0, 0.2]
#             )

#     def _add_anatomical_structures(self):
#         """Add anatomical structures with improved stability"""
#         # Tissue - larger and more stable
#         tissue_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.12, 0.08, 0.03])
#         tissue_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.08, 0.03], rgbaColor=[0.8, 0.4, 0.4, 0.8])
#         self.anatomy_objects['tissue'] = p.createMultiBody(
#             baseMass=0.2,  # Increased mass for stability
#             baseCollisionShapeIndex=tissue_collision,
#             baseVisualShapeIndex=tissue_visual,
#             basePosition=[0.6, 0, 0.13]
#         )

#         # Better physics properties for tissue
#         p.changeDynamics(self.anatomy_objects['tissue'], -1, 
#                          lateralFriction=3.0,  # Higher friction
#                          rollingFriction=1.0, 
#                          restitution=0.1, 
#                          linearDamping=0.95,  # Higher damping
#                          angularDamping=0.95, 
#                          contactStiffness=2000, 
#                          contactDamping=200)
        
#         # Organ - positioned away from tissue to avoid collisions
#         organ_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.05)
#         organ_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.6, 0.3, 0.3, 0.8])
#         self.anatomy_objects['organ'] = p.createMultiBody(
#             baseMass=0.08,
#             baseCollisionShapeIndex=organ_collision,
#             baseVisualShapeIndex=organ_visual,
#             basePosition=[0.45, 0.2, 0.15]  # Further from tissue
#         )

#         p.changeDynamics(self.anatomy_objects['organ'], -1, 
#                          lateralFriction=2.0, 
#                          restitution=0.1, 
#                          linearDamping=0.9, 
#                          angularDamping=0.9, 
#                          contactStiffness=1000, 
#                          contactDamping=100)

#         # Bone - heavier and more stable, TEST: move directly under robot
#         # Slightly increase bone collision radius for easier contact
#         bone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.04, height=0.12)
#         bone_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.04, length=0.12, rgbaColor=[0.9, 0.9, 0.8, 1])
#         bone_position = [0.6, 0, 0.13]  # TEST: directly under robot
#         self.anatomy_objects['bone'] = p.createMultiBody(
#             baseMass=0.5,  # Much heavier
#             baseCollisionShapeIndex=bone_collision,
#             baseVisualShapeIndex=bone_visual,
#             basePosition=bone_position
#         )

#         p.changeDynamics(self.anatomy_objects['bone'], -1,
#                          lateralFriction=1.5, 
#                          restitution=0.1, 
#                          linearDamping=0.8, 
#                          angularDamping=0.8, 
#                          contactStiffness=5000, 
#                          contactDamping=500)
#         # Print bone position and collision info
#         print(f"[DEBUG] Bone created at position: {bone_position}, collision shape: cylinder, radius=0.03, height=0.12, id={self.anatomy_objects['bone']}")

#         # Add stronger constraints
#         self._add_stability_constraints()
        
#         # Print anatomy object IDs for debug
#         print("[DEBUG] Anatomy object IDs:")
#         for name, obj_id in self.anatomy_objects.items():
#             print(f"  {name}: {obj_id}")
        
#         # Let objects settle
#         for _ in range(200):
#             p.stepSimulation()
            
#         print("Anatomical structures added with improved stability")

#     def _add_stability_constraints(self):
#         """Add improved constraints to keep anatomical objects stable"""
#         # Tissue constraint - stronger and more stable
#         tissue_constraint = p.createConstraint(
#             parentBodyUniqueId=self.table_id,
#             parentLinkIndex=-1,
#             childBodyUniqueId=self.anatomy_objects['tissue'],
#             childLinkIndex=-1,
#             jointType=p.JOINT_POINT2POINT,
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[0, 0, 0.08],
#             childFramePosition=[0, 0, -0.03]
#         )
#         p.changeConstraint(tissue_constraint, maxForce=100.0)  # Much stronger
#         self.constraints['tissue'] = tissue_constraint
        
#         # Add rotational constraint for tissue
#         tissue_rot_constraint = p.createConstraint(
#             parentBodyUniqueId=self.table_id,
#             parentLinkIndex=-1,
#             childBodyUniqueId=self.anatomy_objects['tissue'],
#             childLinkIndex=-1,
#             jointType=p.JOINT_FIXED,
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[0, 0, 0.08],
#             childFramePosition=[0, 0, -0.03]
#         )
#         p.changeConstraint(tissue_rot_constraint, maxForce=50.0)
#         self.constraints['tissue_rot'] = tissue_rot_constraint
        
#         # Organ constraint
#         organ_constraint = p.createConstraint(
#             parentBodyUniqueId=self.table_id,
#             parentLinkIndex=-1,
#             childBodyUniqueId=self.anatomy_objects['organ'],
#             childLinkIndex=-1,
#             jointType=p.JOINT_POINT2POINT,
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[-0.15, 0.2, 0.1],
#             childFramePosition=[0, 0, 0]
#         )
#         p.changeConstraint(organ_constraint, maxForce=15.0)
#         self.constraints['organ'] = organ_constraint

#         # Bone constraint - much stronger
#         bone_constraint = p.createConstraint(
#             parentBodyUniqueId=self.table_id,
#             parentLinkIndex=-1,
#             childBodyUniqueId=self.anatomy_objects['bone'],
#             childLinkIndex=-1,
#             jointType=p.JOINT_FIXED,  # Fixed constraint for bone
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[0.15, -0.15, 0.11],
#             childFramePosition=[0, 0, 0]
#         )
#         p.changeConstraint(bone_constraint, maxForce=50.0)  # Very strong
#         self.constraints['bone'] = bone_constraint

#     def _add_surgical_instruments(self):
#         """Add surgical instruments positioned away from anatomy"""
#         # Grasper
#         grasper_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.12)
#         grasper_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.005, length=0.12, rgbaColor=[0.8, 0.8, 0.8, 1])
#         self.instruments['grasper'] = p.createMultiBody(
#             baseMass=0.02,
#             baseCollisionShapeIndex=grasper_collision,
#             baseVisualShapeIndex=grasper_visual,
#             basePosition=[0.3, 0.3, 0.15]
#         )

#         # Scissors
#         scissors_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04])
#         scissors_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.005, 0.02, 0.04], rgbaColor=[0.7, 0.7, 0.9, 1])
#         self.instruments['scissors'] = p.createMultiBody(
#             baseMass=0.025,
#             baseCollisionShapeIndex=scissors_collision,
#             baseVisualShapeIndex=scissors_visual,
#             basePosition=[0.3, -0.3, 0.13]
#         )

#         # Scalpel
#         scalpel_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05])
#         scalpel_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.01, 0.05], rgbaColor=[0.9, 0.9, 0.7, 1])
#         self.instruments['scalpel'] = p.createMultiBody(
#             baseMass=0.015,
#             baseCollisionShapeIndex=scalpel_collision,
#             baseVisualShapeIndex=scalpel_visual,
#             basePosition=[0.9, 0, 0.12]
#         )

#         for instrument_id in self.instruments.values():
#             p.changeDynamics(instrument_id, -1, 
#                            lateralFriction=0.8, restitution=0.1, 
#                            linearDamping=0.5, angularDamping=0.5)

#         print("Surgical instruments added to workspace")
#         # Print all instrument IDs for debug
#         print("[DEBUG] Instrument IDs:")
#         for name, obj_id in self.instruments.items():
#             print(f"  {name}: {obj_id}")
#         print("End effector attached.")
#         print(f"[DEBUG] End effector ID: {self.end_effector_id}")
#         # Print anatomy object IDs for clarity
#         print("[DEBUG] Anatomy Object IDs:")
#         for name, obj_id in self.anatomy_objects.items():
#             print(f"  {name}: {obj_id}")

#     def _attach_end_effector(self):
#         """Attach end effector with improved design"""
#         num_joints = p.getNumJoints(self.robot_id)
#         end_effector_link = num_joints - 1
    
#         # Larger probe for better contact detection
#         probe_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=0.008)
#         probe_visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.008, rgbaColor=[0.5, 0.5, 0.8, 1])

#         end_effector_state = p.getLinkState(self.robot_id, end_effector_link)
#         end_effector_pos = end_effector_state[0]
#         probe_pos = [end_effector_pos[0], end_effector_pos[1], end_effector_pos[2] - 0.05]

#         self.end_effector_id = p.createMultiBody(
#             baseMass=0.005,  # Slightly heavier for stability
#             baseCollisionShapeIndex=probe_collision,
#             baseVisualShapeIndex=probe_visual,
#             basePosition=probe_pos
#         )

#         self.end_effector_constraint = p.createConstraint(
#             parentBodyUniqueId=self.robot_id,
#             parentLinkIndex=end_effector_link,
#             childBodyUniqueId=self.end_effector_id,
#             childLinkIndex=-1,
#             jointType=p.JOINT_FIXED,
#             jointAxis=[0, 0, 0],
#             parentFramePosition=[0, 0, -0.05],
#             childFramePosition=[0, 0, 0]
#         )

#         print("Improved end effector attached to robot")

#     def detect_surgical_contact(self) -> List[SurgicalContact]:
#         """Improved contact detection with better filtering"""
#         contacts = []

#         unique_contact_ids = set()
#         # Check contacts from end effector (primary) and robot
#         contact_sources = []
#         if self.end_effector_id is not None:
#             contact_sources.append(self.end_effector_id)
#         contact_sources.append(self.robot_id)

#         for source_id in contact_sources:
#             contact_points = p.getContactPoints(source_id)

#             for contact in contact_points:
#                 if len(contact) < 10:
#                     continue

#                 bodyA, bodyB = contact[1], contact[2]
#                 linkA, linkB = contact[3], contact[4]
#                 contact_pos = contact[5]
#                 contact_normal = contact[7]
#                 normal_force = contact[9]

#                 unique_contact_ids.add(bodyA)
#                 unique_contact_ids.add(bodyB)

#                 # Determine which is the 'other' body (not the robot/end effector)
#                 if bodyA == source_id:
#                     other_body = bodyB
#                 else:
#                     other_body = bodyA

#                 contact_type = self._classify_contact(other_body)

#                 # Debug print: show both bodies in contact and their classification
#                 print(f"[DEBUG] Contact pair: bodyA={bodyA}, bodyB={bodyB}, linkA={linkA}, linkB={linkB}, classified_type={contact_type}, normal_force={normal_force}")
                
#                 # Only report significant contacts
#                 if abs(normal_force) > self.contact_threshold:
#                     surgical_contact = SurgicalContact(
#                         contact_type=contact_type,
#                         force_magnitude=abs(normal_force),
#                         contact_position=list(contact_pos),
#                         contact_normal=list(contact_normal),
#                         timestamp=time.time()
#                     )
#                     contacts.append(surgical_contact)
        
#         print(f"[DEBUG] Unique body IDs involved in contact this scenario: {sorted(unique_contact_ids)}")
#         return contacts

#     def _classify_contact(self, body_id) -> str:
#         """Classify the type of contact, with debug logging"""
#         tissue_id = self.anatomy_objects.get('tissue')
#         organ_id = self.anatomy_objects.get('organ')
#         bone_id = self.anatomy_objects.get('bone')
#         instruments_ids = list(self.instruments.values())
#         classification = 'unknown'
#         if body_id == tissue_id:
#             classification = 'tissue'
#         elif body_id == organ_id:
#             classification = 'organ'
#         elif body_id == bone_id:
#             classification = 'bone'
#         elif body_id in instruments_ids:
#             classification = 'instrument'
#         elif body_id == self.table_id:
#             classification = 'table'
#         elif body_id == self.plane_id:
#             classification = 'ground'
#         print(f"[DEBUG] Classifying contact: body_id={body_id} | tissue_id={tissue_id} | organ_id={organ_id} | bone_id={bone_id} | classification={classification}")
#         return classification

#     def get_end_effector_position(self):
#         """Get end effector position"""
#         if self.end_effector_id is not None:
#             pos, _ = p.getBasePositionAndOrientation(self.end_effector_id)
#             return pos
#         else:
#             num_joints = p.getNumJoints(self.robot_id)
#             link_state = p.getLinkState(self.robot_id, num_joints-1)
#             return link_state[0]

#     def execute_surgical_scenario(self, scenario_type: str = "tissue_manipulation"):
#         """Execute surgical scenario with improved trajectories"""
#         self.scenario_type = scenario_type

#         if scenario_type == "tissue_manipulation":
#             return self._tissue_manipulation_scenario()
#         elif scenario_type == "organ_examination":
#             return self._organ_examination_scenario()
#         elif scenario_type == "bone_drilling":
#             return self._bone_drilling_scenario()
#         else:
#             return self._default_scenario()

#     def _tissue_manipulation_scenario(self):
#         """Improved tissue manipulation with better contact"""
#         target_positions = [
#             [0.4, 0, 0.5],      # start position
#             [0.6, 0, 0.4],      # move over table
#             [0.6, 0, 0.25],     # approach tissue
#             [0.6, 0, 0.18],     # near tissue
#             [0.6, 0, 0.16],     # contact tissue
#             [0.6, 0.02, 0.16],  # gentle manipulation
#             [0.6, -0.02, 0.16], # continue manipulation
#             [0.6, 0, 0.15],     # press down slightly
#             [0.6, 0.03, 0.15],  # more manipulation
#             [0.6, 0, 0.16],     # return to center
#             [0.6, 0, 0.25],     # lift
#             [0.6, 0, 0.4],      # retract
#         ]

#         return self._execute_waypoint_motion(target_positions, speed_factor=0.5)

#     def _organ_examination_scenario(self):
#         """Improved organ examination"""
#         target_positions = [
#             [0.4, 0, 0.5],      # start
#             [0.45, 0.2, 0.4],   # move over organ
#             [0.45, 0.2, 0.25],  # approach organ
#             [0.45, 0.2, 0.20],  # near organ
#             [0.45, 0.2, 0.18],  # contact organ
#             [0.45, 0.18, 0.18], # examine
#             [0.45, 0.22, 0.18], # continue examination
#             [0.45, 0.2, 0.17],  # gentle press
#             [0.45, 0.2, 0.25],  # retract
#             [0.45, 0.2, 0.4],   # lift
#         ]

#         return self._execute_waypoint_motion(target_positions, speed_factor=0.4)

#     def _bone_drilling_scenario(self):
#         """Improved bone drilling with better approach"""
#         target_positions = [
#             [0.4, 0, 0.5],      # start
#             [0.75, -0.15, 0.4], # move over bone
#             [0.75, -0.15, 0.3], # approach bone
#             [0.75, -0.15, 0.23], # near bone (lowered)
#             [0.75, -0.15, 0.20], # contact bone (lowered)
#             [0.75, -0.13, 0.20], # drilling motion (lowered)
#             [0.75, -0.17, 0.20], # continue drilling (lowered)
#             [0.75, -0.15, 0.19], # press down (lowered)
#             [0.75, -0.15, 0.23], # retract (lowered)
#             [0.75, -0.15, 0.4],  # lift
#         ]

#         result = self._execute_waypoint_motion(target_positions, speed_factor=0.3)
#         # Debug: print if any bone contact detected in this scenario
#         if any(d.get('contact_type') == 'bone' and d.get('contact_detected') for d in result):
#             print("[DEBUG] Bone contact detected during bone drilling scenario.")
#         else:
#             print("[DEBUG] No bone contact detected during bone drilling scenario.")
#         return result

#     def _default_scenario(self):
#         return self._tissue_manipulation_scenario()

#     def _execute_waypoint_motion(self, waypoints: List[List[float]], speed_factor: float = 1.0):
#         """Execute waypoint motion with improved control"""
#         num_joints = p.getNumJoints(self.robot_id)
#         joint_data = []

#         for waypoint_idx, waypoint in enumerate(waypoints):
#             print(f"Moving to waypoint {waypoint_idx + 1}/{len(waypoints)}: {waypoint}")

#             try:
#                 joint_angles = p.calculateInverseKinematics(
#                     self.robot_id,
#                     num_joints - 1,
#                     waypoint,
#                     maxNumIterations=500,
#                     residualThreshold=0.0001,
#                     jointDamping=[0.05] * num_joints
#                 )
#             except Exception as e:
#                 print(f"IK failed for waypoint {waypoint}: {e}")
#                 continue

#             steps = int(150 * speed_factor)  # More steps for smoother motion
#             for step in range(steps):
#                 for joint_idx in range(min(7, len(joint_angles))):
#                     p.setJointMotorControl2(
#                         self.robot_id,
#                         joint_idx,
#                         p.POSITION_CONTROL,
#                         joint_angles[joint_idx],
#                         maxVelocity=0.1,  # Slower for more control
#                         force=30.0
#                     )

#                 p.stepSimulation()

#                 # Collect data more frequently
#                 if step % 3 == 0:
#                     joint_states = p.getJointStates(self.robot_id, range(min(7, num_joints)))
#                     contacts = self.detect_surgical_contact()

#                     positions = [state[0] for state in joint_states]
#                     velocities = [state[1] for state in joint_states]
#                     torques = [state[3] for state in joint_states]

#                     # Pad to 7 joints
#                     while len(positions) < 7:
#                         positions.append(0.0)
#                         velocities.append(0.0)
#                         torques.append(0.0)

#                     contact_detected = len(contacts) > 0
#                     contact_force = max([c.force_magnitude for c in contacts]) if contacts else 0.0
#                     contact_type = contacts[0].contact_type if contacts else 'none'

#                     joint_data.append({
#                         'positions': positions[:7],
#                         'velocities': velocities[:7],
#                         'torques': torques[:7],
#                         'contact_detected': contact_detected,
#                         'contact_force': contact_force,
#                         'contact_type': contact_type,
#                         'timestamp': time.time()
#                     })

#                     if contact_detected:
#                         ee_pos = self.get_end_effector_position()
#                         print(f"  -> Contact: {contact_type}, Force: {contact_force:.3f}, EE: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

#                 if self.gui:
#                     time.sleep(1.0 / 240.0)

#         return joint_data

#     def collect_surgical_data(self, num_scenarios: int = 9, data_file: str = "surgical_data.csv"):
#         """Collect surgical data with improved scenarios"""
#         print(f"Collecting surgical data with {num_scenarios} scenarios...")

#         os.makedirs("data", exist_ok=True)
#         filepath = os.path.join("data", data_file)

#         all_data = []
#         scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]

#         for scenario_idx in range(num_scenarios):
#             scenario = scenarios[scenario_idx % len(scenarios)]
#             print(f"\n--- Executing {scenario} scenario {scenario_idx + 1}/{num_scenarios} ---")

#             # Reset any displaced objects before each scenario
#             self._reset_anatomy_positions()

#             scenario_data = self.execute_surgical_scenario(scenario)

#             for data_point in scenario_data:
#                 data_point['scenario'] = scenario
#                 all_data.append(data_point)

#         # Save data
#         with open(filepath, 'w', newline='') as csvfile:
#             if not all_data:
#                 print("No data collected")
#                 return

#             fieldnames = []
#             for i in range(7):
#                 fieldnames.extend([f'joint_{i}_pos', f'joint_{i}_vel', f'joint_{i}_torque'])
#             fieldnames.extend(['contact_detected', 'contact_force', 'contact_type', 'scenario'])

#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()

#             for data_point in all_data:
#                 row = {}
#                 for i in range(7):
#                     row[f'joint_{i}_pos'] = data_point['positions'][i]
#                     row[f'joint_{i}_vel'] = data_point['velocities'][i]
#                     row[f'joint_{i}_torque'] = data_point['torques'][i]

#                 row['contact_detected'] = int(data_point['contact_detected'])
#                 row['contact_force'] = data_point['contact_force']
#                 row['contact_type'] = data_point['contact_type']
#                 row['scenario'] = data_point['scenario']

#                 writer.writerow(row)

#         print(f'Surgical data collection complete! Saved {len(all_data)} samples to {filepath}')

#         # Summary stats
#         contact_data = [d for d in all_data if d['contact_detected']]
#         print(f"\nData Summary:")
#         print(f"  Total samples: {len(all_data)}")
#         print(f"  Contact samples: {len(contact_data)} ({len(contact_data)/len(all_data)*100:.1f}%)")
        
#         contact_types = {}
#         for d in contact_data:
#             contact_types[d['contact_type']] = contact_types.get(d['contact_type'], 0) + 1
        
#         print(f"  Contact types: {contact_types}")

#     def _reset_anatomy_positions(self):
#         """Reset anatomical objects to their original positions"""
#         if 'tissue' in self.anatomy_objects:
#             p.resetBasePositionAndOrientation(self.anatomy_objects['tissue'], [0.6, 0, 0.13], [0, 0, 0, 1])
#         if 'organ' in self.anatomy_objects:
#             p.resetBasePositionAndOrientation(self.anatomy_objects['organ'], [0.45, 0.2, 0.15], [0, 0, 0, 1])
#         if 'bone' in self.anatomy_objects:
#             p.resetBasePositionAndOrientation(self.anatomy_objects['bone'], [0.75, -0.15, 0.16], [0, 0, 0, 1])
        
#         # Let physics settle
#         for _ in range(50):
#             p.stepSimulation()

#     def run_surgical_demo(self, duration: int = 60):
#         """Run surgical demo with improved scenarios"""
#         print(f"Running surgical demo for {duration} seconds...")
#         scenarios = ["tissue_manipulation", "organ_examination", "bone_drilling"]
#         scenario_duration = duration // len(scenarios)

#         for scenario in scenarios:
#             print(f"\n--- {scenario.upper()} SCENARIO ---")
            
#             # Reset positions before each scenario
#             self._reset_anatomy_positions()
            
#             scenario_data = self.execute_surgical_scenario(scenario)

#             contacts = [d for d in scenario_data if d['contact_detected']]
#             if contacts:
#                 avg_force = np.mean([c['contact_force'] for c in contacts])
#                 contact_types = {}
#                 for c in contacts:
#                     contact_types[c['contact_type']] = contact_types.get(c['contact_type'], 0) + 1
                
#                 print(f"  Contacts detected: {len(contacts)}")
#                 print(f"  Average force: {avg_force:.3f}")
#                 print(f"  Contact types: {contact_types}")
#             else:
#                 print("  No contacts detected in this scenario")

#         print("\nSurgical demo complete!")

#     def cleanup(self):
#         """Clean up the simulation"""
#         if hasattr(self, 'end_effector_constraint'):
#             try:
#                 p.removeConstraint(self.end_effector_constraint)
#             except:
#                 pass

#         for constraint_id in self.constraints.values():
#             try:
#                 p.removeConstraint(constraint_id)
#             except:
#                 pass

#         if self.physics_client is not None:
#             p.disconnect()


# def main():
#     print("Starting Fixed Surgical Contact Detection Simulator...")
#     simulator = SurgicalContactSimulator(gui=True)

#     try:
#         simulator.setup_surgical_environment()
#         simulator.run_surgical_demo(duration=60)

#         print("\nCollecting surgical training data...")
#         simulator.collect_surgical_data(num_scenarios=18)

#     except KeyboardInterrupt:
#         print("\nStopping simulation...")
    
#     finally:
#         simulator.cleanup()


# if __name__ == "__main__":
#     main()