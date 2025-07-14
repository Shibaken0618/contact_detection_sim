"""
Complete Contact Detection Demo Runner
Orchestrates the entire pipeline from data collection to real-time detection
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

class ContactDetectionDemo:
    """Main demo orchestrator"""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.examples_dir = self.base_dir / "examples"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Track running processes
        self.processes = []
        
    def check_dependencies(self):
        """Check if all required dependencies are installed"""
        print("🔍 Checking dependencies...")
        
        dependencies = [
            ('pybullet', 'PyBullet physics simulation'),
            ('torch', 'PyTorch machine learning'),
            ('numpy', 'NumPy numerical computing'),
            ('pandas', 'Pandas data processing'),
            ('matplotlib', 'Matplotlib visualization'),
            ('sklearn', 'Scikit-learn machine learning'),
            ('rclpy', 'ROS2 Python client library')
        ]
        
        missing = []
        for dep, desc in dependencies:
            try:
                __import__(dep)
                print(f"  ✅ {desc}")
            except ImportError:
                print(f"  ❌ {desc} - MISSING")
                missing.append(dep)
        
        if missing:
            print(f"\n❌ Missing dependencies: {', '.join(missing)}")
            print("Please install missing dependencies and try again.")
            return False
        
        print("✅ All dependencies satisfied!")
        return True
    
    def step_1_test_pybullet(self):
        """Step 1: Test PyBullet installation"""
        print("\n" + "="*50)
        print("🔧 STEP 1: Testing PyBullet Installation")
        print("="*50)
        
        script_path = self.examples_dir / "pybullet_test.py"
        if not script_path.exists():
            print("❌ pybullet_test.py not found!")
            return False
        
        try:
            result = subprocess.run([sys.executable, str(script_path)], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ PyBullet test completed successfully!")
                return True
            else:
                print(f"❌ PyBullet test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ PyBullet test timed out")
            return False
        except Exception as e:
            print(f"❌ Error running PyBullet test: {e}")
            return False
    
    def step_2_collect_data(self):
        """Step 2: Collect training data"""
        print("\n" + "="*50)
        print("📊 STEP 2: Collecting Training Data")
        print("="*50)
        
        # Check if data already exists
        basic_data = self.data_dir / "joint_data.csv"
        surgical_data = self.data_dir / "surgical_data.csv"
        
        if basic_data.exists() and surgical_data.exists():
            print("📁 Training data already exists!")
            response = input("Do you want to regenerate data? (y/N): ")
            if response.lower() != 'y':
                print("✅ Using existing data")
                return True
        
        # Collect basic data
        print("🤖 Collecting basic contact detection data...")
        basic_script = self.examples_dir / "basic_simulator.py"
        try:
            result = subprocess.run([sys.executable, str(basic_script)], 
                                  timeout=120)
            if result.returncode != 0:
                print("❌ Basic data collection failed")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Basic data collection timed out")
            return False
        
        # Collect surgical data
        print("🏥 Collecting surgical scenario data...")
        surgical_script = self.examples_dir / "surgical_simulator.py"
        try:
            result = subprocess.run([sys.executable, str(surgical_script)], 
                                  timeout=180)
            if result.returncode != 0:
                print("❌ Surgical data collection failed")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Surgical data collection timed out")
            return False
        
        print("✅ Training data collection completed!")
        return True
    
    def step_3_train_model(self):
        """Step 3: Train the ML model"""
        print("\n" + "="*50)
        print("🧠 STEP 3: Training ML Model")
        print("="*50)
        
        # Check if model already exists
        model_path = self.models_dir / "contact_detector.pth"
        if model_path.exists():
            print("🤖 Trained model already exists!")
            response = input("Do you want to retrain? (y/N): ")
            if response.lower() != 'y':
                print("✅ Using existing model")
                return True
        
        # Train the model
        print("🔄 Training contact detection model...")
        train_script = self.examples_dir / "train_contact_detector.py"
        
        try:
            result = subprocess.run([sys.executable, str(train_script)], 
                                  timeout=300)
            if result.returncode != 0:
                print("❌ Model training failed")
                return False
        except subprocess.TimeoutExpired:
            print("❌ Model training timed out")
            return False
        
        print("✅ Model training completed!")
        return True
    
    def step_4_run_ros_demo(self):
        """Step 4: Run ROS2 real-time demo"""
        print("\n" + "="*50)
        print("🚀 STEP 4: Running ROS2 Real-time Demo")
        print("="*50)
        
        print("🤖 Starting ROS2 contact detection node...")
        print("   - Publishing contact detection results")
        print("   - Simulating robot joint states")
        print("   - Press Ctrl+C to stop")
        
        ros_script = self.examples_dir / "contact_detector_node.py"
        
        try:
            # Start ROS2 node
            process = subprocess.Popen([sys.executable, str(ros_script)])
            self.processes.append(process)
            
            # Let it run for a demo period
            print("⏱️  Demo running for 30 seconds...")
            time.sleep(30)
            
            # Stop the process
            process.terminate()
            process.wait(timeout=5)
            
            print("✅ ROS2 demo completed!")
            return True
            
        except Exception as e:
            print(f"❌ ROS2 demo failed: {e}")
            return False
    
    def run_complete_demo(self):
        """Run the complete demo pipeline"""
        print("🎯 Starting Complete Contact Detection Demo")
        print("This will run the entire pipeline from setup to real-time detection")
        print()
        
        # Check dependencies first
        if not self.check_dependencies():
            return False
        
        # Step 1: Test PyBullet
        if not self.step_1_test_pybullet():
            print("❌ Demo failed at Step 1")
            return False
        
        # Step 2: Collect data
        if not self.step_2_collect_data():
            print("❌ Demo failed at Step 2")
            return False
        
        # Step 3: Train model
        if not self.step_3_train_model():
            print("❌ Demo failed at Step 3")
            return False
        
        # Step 4: Run ROS2 demo
        if not self.step_4_run_ros_demo():
            print("❌ Demo failed at Step 4")
            return False
        
        # Success!
        print("\n" + "="*50)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("✅ PyBullet simulation working")
        print("✅ Training data collected")
        print("✅ ML model trained and saved")
        print("✅ ROS2 real-time detection demonstrated")
        print()
        print("📁 Generated Files:")
        print(f"   - {self.data_dir}/joint_data.csv")
        print(f"   - {self.data_dir}/surgical_data.csv")
        print(f"   - {self.models_dir}/contact_detector.pth")
        print(f"   - {self.models_dir}/scaler.pkl")
        print()
        print("🚀 Your contact detection system is ready!")
        print("   Use 'python3 examples/contact_detector_node.py' to run real-time detection")
        
        return True
    
    def run_interactive_demo(self):
        """Run an interactive demo where user chooses steps"""
        print("🎯 Interactive Contact Detection Demo")
        print("Choose which steps to run:")
        print()
        
        steps = [
            ("Test PyBullet Installation", self.step_1_test_pybullet),
            ("Collect Training Data", self.step_2_collect_data),
            ("Train ML Model", self.step_3_train_model),
            ("Run ROS2 Demo", self.step_4_run_ros_demo),
            ("Run Complete Pipeline", self.run_complete_demo)
        ]
        
        while True:
            print("\nAvailable steps:")
            for i, (name, _) in enumerate(steps, 1):
                print(f"  {i}. {name}")
            print("  0. Exit")
            
            try:
                choice = int(input("\nSelect step (0-5): "))
                if choice == 0:
                    break
                elif 1 <= choice <= len(steps):
                    name, func = steps[choice - 1]
                    print(f"\n🔄 Running: {name}")
                    success = func()
                    if success:
                        print(f"✅ {name} completed successfully!")
                    else:
                        print(f"❌ {name} failed!")
                else:
                    print("❌ Invalid choice!")
            except ValueError:
                print("❌ Please enter a number!")
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
    
    def cleanup(self):
        """Clean up running processes"""
        for