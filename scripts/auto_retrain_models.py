#!/usr/bin/env python3
"""
Automated Model Retraining Pipeline

This script automatically retrains all ML models when new data is available.
Designed to run after Kaggle data updates.

Author: Football AI System
"""

import os
import sys
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class AutoModelRetrainer:
    """
    Automated model retraining system
    """
    
    def __init__(self, trigger_source: str = "manual"):
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.logs_dir = self.project_root / "logs"
        self.log_file = self.logs_dir / "auto_retrain.log"
        self.models_dir = self.project_root / "models"
        
        # Training pipeline scripts
        self.training_scripts = [
            "pipelines/generate_features.py",
            "pipelines/train_poisson.py", 
            "pipelines/train_ml_models.py",
            "pipelines/train_btts_improved.py",
            "pipelines/train_ensemble.py"
        ]
        
        # Metadata
        self.trigger_source = trigger_source
        self.start_time = datetime.now()
        
        # Create directories
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_prerequisites(self) -> bool:
        """
        Check if all prerequisites are met for retraining
        
        Returns:
            bool: True if ready for retraining, False otherwise
        """
        try:
            self.logger.info("🔍 Checking prerequisites for model retraining...")
            
            # Check if training scripts exist
            missing_scripts = []
            for script in self.training_scripts:
                script_path = self.project_root / script
                if not script_path.exists():
                    missing_scripts.append(script)
            
            if missing_scripts:
                self.logger.error(f"❌ Missing training scripts: {missing_scripts}")
                return False
            
            # Check if data directory exists
            data_dir = self.project_root / "data"
            if not data_dir.exists():
                self.logger.warning("⚠️  Main data directory not found, checking data_raw...")
                data_raw_dir = self.project_root / "data_raw" / "espn"
                if not data_raw_dir.exists():
                    self.logger.error("❌ No data directories found")
                    return False
            
            # Check Python environment
            try:
                import pandas as pd
                import numpy as np
                import sklearn
                import xgboost
                import lightgbm
                self.logger.info("✅ All required Python packages available")
            except ImportError as e:
                self.logger.error(f"❌ Missing required package: {e}")
                return False
            
            self.logger.info("✅ All prerequisites met")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking prerequisites: {e}")
            return False
    
    def backup_existing_models(self) -> bool:
        """
        Create backup of existing models before retraining
        
        Returns:
            bool: True if backup successful, False otherwise
        """
        try:
            backup_timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
            backup_dir = self.project_root / "models" / f"backup_{backup_timestamp}"
            
            self.logger.info(f"💾 Creating model backup at: {backup_dir}")
            
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model directories
            import shutil
            model_dirs = [
                "model_1x2_v1",
                "model_ou25_v1", 
                "model_btts_v1",
                "model_btts_improved",
                "model_poisson_v1",
                "ensemble_v1"
            ]
            
            backed_up_count = 0
            for model_dir in model_dirs:
                source_dir = self.models_dir / model_dir
                if source_dir.exists():
                    target_dir = backup_dir / model_dir
                    shutil.copytree(source_dir, target_dir)
                    backed_up_count += 1
                    self.logger.debug(f"📁 Backed up: {model_dir}")
            
            self.logger.info(f"✅ Backup completed: {backed_up_count} model directories backed up")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backup failed: {e}")
            return False
    
    def run_training_script(self, script_path: str, timeout_minutes: int = 30) -> Tuple[bool, str]:
        """
        Run a single training script
        
        Args:
            script_path: Path to the training script
            timeout_minutes: Timeout in minutes
            
        Returns:
            Tuple[bool, str]: (success, output/error_message)
        """
        try:
            script_name = Path(script_path).name
            self.logger.info(f"🚀 Starting: {script_name}")
            
            # Build command
            cmd = ["python3", script_path]
            
            # Execute with timeout
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=timeout_minutes * 60
            )
            
            if result.returncode == 0:
                self.logger.info(f"✅ Completed: {script_name}")
                self.logger.debug(f"Output: {result.stdout[-500:]}")  # Last 500 chars
                return True, result.stdout
            else:
                self.logger.error(f"❌ Failed: {script_name} (exit code: {result.returncode})")
                self.logger.error(f"Error: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            self.logger.error(f"❌ Timeout: {script_name} exceeded {timeout_minutes} minutes")
            return False, f"Timeout after {timeout_minutes} minutes"
        except Exception as e:
            self.logger.error(f"❌ Exception in {script_name}: {e}")
            return False, str(e)
    
    def hot_reload_services(self) -> bool:
        """
        Hot reload prediction services to use new models
        
        Returns:
            bool: True if reload successful, False otherwise
        """
        try:
            self.logger.info("🔄 Hot reloading prediction services...")
            
            # Test if services can load new models
            test_script = """
import sys
sys.path.insert(0, '.')
from api.prediction_service import PredictionService

try:
    service = PredictionService()
    print("✅ PredictionService loaded successfully")
    
    # Quick test prediction
    result = service.predict('Test Team A', 'Test Team B', 'Premier League')
    print("✅ Test prediction successful")
    
except Exception as e:
    print(f"❌ Service reload failed: {e}")
    sys.exit(1)
"""
            
            # Write and execute test script
            test_file = self.project_root / "temp_service_test.py"
            with open(test_file, 'w') as f:
                f.write(test_script)
            
            result = subprocess.run(
                ["python3", str(test_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Cleanup test file
            test_file.unlink()
            
            if result.returncode == 0:
                self.logger.info("✅ Services hot reloaded successfully")
                return True
            else:
                self.logger.error(f"❌ Service reload failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Hot reload error: {e}")
            return False
    
    def generate_training_report(self, results: List[Tuple[str, bool, str]]) -> Dict:
        """
        Generate comprehensive training report
        
        Args:
            results: List of (script_name, success, output) tuples
            
        Returns:
            Dict: Training report
        """
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        successful_scripts = [r for r in results if r[1]]
        failed_scripts = [r for r in results if not r[1]]
        
        report = {
            "timestamp": end_time.isoformat(),
            "trigger_source": self.trigger_source,
            "duration_seconds": duration.total_seconds(),
            "total_scripts": len(results),
            "successful_scripts": len(successful_scripts),
            "failed_scripts": len(failed_scripts),
            "success_rate": len(successful_scripts) / len(results) if results else 0,
            "script_results": [
                {
                    "script": r[0],
                    "success": r[1],
                    "output_preview": r[2][:200] if r[2] else ""
                }
                for r in results
            ],
            "status": "success" if len(failed_scripts) == 0 else "partial_failure" if len(successful_scripts) > 0 else "failure"
        }
        
        # Save report
        report_file = self.logs_dir / f"auto_retrain_report_{end_time.strftime('%Y%m%d_%H%M%S')}.json"
        try:
            import json
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📊 Training report saved: {report_file}")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save training report: {e}")
        
        return report
    
    def run_full_retrain(self, skip_backup: bool = False) -> bool:
        """
        Run the complete model retraining pipeline
        
        Args:
            skip_backup: Skip model backup (for testing)
            
        Returns:
            bool: True if retraining completed successfully, False otherwise
        """
        self.logger.info("=" * 70)
        self.logger.info("🤖 STARTING AUTOMATED MODEL RETRAINING PIPELINE")
        self.logger.info(f"📅 Timestamp: {self.start_time.isoformat()}")
        self.logger.info(f"🎯 Trigger: {self.trigger_source}")
        self.logger.info("=" * 70)
        
        try:
            # Step 1: Prerequisites check
            if not self.check_prerequisites():
                self.logger.error("❌ Prerequisites check failed")
                return False
            
            # Step 2: Backup existing models
            if not skip_backup:
                if not self.backup_existing_models():
                    self.logger.error("❌ Model backup failed")
                    return False
            else:
                self.logger.info("⏭️  Skipping model backup")
            
            # Step 3: Run training pipeline
            results = []
            self.logger.info("🎯 Starting training pipeline...")
            
            for script in self.training_scripts:
                script_name = Path(script).name
                
                # Set timeout based on script type
                timeout = 60 if "ensemble" in script else 30  # Ensemble takes longer
                
                success, output = self.run_training_script(script, timeout)
                results.append((script_name, success, output))
                
                if not success:
                    self.logger.error(f"❌ Training pipeline failed at: {script_name}")
                    break
            
            # Step 4: Hot reload services
            if all(r[1] for r in results):  # All scripts successful
                self.logger.info("🔄 All training completed, reloading services...")
                service_reload_success = self.hot_reload_services()
                results.append(("service_reload", service_reload_success, "Hot reload of prediction services"))
            
            # Step 5: Generate report
            report = self.generate_training_report(results)
            
            # Step 6: Final summary
            duration = datetime.now() - self.start_time
            success_count = len([r for r in results if r[1]])
            total_count = len(results)
            
            self.logger.info("📈 RETRAINING PIPELINE SUMMARY:")
            self.logger.info(f"   ✅ Successful steps: {success_count}/{total_count}")
            self.logger.info(f"   ⏱️  Total duration: {duration.total_seconds():.1f} seconds")
            self.logger.info(f"   📊 Success rate: {report['success_rate']:.1%}")
            
            if report['status'] == 'success':
                self.logger.info("🎉 AUTOMATED RETRAINING COMPLETED SUCCESSFULLY!")
                self.logger.info("=" * 70)
                return True
            else:
                self.logger.error(f"❌ RETRAINING FAILED - Status: {report['status']}")
                self.logger.error("=" * 70)
                return False
                
        except Exception as e:
            self.logger.error(f"❌ CRITICAL ERROR IN RETRAINING PIPELINE: {e}")
            self.logger.error("=" * 70)
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Automated Model Retraining Pipeline")
    parser.add_argument("--trigger", default="manual", help="Source that triggered the retraining")
    parser.add_argument("--skip-backup", action="store_true", help="Skip model backup (for testing)")
    
    args = parser.parse_args()
    
    retrainer = AutoModelRetrainer(trigger_source=args.trigger)
    success = retrainer.run_full_retrain(skip_backup=args.skip_backup)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
