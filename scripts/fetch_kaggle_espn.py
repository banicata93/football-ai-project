#!/usr/bin/env python3
"""
Automated Daily Kaggle ESPN Soccer Data Fetcher

This script automatically downloads the latest ESPN soccer data from Kaggle
and processes it for the Football AI system.

Dataset: excel4soccer/espn-soccer-data
URL: https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data

Author: Football AI System
"""

import os
import sys
import json
import logging
import zipfile
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple


class KaggleESPNFetcher:
    """
    Automated fetcher for ESPN soccer data from Kaggle
    """
    
    def __init__(self):
        # Setup paths
        self.project_root = Path(__file__).parent.parent
        self.data_raw_dir = self.project_root / "data_raw" / "espn"
        self.logs_dir = self.project_root / "logs"
        self.log_file = self.logs_dir / "kaggle_fetch.log"
        
        # Kaggle dataset info
        self.dataset_name = "excel4soccer/espn-soccer-data"
        self.download_path = self.data_raw_dir / "temp_download"
        
        # Create directories
        self.data_raw_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.download_path.mkdir(parents=True, exist_ok=True)
        
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
        
    def check_kaggle_credentials(self) -> bool:
        """
        Check if Kaggle API credentials are properly configured
        
        Returns:
            bool: True if credentials are found, False otherwise
        """
        try:
            # Check for kaggle.json in standard locations
            kaggle_config_paths = [
                Path.home() / ".kaggle" / "kaggle.json",
                Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json" if os.environ.get("KAGGLE_CONFIG_DIR") else None
            ]
            
            for config_path in kaggle_config_paths:
                if config_path and config_path.exists():
                    self.logger.info(f"✅ Found Kaggle credentials at: {config_path}")
                    return True
                    
            # Check environment variables
            if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
                self.logger.info("✅ Found Kaggle credentials in environment variables")
                return True
                
            self.logger.error("❌ No Kaggle credentials found!")
            self.logger.error("Please ensure kaggle.json is in ~/.kaggle/ or set KAGGLE_USERNAME/KAGGLE_KEY")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking Kaggle credentials: {e}")
            return False
    
    def download_dataset(self) -> bool:
        """
        Download the ESPN dataset from Kaggle
        
        Returns:
            bool: True if download successful, False otherwise
        """
        try:
            self.logger.info(f"📥 Starting download of dataset: {self.dataset_name}")
            
            # Build kaggle command
            cmd = [
                "kaggle", "datasets", "download", 
                "-d", self.dataset_name,
                "-p", str(self.download_path),
                "--unzip"
            ]
            
            # Execute download
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Dataset downloaded successfully")
                self.logger.debug(f"Download output: {result.stdout}")
                return True
            else:
                self.logger.error(f"❌ Download failed with return code: {result.returncode}")
                self.logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Download timed out after 5 minutes")
            return False
        except FileNotFoundError:
            self.logger.error("❌ Kaggle CLI not found. Please install: pip install kaggle")
            return False
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during download: {e}")
            return False
    
    def get_existing_files(self) -> set:
        """
        Get set of existing files in the data directory
        
        Returns:
            set: Set of existing file names
        """
        existing_files = set()
        if self.data_raw_dir.exists():
            for file_path in self.data_raw_dir.rglob("*"):
                if file_path.is_file():
                    # Store relative path from data_raw_dir
                    rel_path = file_path.relative_to(self.data_raw_dir)
                    existing_files.add(str(rel_path))
        return existing_files
    
    def process_downloaded_files(self) -> Tuple[List[str], List[str]]:
        """
        Process downloaded files, moving new ones and skipping existing ones
        
        Returns:
            Tuple[List[str], List[str]]: (new_files, skipped_files)
        """
        new_files = []
        skipped_files = []
        existing_files = self.get_existing_files()
        
        try:
            # Process all files in download directory
            for file_path in self.download_path.rglob("*"):
                if file_path.is_file():
                    # Calculate relative path
                    rel_path = file_path.relative_to(self.download_path)
                    target_path = self.data_raw_dir / rel_path
                    
                    # Check if file already exists
                    if str(rel_path) in existing_files:
                        skipped_files.append(str(rel_path))
                        self.logger.debug(f"⏭️  Skipping existing file: {rel_path}")
                        continue
                    
                    # Create target directory if needed
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Move file to final location
                    file_path.rename(target_path)
                    new_files.append(str(rel_path))
                    self.logger.info(f"📄 New file added: {rel_path}")
            
            return new_files, skipped_files
            
        except Exception as e:
            self.logger.error(f"❌ Error processing downloaded files: {e}")
            return [], []
    
    def cleanup_temp_files(self):
        """Clean up temporary download directory"""
        try:
            import shutil
            if self.download_path.exists():
                shutil.rmtree(self.download_path)
                self.logger.debug("🧹 Cleaned up temporary download directory")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not clean up temp directory: {e}")
    
    def generate_summary_report(self, new_files: List[str], skipped_files: List[str]) -> Dict:
        """
        Generate summary report of the fetch operation
        
        Args:
            new_files: List of newly added files
            skipped_files: List of skipped existing files
            
        Returns:
            Dict: Summary report
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset": self.dataset_name,
            "total_new_files": len(new_files),
            "total_skipped_files": len(skipped_files),
            "new_files": new_files,
            "skipped_files": skipped_files[:10] if len(skipped_files) > 10 else skipped_files,  # Limit for readability
            "status": "success" if new_files or skipped_files else "no_data"
        }
        
        # Save report to JSON
        report_file = self.logs_dir / f"kaggle_fetch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📊 Report saved to: {report_file}")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not save report: {e}")
            
        return report
    
    def trigger_model_retraining(self, new_files_count: int) -> bool:
        """
        Trigger automated model retraining if significant new data was added
        
        Args:
            new_files_count: Number of new files downloaded
            
        Returns:
            bool: True if retraining was triggered successfully, False otherwise
        """
        try:
            # Only retrain if we have significant new data
            if new_files_count < 10:  # Threshold for retraining
                self.logger.info(f"📊 Only {new_files_count} new files, skipping retraining")
                return True
            
            self.logger.info(f"🤖 Triggering model retraining ({new_files_count} new files)...")
            
            # Import and run the retraining pipeline
            retrain_script = self.project_root / "scripts" / "auto_retrain_models.py"
            
            if not retrain_script.exists():
                self.logger.error("❌ Auto retrain script not found")
                return False
            
            # Execute retraining pipeline
            import subprocess
            result = subprocess.run(
                ["python3", str(retrain_script), "--trigger", "kaggle_update"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for full retraining
            )
            
            if result.returncode == 0:
                self.logger.info("✅ Model retraining completed successfully")
                self.logger.info("🔄 Models are now updated with latest data")
                return True
            else:
                self.logger.error(f"❌ Model retraining failed: {result.stderr}")
                self.logger.error("⚠️  Models were NOT updated - using previous versions")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Model retraining timed out after 1 hour")
            return False
        except Exception as e:
            self.logger.error(f"❌ Error triggering model retraining: {e}")
            return False

    def run_daily_update(self) -> bool:
        """
        Main function to run the daily update process
        
        Returns:
            bool: True if update completed successfully, False otherwise
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("🚀 STARTING KAGGLE ESPN DATA DAILY UPDATE")
        self.logger.info(f"📅 Timestamp: {start_time.isoformat()}")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Check credentials
            if not self.check_kaggle_credentials():
                return False
            
            # Step 2: Download dataset
            if not self.download_dataset():
                return False
            
            # Step 3: Process files
            self.logger.info("📁 Processing downloaded files...")
            new_files, skipped_files = self.process_downloaded_files()
            
            # Step 4: Generate report
            report = self.generate_summary_report(new_files, skipped_files)
            
            # Step 5: Log summary
            self.logger.info("📈 DAILY UPDATE SUMMARY:")
            self.logger.info(f"   📄 New files added: {len(new_files)}")
            self.logger.info(f"   ⏭️  Files skipped (existing): {len(skipped_files)}")
            
            if new_files:
                self.logger.info("🆕 NEW FILES:")
                for file_name in new_files[:5]:  # Show first 5
                    self.logger.info(f"   • {file_name}")
                if len(new_files) > 5:
                    self.logger.info(f"   ... and {len(new_files) - 5} more files")
            
            # Step 6: Trigger model retraining if new data was added
            retraining_success = True
            if new_files:
                self.logger.info("🤖 Checking if model retraining is needed...")
                retraining_success = self.trigger_model_retraining(len(new_files))
            else:
                self.logger.info("📊 No new files, skipping model retraining")
            
            # Step 7: Cleanup
            self.cleanup_temp_files()
            
            duration = datetime.now() - start_time
            self.logger.info(f"⏱️  Total duration: {duration.total_seconds():.2f} seconds")
            
            if retraining_success:
                self.logger.info("✅ DAILY UPDATE & MODEL RETRAINING COMPLETED SUCCESSFULLY")
            else:
                self.logger.warning("⚠️  DAILY UPDATE COMPLETED BUT MODEL RETRAINING FAILED")
            
            self.logger.info("=" * 60)
            
            return retraining_success
            
        except Exception as e:
            self.logger.error(f"❌ DAILY UPDATE FAILED: {e}")
            self.logger.error("=" * 60)
            return False


def main():
    """Main entry point"""
    fetcher = KaggleESPNFetcher()
    success = fetcher.run_daily_update()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
