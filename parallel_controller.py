#!/usr/bin/env python3
"""
Parallel Controller for miRNA Curator GPU Processing

This script replaces the shell script approach with better process management,
logging, and error handling for concurrent GPU processing.
"""

import subprocess
import argparse
import json
import logging
import time
import os
import sys
import signal
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ProcessConfig:
    """Configuration for a single GPU process"""
    gpu_id: int
    checkpoint_file: str
    input_data: str
    output_data: str
    config_file: str
    process_id: str


class ParallelController:
    """Controller for managing parallel GPU processes"""
    
    def __init__(self, base_config_file: str, log_dir: str = "logs"):
        self.base_config_file = base_config_file
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup master logging
        self.setup_logging()
        
        # Process management
        self.processes: List[subprocess.Popen] = []
        self.process_configs: List[ProcessConfig] = []
        self.start_time = None
        
        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def setup_logging(self):
        """Setup master logging for the controller"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"controller_{timestamp}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        # Setup console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        self.logger = logging.getLogger('ParallelController')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        self.logger.info(f"Controller logging initialized. Log file: {log_file}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
        self.cleanup_processes()
        sys.exit(1)
    
    def create_process_configs(self, 
                             gpu_count: int = 4,
                             checkpoint_pattern: str = "gfllm_qwq_checkpoint_split_{}.parquet",
                             input_pattern: str = "production_test_data_2025-03-31_split_{}.parquet",
                             output_pattern: str = "production_data_output_2025-03-31_chunk_{}.parquet") -> List[ProcessConfig]:
        """Create configuration for each GPU process"""
        configs = []
        
        for gpu_id in range(gpu_count):
            config = ProcessConfig(
                gpu_id=gpu_id,
                checkpoint_file=checkpoint_pattern.format(gpu_id),
                input_data=input_pattern.format(gpu_id),
                output_data=output_pattern.format(gpu_id),
                config_file=self.base_config_file,
                process_id=f"gpu_{gpu_id}"
            )
            configs.append(config)
            
        return configs
    
    def validate_input_files(self, configs: List[ProcessConfig]) -> bool:
        """Validate that all required input files exist"""
        missing_files = []
        
        # Check base config file
        if not Path(self.base_config_file).exists():
            missing_files.append(self.base_config_file)
        
        # Check input files for each process
        for config in configs:
            if not Path(config.checkpoint_file).exists():
                self.logger.warning(f"Checkpoint file not found: {config.checkpoint_file}")
            
            if not Path(config.input_data).exists():
                missing_files.append(config.input_data)
        
        if missing_files:
            self.logger.error(f"Missing required files: {missing_files}")
            return False
        
        return True
    
    def create_process_command(self, config: ProcessConfig) -> List[str]:
        """Create the command line for a single process"""
        cmd = [
            sys.executable,  # Use the same Python interpreter
            "src/mirna_curator/main.py",
            "--config", config.config_file,
            "--checkpoint_file_path", config.checkpoint_file,
            "--input_data", config.input_data,
            "--output_data", config.output_data,
            "--gpu", str(config.gpu_id)
        ]
        
        return cmd
    
    def create_process_environment(self, config: ProcessConfig) -> Dict[str, str]:
        """Create environment variables for a single process"""
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(config.gpu_id)
        env['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        
        return env
    
    def start_process(self, config: ProcessConfig) -> subprocess.Popen:
        """Start a single GPU process"""
        cmd = self.create_process_command(config)
        env = self.create_process_environment(config)
        
        # Create log files for this process
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stdout_log = self.log_dir / f"{config.process_id}_{timestamp}_stdout.log"
        stderr_log = self.log_dir / f"{config.process_id}_{timestamp}_stderr.log"
        
        self.logger.info(f"Starting process {config.process_id} on GPU {config.gpu_id}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        self.logger.info(f"Output logs: {stdout_log}, {stderr_log}")
        
        # Start the process
        with open(stdout_log, 'w') as stdout_file, open(stderr_log, 'w') as stderr_file:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=stdout_file,
                stderr=stderr_file,
                cwd=".",  # Run from project root
                preexec_fn=os.setsid if os.name != 'nt' else None  # Create process group on Unix
            )
        
        self.logger.info(f"Process {config.process_id} started with PID {process.pid}")
        return process
    
    def monitor_processes(self, check_interval: int = 30) -> Dict[str, any]:
        """Monitor all running processes"""
        results = {}
        
        while self.processes:
            time.sleep(check_interval)
            
            # Check each process
            for i, (process, config) in enumerate(zip(self.processes[:], self.process_configs[:])):
                if process.poll() is not None:  # Process has finished
                    exit_code = process.returncode
                    
                    if exit_code == 0:
                        self.logger.info(f"Process {config.process_id} completed successfully")
                        results[config.process_id] = {"status": "success", "exit_code": exit_code}
                    else:
                        self.logger.error(f"Process {config.process_id} failed with exit code {exit_code}")
                        results[config.process_id] = {"status": "failed", "exit_code": exit_code}
                    
                    # Remove completed process from monitoring
                    self.processes.remove(process)
                    self.process_configs.remove(config)
            
            if self.processes:
                self.logger.info(f"Still monitoring {len(self.processes)} processes...")
            
            # Log progress
            self.log_progress()
        
        return results
    
    def log_progress(self):
        """Log current progress information"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            self.logger.info(f"Runtime: {elapsed:.1f}s, Active processes: {len(self.processes)}")
    
    def cleanup_processes(self):
        """Cleanup any remaining processes"""
        if not self.processes:
            return
        
        self.logger.info(f"Cleaning up {len(self.processes)} remaining processes...")
        
        # First, try graceful termination
        for process in self.processes:
            if process.poll() is None:  # Still running
                self.logger.info(f"Terminating process {process.pid}")
                if os.name != 'nt':
                    # On Unix, terminate the process group
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    except (OSError, ProcessLookupError):
                        process.terminate()
                else:
                    process.terminate()
        
        # Wait a bit for graceful shutdown
        time.sleep(5)
        
        # Force kill if necessary
        for process in self.processes:
            if process.poll() is None:
                self.logger.warning(f"Force killing process {process.pid}")
                if os.name != 'nt':
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except (OSError, ProcessLookupError):
                        process.kill()
                else:
                    process.kill()
        
        self.processes.clear()
        self.process_configs.clear()
    
    def run_parallel_processing(self, 
                               gpu_count: int = 4,
                               checkpoint_pattern: str = "gfllm_qwq_checkpoint_split_{}.parquet",
                               input_pattern: str = "production_test_data_2025-03-31_split_{}.parquet",
                               output_pattern: str = "production_data_output_2025-03-31_chunk_{}.parquet",
                               check_interval: int = 30) -> bool:
        """Run the complete parallel processing workflow"""
        
        self.logger.info("="*50)
        self.logger.info("Starting Parallel miRNA Curator Processing")
        self.logger.info("="*50)
        
        # Create process configurations
        configs = self.create_process_configs(gpu_count, checkpoint_pattern, input_pattern, output_pattern)
        self.logger.info(f"Created configurations for {len(configs)} processes")
        
        # Validate input files
        if not self.validate_input_files(configs):
            self.logger.error("Input validation failed. Aborting.")
            return False
        
        self.logger.info("Input validation passed")
        
        try:
            # Start all processes
            self.start_time = time.time()
            for config in configs:
                process = self.start_process(config)
                self.processes.append(process)
                self.process_configs.append(config)
                
                # Small delay between process starts to avoid resource conflicts
                time.sleep(2)
            
            self.logger.info(f"All {len(self.processes)} processes started successfully")
            
            # Monitor processes until completion
            results = self.monitor_processes(check_interval)
            
            # Report final results
            self.report_final_results(results)
            
            # Check if all processes succeeded
            success_count = sum(1 for r in results.values() if r["status"] == "success")
            total_time = time.time() - self.start_time
            
            self.logger.info(f"Processing completed in {total_time:.1f}s")
            self.logger.info(f"Successful processes: {success_count}/{len(results)}")
            
            return success_count == len(results)
            
        except Exception as e:
            self.logger.error(f"Error during parallel processing: {e}")
            self.cleanup_processes()
            return False
        
        finally:
            self.cleanup_processes()
    
    def report_final_results(self, results: Dict[str, any]):
        """Report final processing results"""
        self.logger.info("="*50)
        self.logger.info("FINAL RESULTS")
        self.logger.info("="*50)
        
        for process_id, result in results.items():
            status = result["status"]
            exit_code = result["exit_code"]
            self.logger.info(f"{process_id}: {status.upper()} (exit code: {exit_code})")
        
        # Check output files
        self.logger.info("\nOutput File Status:")
        for config in self.process_configs:
            if Path(config.output_data).exists():
                size = Path(config.output_data).stat().st_size
                self.logger.info(f"{config.output_data}: EXISTS ({size} bytes)")
            else:
                self.logger.warning(f"{config.output_data}: MISSING")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Parallel Controller for miRNA Curator GPU Processing")
    
    parser.add_argument(
        "--config",
        default="configs/curation_config_QwQ_prod.json",
        help="Base configuration file for the curation process"
    )
    
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=4,
        help="Number of GPU processes to run"
    )
    
    parser.add_argument(
        "--checkpoint-pattern",
        default="gfllm_qwq_checkpoint_split_{}.parquet",
        help="Pattern for checkpoint files (use {} for GPU ID placeholder)"
    )
    
    parser.add_argument(
        "--input-pattern",
        default="production_test_data_2025-03-31_split_{}.parquet",
        help="Pattern for input data files (use {} for GPU ID placeholder)"
    )
    
    parser.add_argument(
        "--output-pattern",
        default="production_data_output_2025-03-31_chunk_{}.parquet",
        help="Pattern for output data files (use {} for GPU ID placeholder)"
    )
    
    parser.add_argument(
        "--check-interval",
        type=int,
        default=30,
        help="Interval in seconds between process status checks"
    )
    
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for log files"
    )
    
    args = parser.parse_args()
    
    # Create and run controller
    controller = ParallelController(args.config, args.log_dir)
    
    success = controller.run_parallel_processing(
        gpu_count=args.gpu_count,
        checkpoint_pattern=args.checkpoint_pattern,
        input_pattern=args.input_pattern,
        output_pattern=args.output_pattern,
        check_interval=args.check_interval
    )
    
    if success:
        print("All processes completed successfully!")
        sys.exit(0)
    else:
        print("Some processes failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
