#!/usr/bin/env python3
"""
Terragon Autonomous SDLC Scheduler
Manages continuous value discovery and execution scheduling
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any


class AutonomousScheduler:
    """Manages continuous execution of the autonomous SDLC system"""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_path = self.repo_path / ".terragon" / "value-config.yaml"
        self.metrics_path = self.repo_path / ".terragon" / "value-metrics.json"
        self.discovery_engine = self.repo_path / ".terragon" / "value-discovery-engine.py"
        self.executor = self.repo_path / ".terragon" / "autonomous-executor.py"
        
    def should_run_discovery(self) -> bool:
        """Check if it's time to run value discovery"""
        
        # For demo purposes, always run discovery
        # In production, this would check:
        # - Last discovery time
        # - Git changes since last run
        # - Scheduled intervals
        return True
    
    def should_execute_tasks(self) -> bool:
        """Check if it's time to execute tasks"""
        
        # Check execution window (2 AM - 6 AM UTC for safety)
        current_hour = datetime.utcnow().hour
        execution_window = (2, 6)  # 2 AM to 6 AM UTC
        
        # For demo, allow execution anytime
        return True
        
        # In production:
        # return execution_window[0] <= current_hour < execution_window[1]
    
    def run_value_discovery(self) -> Dict[str, Any]:
        """Run the value discovery engine"""
        
        print("🔍 Running value discovery cycle...")
        
        try:
            result = subprocess.run([
                sys.executable, str(self.discovery_engine), "--dry-run"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                print("✅ Value discovery completed successfully")
                return {"success": True, "output": result.stdout}
            else:
                print(f"⚠️  Value discovery completed with warnings: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            print(f"❌ Value discovery failed: {e}")
            return {"success": False, "error": str(e)}
    
    def run_execution_cycle(self) -> Dict[str, Any]:
        """Run the autonomous executor"""
        
        print("🤖 Running execution cycle...")
        
        try:
            result = subprocess.run([
                sys.executable, str(self.executor)
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.returncode == 0:
                print("✅ Execution cycle completed successfully")
                return {"success": True, "output": result.stdout}
            else:
                print(f"⚠️  Execution cycle completed with issues: {result.stderr}")
                return {"success": False, "error": result.stderr}
                
        except Exception as e:
            print(f"❌ Execution cycle failed: {e}")
            return {"success": False, "error": str(e)}
    
    def update_schedule_metrics(self, cycle_result: Dict[str, Any]) -> None:
        """Update scheduling metrics"""
        
        try:
            # Load existing metrics
            if self.metrics_path.exists():
                with open(self.metrics_path) as f:
                    metrics = json.load(f)
            else:
                metrics = {"schedulingMetrics": {}}
            
            # Update scheduling metrics
            if "schedulingMetrics" not in metrics:
                metrics["schedulingMetrics"] = {}
            
            scheduling = metrics["schedulingMetrics"]
            scheduling["lastScheduleRun"] = datetime.now().isoformat()
            scheduling["cyclesCompleted"] = scheduling.get("cyclesCompleted", 0) + 1
            scheduling["successfulCycles"] = scheduling.get("successfulCycles", 0)
            
            if cycle_result.get("success"):
                scheduling["successfulCycles"] += 1
            
            scheduling["successRate"] = scheduling["successfulCycles"] / scheduling["cyclesCompleted"]
            
            # Save updated metrics
            with open(self.metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Failed to update scheduling metrics: {e}")
    
    def run_continuous_cycle(self, max_iterations: int = 1) -> Dict[str, Any]:
        """Run continuous autonomous SDLC cycles"""
        
        print("🚀 Starting Terragon Continuous Autonomous SDLC")
        print("=" * 60)
        
        cycle_results = []
        
        for iteration in range(max_iterations):
            print(f"\n📅 Cycle {iteration + 1}/{max_iterations}")
            print("-" * 40)
            
            cycle_start = datetime.now()
            
            # 1. Run value discovery
            if self.should_run_discovery():
                discovery_result = self.run_value_discovery()
                
                # 2. Execute tasks if discovery successful and in execution window
                if discovery_result.get("success") and self.should_execute_tasks():
                    execution_result = self.run_execution_cycle()
                else:
                    execution_result = {"success": False, "reason": "outside_execution_window"}
            else:
                discovery_result = {"success": False, "reason": "not_scheduled"}
                execution_result = {"success": False, "reason": "discovery_skipped"}
            
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            
            cycle_result = {
                "iteration": iteration + 1,
                "timestamp": cycle_start.isoformat(),
                "duration": cycle_duration,
                "discovery": discovery_result,
                "execution": execution_result,
                "success": discovery_result.get("success", False) and execution_result.get("success", False)
            }
            
            cycle_results.append(cycle_result)
            
            # Update metrics
            self.update_schedule_metrics(cycle_result)
            
            print(f"⏱️  Cycle completed in {cycle_duration:.2f} seconds")
            
            # Sleep between iterations (in production, this would be hours/days)
            if iteration < max_iterations - 1:
                print("⏳ Waiting for next cycle...")
                time.sleep(1)  # Short sleep for demo
        
        print("=" * 60)
        successful_cycles = sum(1 for r in cycle_results if r["success"])
        print(f"✅ Completed {successful_cycles}/{max_iterations} cycles successfully")
        
        return {
            "totalCycles": max_iterations,
            "successfulCycles": successful_cycles,
            "results": cycle_results
        }


def main():
    """CLI entry point for the autonomous scheduler"""
    
    # Parse command line arguments
    max_iterations = 1
    if len(sys.argv) > 1:
        try:
            max_iterations = int(sys.argv[1])
        except ValueError:
            print("Usage: python scheduler.py [max_iterations]")
            sys.exit(1)
    
    try:
        scheduler = AutonomousScheduler()
        result = scheduler.run_continuous_cycle(max_iterations)
        
        if result["successfulCycles"] > 0:
            print(f"\n🎉 Autonomous SDLC system completed {result['successfulCycles']} successful cycles!")
            sys.exit(0)
        else:
            print("\n⚠️  No successful cycles completed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Autonomous SDLC scheduler stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Scheduler failed: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()