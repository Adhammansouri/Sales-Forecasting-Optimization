from typing import Optional, Dict, Any
import time
from datetime import datetime
from dataclasses import dataclass, field
import json
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class StageProgress:
    """Progress information for a pipeline stage."""
    name: str
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Calculate stage duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class PipelineProgress:
    """Tracks progress of the entire pipeline."""
    
    def __init__(self, output_dir: str):
        """Initialize progress tracker.
        
        Args:
            output_dir: Directory to save progress reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.stages: Dict[str, StageProgress] = {}
        self.start_time = time.time()
        self.end_time: Optional[float] = None
    
    def start_stage(self, stage_name: str) -> None:
        """Mark a stage as started.
        
        Args:
            stage_name: Name of the stage
        """
        self.stages[stage_name] = StageProgress(
            name=stage_name,
            status="running",
            start_time=time.time()
        )
        logger.info(f"Started stage: {stage_name}")
    
    def complete_stage(self, stage_name: str, metrics: Optional[Dict[str, Any]] = None) -> None:
        """Mark a stage as completed.
        
        Args:
            stage_name: Name of the stage
            metrics: Optional metrics for the stage
        """
        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} was not started")
            
        stage = self.stages[stage_name]
        stage.status = "completed"
        stage.end_time = time.time()
        if metrics:
            stage.metrics = metrics
            
        logger.info(f"Completed stage: {stage_name}")
        if metrics:
            logger.info(f"Stage metrics: {metrics}")
    
    def fail_stage(self, stage_name: str, error: str) -> None:
        """Mark a stage as failed.
        
        Args:
            stage_name: Name of the stage
            error: Error message
        """
        if stage_name not in self.stages:
            raise ValueError(f"Stage {stage_name} was not started")
            
        stage = self.stages[stage_name]
        stage.status = "failed"
        stage.end_time = time.time()
        stage.error = error
        
        logger.error(f"Failed stage: {stage_name}")
        logger.error(f"Error: {error}")
    
    def get_stage_status(self, stage_name: str) -> Optional[StageProgress]:
        """Get status of a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Stage progress information if exists, None otherwise
        """
        return self.stages.get(stage_name)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get status of the entire pipeline.
        
        Returns:
            Dictionary with pipeline status information
        """
        completed_stages = sum(1 for s in self.stages.values() if s.status == "completed")
        failed_stages = sum(1 for s in self.stages.values() if s.status == "failed")
        total_stages = len(self.stages)
        
        return {
            "total_stages": total_stages,
            "completed_stages": completed_stages,
            "failed_stages": failed_stages,
            "in_progress_stages": total_stages - completed_stages - failed_stages,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() if self.end_time else None,
            "duration": time.time() - self.start_time,
            "stages": {
                name: {
                    "status": stage.status,
                    "duration": stage.duration,
                    "error": stage.error,
                    "metrics": stage.metrics
                }
                for name, stage in self.stages.items()
            }
        }
    
    def save_progress(self, filename: str = "pipeline_progress.json") -> None:
        """Save progress report to file.
        
        Args:
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        
        try:
            with open(output_path, 'w') as f:
                json.dump(self.get_pipeline_status(), f, indent=4)
            logger.info(f"Progress report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving progress report: {e}")
            raise
    
    def complete_pipeline(self) -> None:
        """Mark the pipeline as completed and save final progress."""
        self.end_time = time.time()
        self.save_progress()
        
        status = self.get_pipeline_status()
        logger.info("Pipeline completed!")
        logger.info(f"Total stages: {status['total_stages']}")
        logger.info(f"Completed stages: {status['completed_stages']}")
        logger.info(f"Failed stages: {status['failed_stages']}")
        logger.info(f"Total duration: {status['duration']:.2f} seconds") 