"""
Progress Tracking Module for AI Task Allocation Agent

This module implements progress tracking and analytics features including
performance metrics, progress visualization, and milestone tracking.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

@dataclass
class PerformanceMetrics:
    """Stores performance metrics for a user"""
    completion_rate: float  # 0-1
    average_completion_time: timedelta
    on_time_completion_rate: float  # 0-1
    skill_improvement: Dict[str, float]  # skill: improvement_score
    task_complexity_handling: Dict[int, float]  # complexity: success_rate

class ProgressTracker:
    """Tracks and analyzes task progress and performance"""
    
    def __init__(self):
        """Initialize the progress tracker"""
        self.task_history = defaultdict(list)
        self.performance_metrics = {}
        self.milestones = defaultdict(list)
    
    def update_task_progress(self, user_id: str, task_id: str, progress: Dict) -> None:
        """Update task progress"""
        try:
            # Store progress update
            self.task_history[user_id].append({
                'task_id': task_id,
                'timestamp': datetime.now().isoformat(),
                'progress': progress
            })
            
            # Check for milestone completion
            self._check_milestones(user_id, task_id, progress)
            
            # Update performance metrics
            self._update_performance_metrics(user_id)
            
        except Exception as e:
            print(f"Error updating task progress: {str(e)}")
    
    def _check_milestones(self, user_id: str, task_id: str, progress: Dict) -> None:
        """Check and update milestone completion"""
        try:
            # Get task data
            task_data = self._get_task_data(task_id)
            if not task_data:
                return
            
            # Check predefined milestones
            if 'milestones' in task_data:
                for milestone in task_data['milestones']:
                    if milestone['id'] not in [m['id'] for m in self.milestones[user_id]]:
                        if progress.get('completion_percentage', 0) >= milestone['threshold']:
                            self.milestones[user_id].append({
                                'id': milestone['id'],
                                'task_id': task_id,
                                'name': milestone['name'],
                                'completed_at': datetime.now().isoformat()
                            })
            
            # Check dynamic milestones (e.g., 25%, 50%, 75%, 100%)
            completion_percentage = progress.get('completion_percentage', 0)
            for threshold in [25, 50, 75, 100]:
                milestone_id = f"{task_id}_progress_{threshold}"
                if milestone_id not in [m['id'] for m in self.milestones[user_id]]:
                    if completion_percentage >= threshold:
                        self.milestones[user_id].append({
                            'id': milestone_id,
                            'task_id': task_id,
                            'name': f"{threshold}% Completion",
                            'completed_at': datetime.now().isoformat()
                        })
        
        except Exception as e:
            print(f"Error checking milestones: {str(e)}")
    
    def _get_task_data(self, task_id: str) -> Optional[Dict]:
        """Get task data from your task management system"""
        # This would typically come from your task database
        return None
    
    def _update_performance_metrics(self, user_id: str) -> None:
        """Update performance metrics for a user"""
        try:
            if user_id not in self.task_history:
                return
            
            # Get user's task history
            history = self.task_history[user_id]
            if not history:
                return
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(history)
            
            # Calculate completion rate
            completed_tasks = df[df['progress'].apply(lambda x: x.get('completion_percentage', 0) == 100)]
            completion_rate = len(completed_tasks) / len(df) if len(df) > 0 else 0
            
            # Calculate average completion time
            completion_times = []
            for task in completed_tasks.itertuples():
                start_time = datetime.fromisoformat(task.progress.get('start_time', task.timestamp))
                end_time = datetime.fromisoformat(task.progress.get('completion_time', task.timestamp))
                completion_times.append(end_time - start_time)
            
            average_completion_time = sum(completion_times, timedelta()) / len(completion_times) if completion_times else timedelta()
            
            # Calculate on-time completion rate
            on_time_tasks = 0
            for task in completed_tasks.itertuples():
                if task.progress.get('completed_on_time', False):
                    on_time_tasks += 1
            
            on_time_completion_rate = on_time_tasks / len(completed_tasks) if len(completed_tasks) > 0 else 0
            
            # Calculate skill improvement
            skill_improvement = defaultdict(float)
            for task in history:
                if 'skill_gains' in task['progress']:
                    for skill, gain in task['progress']['skill_gains'].items():
                        skill_improvement[skill] += gain
            
            # Calculate task complexity handling
            complexity_handling = defaultdict(list)
            for task in history:
                if 'estimated_complexity' in task['progress']:
                    complexity = task['progress']['estimated_complexity']
                    success = task['progress'].get('completion_percentage', 0) == 100
                    complexity_handling[complexity].append(1 if success else 0)
            
            complexity_success_rate = {
                complexity: sum(successes) / len(successes) if successes else 0
                for complexity, successes in complexity_handling.items()
            }
            
            # Update metrics
            self.performance_metrics[user_id] = PerformanceMetrics(
                completion_rate=completion_rate,
                average_completion_time=average_completion_time,
                on_time_completion_rate=on_time_completion_rate,
                skill_improvement=dict(skill_improvement),
                task_complexity_handling=complexity_success_rate
            )
        
        except Exception as e:
            print(f"Error updating performance metrics: {str(e)}")
    
    def get_performance_report(self, user_id: str) -> Dict:
        """Generate a performance report for a user"""
        try:
            if user_id not in self.performance_metrics:
                return {}
            
            metrics = self.performance_metrics[user_id]
            
            return {
                'completion_rate': metrics.completion_rate,
                'average_completion_time': str(metrics.average_completion_time),
                'on_time_completion_rate': metrics.on_time_completion_rate,
                'skill_improvement': metrics.skill_improvement,
                'task_complexity_handling': metrics.task_complexity_handling,
                'milestones_completed': len(self.milestones.get(user_id, [])),
                'recent_tasks': self._get_recent_tasks(user_id)
            }
        except Exception as e:
            print(f"Error generating performance report: {str(e)}")
            return {}
    
    def _get_recent_tasks(self, user_id: str) -> List[Dict]:
        """Get recent tasks for a user"""
        try:
            if user_id not in self.task_history:
                return []
            
            # Get last 5 tasks
            recent_tasks = self.task_history[user_id][-5:]
            
            return [{
                'task_id': task['task_id'],
                'progress': task['progress'],
                'timestamp': task['timestamp']
            } for task in recent_tasks]
        except Exception as e:
            print(f"Error getting recent tasks: {str(e)}")
            return []
    
    def generate_progress_visualization(self, user_id: str, output_path: str) -> bool:
        """Generate progress visualization charts"""
        try:
            if user_id not in self.task_history:
                return False
            
            # Get task history
            history = self.task_history[user_id]
            if not history:
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(history)
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Completion Rate Over Time
            df['date'] = pd.to_datetime(df['timestamp'])
            df['completion'] = df['progress'].apply(lambda x: x.get('completion_percentage', 0))
            df.set_index('date')['completion'].plot(ax=axes[0, 0])
            axes[0, 0].set_title('Completion Rate Over Time')
            axes[0, 0].set_ylabel('Completion Percentage')
            
            # Plot 2: Skill Improvement
            skill_data = defaultdict(list)
            for task in history:
                if 'skill_gains' in task['progress']:
                    for skill, gain in task['progress']['skill_gains'].items():
                        skill_data[skill].append(gain)
            
            if skill_data:
                pd.DataFrame(skill_data).plot(kind='bar', ax=axes[0, 1])
                axes[0, 1].set_title('Skill Improvement')
                axes[0, 1].set_ylabel('Improvement Score')
            
            # Plot 3: Task Complexity Handling
            complexity_data = defaultdict(list)
            for task in history:
                if 'estimated_complexity' in task['progress']:
                    complexity = task['progress']['estimated_complexity']
                    success = task['progress'].get('completion_percentage', 0) == 100
                    complexity_data[complexity].append(1 if success else 0)
            
            if complexity_data:
                success_rates = {
                    complexity: sum(successes) / len(successes)
                    for complexity, successes in complexity_data.items()
                }
                pd.Series(success_rates).plot(kind='bar', ax=axes[1, 0])
                axes[1, 0].set_title('Task Complexity Handling')
                axes[1, 0].set_ylabel('Success Rate')
            
            # Plot 4: Milestone Completion
            if user_id in self.milestones:
                milestone_dates = [
                    datetime.fromisoformat(m['completed_at'])
                    for m in self.milestones[user_id]
                ]
                if milestone_dates:
                    pd.Series(milestone_dates).value_counts().sort_index().plot(kind='bar', ax=axes[1, 1])
                    axes[1, 1].set_title('Milestone Completion')
                    axes[1, 1].set_ylabel('Number of Milestones')
            
            # Adjust layout and save
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            
            return True
        except Exception as e:
            print(f"Error generating progress visualization: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    # Test progress tracker
    tracker = ProgressTracker()
    
    # Sample progress updates
    sample_progress = {
        'completion_percentage': 50,
        'start_time': datetime.now().isoformat(),
        'skill_gains': {'Python': 0.2, 'React': 0.1},
        'estimated_complexity': 3,
        'completed_on_time': True
    }
    
    # Update progress
    print("Updating task progress...")
    tracker.update_task_progress("user-123", "task-456", sample_progress)
    
    # Get performance report
    print("\nGenerating performance report...")
    report = tracker.get_performance_report("user-123")
    print(json.dumps(report, indent=2))
    
    # Generate visualization
    print("\nGenerating progress visualization...")
    success = tracker.generate_progress_visualization("user-123", "progress_visualization.png")
    print(f"Visualization generated: {success}") 