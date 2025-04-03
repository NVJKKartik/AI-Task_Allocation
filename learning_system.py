"""
Learning System Module for AI Task Allocation Agent

This module implements learning and adaptation features that improve task
allocation and user preferences over time.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict

@dataclass
class UserPreference:
    """Stores user preferences learned over time"""
    preferred_task_types: List[str]
    preferred_working_hours: Dict[str, float]  # hour: preference_score
    skill_preferences: Dict[str, float]  # skill: preference_score
    task_complexity_preference: float  # 1-5
    collaboration_preference: float  # 0-1

class LearningSystem:
    """Manages learning and adaptation features"""
    
    def __init__(self):
        """Initialize the learning system"""
        self.user_preferences = {}
        self.historical_data = defaultdict(list)
        self.scaler = StandardScaler()
    
    def update_user_preferences(self, user_id: str, task_data: Dict, feedback: Dict) -> None:
        """Update user preferences based on task completion and feedback"""
        try:
            # Initialize preferences if not exists
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = UserPreference(
                    preferred_task_types=[],
                    preferred_working_hours={},
                    skill_preferences={},
                    task_complexity_preference=3.0,
                    collaboration_preference=0.5
                )
            
            # Update task type preferences
            task_type = task_data.get('category', 'General')
            if task_type not in self.user_preferences[user_id].preferred_task_types:
                self.user_preferences[user_id].preferred_task_types.append(task_type)
            
            # Update working hour preferences
            completion_time = task_data.get('completion_time')
            if completion_time:
                hour = datetime.fromisoformat(completion_time).hour
                self.user_preferences[user_id].preferred_working_hours[hour] = \
                    self.user_preferences[user_id].preferred_working_hours.get(hour, 0) + 1
            
            # Update skill preferences
            for skill, level in task_data.get('required_skills', {}).items():
                current_pref = self.user_preferences[user_id].skill_preferences.get(skill, 0)
                self.user_preferences[user_id].skill_preferences[skill] = \
                    current_pref + (level * feedback.get('satisfaction', 0.5))
            
            # Update complexity preference
            complexity = task_data.get('estimated_complexity', 3)
            current_pref = self.user_preferences[user_id].task_complexity_preference
            self.user_preferences[user_id].task_complexity_preference = \
                (current_pref + complexity) / 2
            
            # Update collaboration preference
            if 'collaboration_feedback' in feedback:
                current_pref = self.user_preferences[user_id].collaboration_preference
                self.user_preferences[user_id].collaboration_preference = \
                    (current_pref + feedback['collaboration_feedback']) / 2
            
            # Store historical data
            self.historical_data[user_id].append({
                'task_data': task_data,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            print(f"Error updating user preferences: {str(e)}")
    
    def analyze_historical_data(self, user_id: str) -> Dict:
        """Analyze historical data to identify patterns and preferences"""
        try:
            if user_id not in self.historical_data:
                return {}
            
            data = self.historical_data[user_id]
            if not data:
                return {}
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(data)
            
            # Analyze completion times
            completion_times = []
            for record in data:
                if 'task_data' in record and 'completion_time' in record['task_data']:
                    completion_times.append(datetime.fromisoformat(record['task_data']['completion_time']))
            
            # Calculate most productive hours
            if completion_times:
                hours = [t.hour for t in completion_times]
                hour_distribution = np.bincount(hours, minlength=24)
                most_productive_hours = np.argsort(hour_distribution)[-3:].tolist()
            else:
                most_productive_hours = []
            
            # Analyze task types
            task_types = df['task_data'].apply(lambda x: x.get('category', 'General')).value_counts()
            preferred_task_types = task_types.head(3).index.tolist()
            
            # Analyze skill preferences
            skill_preferences = defaultdict(float)
            for record in data:
                if 'task_data' in record and 'required_skills' in record['task_data']:
                    for skill, level in record['task_data']['required_skills'].items():
                        skill_preferences[skill] += level * record['feedback'].get('satisfaction', 0.5)
            
            # Normalize skill preferences
            if skill_preferences:
                max_pref = max(skill_preferences.values())
                skill_preferences = {k: v/max_pref for k, v in skill_preferences.items()}
            
            return {
                'most_productive_hours': most_productive_hours,
                'preferred_task_types': preferred_task_types,
                'skill_preferences': dict(skill_preferences),
                'average_complexity': df['task_data'].apply(lambda x: x.get('estimated_complexity', 3)).mean(),
                'collaboration_score': df['feedback'].apply(lambda x: x.get('collaboration_feedback', 0.5)).mean()
            }
        except Exception as e:
            print(f"Error analyzing historical data: {str(e)}")
            return {}
    
    def predict_task_suitability(self, user_id: str, task: Dict) -> float:
        """Predict how suitable a task is for a user based on learned preferences"""
        try:
            if user_id not in self.user_preferences:
                return 0.5  # Default suitability score
            
            preferences = self.user_preferences[user_id]
            
            # Calculate task type match
            task_type_match = 1.0 if task.get('category') in preferences.preferred_task_types else 0.5
            
            # Calculate skill match
            skill_match = 0.0
            if task.get('required_skills'):
                for skill, level in task['required_skills'].items():
                    skill_pref = preferences.skill_preferences.get(skill, 0.5)
                    skill_match += (skill_pref * level) / 5.0
                skill_match /= len(task['required_skills'])
            else:
                skill_match = 0.5
            
            # Calculate complexity match
            complexity = task.get('estimated_complexity', 3)
            complexity_match = 1.0 - abs(complexity - preferences.task_complexity_preference) / 5.0
            
            # Calculate collaboration match
            collaboration_needed = task.get('requires_collaboration', False)
            collaboration_match = 1.0 if collaboration_needed == (preferences.collaboration_preference > 0.5) else 0.5
            
            # Calculate weighted average
            weights = {
                'task_type': 0.3,
                'skill': 0.3,
                'complexity': 0.2,
                'collaboration': 0.2
            }
            
            suitability_score = (
                weights['task_type'] * task_type_match +
                weights['skill'] * skill_match +
                weights['complexity'] * complexity_match +
                weights['collaboration'] * collaboration_match
            )
            
            return suitability_score
        except Exception as e:
            print(f"Error predicting task suitability: {str(e)}")
            return 0.5
    
    def cluster_users_by_preferences(self) -> Dict[str, List[str]]:
        """Cluster users based on their preferences"""
        try:
            if not self.user_preferences:
                return {}
            
            # Prepare data for clustering
            features = []
            user_ids = []
            
            for user_id, preferences in self.user_preferences.items():
                # Convert preferences to feature vector
                feature_vector = [
                    len(preferences.preferred_task_types),
                    np.mean(list(preferences.preferred_working_hours.values())),
                    np.mean(list(preferences.skill_preferences.values())),
                    preferences.task_complexity_preference,
                    preferences.collaboration_preference
                ]
                
                features.append(feature_vector)
                user_ids.append(user_id)
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Perform clustering
            n_clusters = min(5, len(user_ids))  # Maximum 5 clusters
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(features_scaled)
            
            # Group users by cluster
            user_clusters = defaultdict(list)
            for user_id, cluster in zip(user_ids, clusters):
                user_clusters[f"cluster_{cluster}"].append(user_id)
            
            return dict(user_clusters)
        except Exception as e:
            print(f"Error clustering users: {str(e)}")
            return {}
    
    def get_recommendations(self, user_id: str, tasks: List[Dict]) -> List[Dict]:
        """Get task recommendations for a user based on preferences"""
        try:
            if not tasks:
                return []
            
            # Calculate suitability scores for all tasks
            task_scores = []
            for task in tasks:
                score = self.predict_task_suitability(user_id, task)
                task_scores.append((task, score))
            
            # Sort by score
            task_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top recommendations
            return [task for task, _ in task_scores[:5]]
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return []

# Example usage
if __name__ == "__main__":
    # Test learning system
    system = LearningSystem()
    
    # Sample task data
    sample_task = {
        "title": "Implement User Authentication",
        "category": "Development",
        "required_skills": {"Python": 4, "Security": 3},
        "estimated_complexity": 4,
        "requires_collaboration": True,
        "completion_time": datetime.now().isoformat()
    }
    
    # Sample feedback
    sample_feedback = {
        "satisfaction": 0.8,
        "collaboration_feedback": 0.7,
        "comments": "Enjoyed working on this task"
    }
    
    # Test preference updates
    print("Updating user preferences...")
    system.update_user_preferences("user-123", sample_task, sample_feedback)
    
    # Test historical data analysis
    print("\nAnalyzing historical data...")
    analysis = system.analyze_historical_data("user-123")
    print(json.dumps(analysis, indent=2))
    
    # Test task suitability prediction
    print("\nPredicting task suitability...")
    suitability = system.predict_task_suitability("user-123", sample_task)
    print(f"Suitability score: {suitability:.2f}")
    
    # Test user clustering
    print("\nClustering users...")
    clusters = system.cluster_users_by_preferences()
    print(json.dumps(clusters, indent=2))
    
    # Test recommendations
    print("\nGetting task recommendations...")
    recommendations = system.get_recommendations("user-123", [sample_task])
    print(json.dumps(recommendations, indent=2)) 