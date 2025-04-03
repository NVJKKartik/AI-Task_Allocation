"""
Personalization Module for AI Task Allocation Agent

This module implements personalization features that allow the system to learn
from past allocations and adapt to user preferences over time.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import pytz
from collections import defaultdict
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Import our task matching and availability management modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root to path
from task_matching_algorithm import Task, UserProfile, TaskAllocation, TaskMatchingEngine # Added TaskMatchingEngine
from availability_management import TimeSlot, AvailabilityManager

class AllocationHistory:
    """Manages the history of task allocations and their outcomes"""
    
    def __init__(self, history_file: str = "allocation_history.json"):
        """
        Initialize the allocation history
        
        Args:
            history_file: Path to the history file
        """
        self.history_file = history_file
        self.allocations: List[Dict] = []
        self.load_history()
    
    def load_history(self) -> None:
        """Load allocation history from file"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    self.allocations = json.load(f)
            except Exception as e:
                print(f"Error loading allocation history: {str(e)}")
                self.allocations = []
    
    def save_history(self) -> None:
        """Save allocation history to file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.allocations, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving allocation history: {str(e)}")
    
    def add_allocation(self, 
                      task: Task, 
                      user: UserProfile, 
                      allocation: TaskAllocation) -> None:
        """
        Add a new allocation to the history
        
        Args:
            task: The task that was allocated
            user: The user the task was allocated to
            allocation: The allocation details
        """
        allocation_record = {
            "allocation_id": f"alloc-{len(self.allocations) + 1000}",
            "task_id": task.task_id,
            "task_title": task.title,
            "task_description": task.description,
            "required_skills": task.required_skills,
            "priority": task.priority,
            "deadline": task.deadline.isoformat(),
            "estimated_duration_minutes": task.estimated_duration.total_seconds() / 60,
            "user_id": user.user_id,
            "user_name": user.name,
            "user_skills": user.skills,
            "user_preferences": user.preferences,
            "user_performance_metrics": user.performance_metrics,
            "user_workload": user.current_workload,
            "confidence_score": allocation.confidence_score,
            "allocation_reason": allocation.allocation_reason,
            "allocated_at": allocation.allocated_at.isoformat(),
            "completed": False,
            "completion_time": None,
            "completion_quality": None,
            "on_time": None,
            "user_satisfaction": None,
            "feedback": None
        }
        
        self.allocations.append(allocation_record)
        self.save_history()
    
    def update_allocation_outcome(self, 
                                 allocation_id: str, 
                                 completed: bool = True,
                                 completion_time: Optional[datetime] = None,
                                 completion_quality: Optional[float] = None,
                                 on_time: Optional[bool] = None,
                                 user_satisfaction: Optional[float] = None,
                                 feedback: Optional[str] = None) -> bool:
        """
        Update the outcome of an allocation
        
        Args:
            allocation_id: ID of the allocation to update
            completed: Whether the task was completed
            completion_time: When the task was completed
            completion_quality: Quality rating of the completed task (1-5)
            on_time: Whether the task was completed on time
            user_satisfaction: User satisfaction rating (1-5)
            feedback: Feedback text
            
        Returns:
            True if the allocation was found and updated, False otherwise
        """
        for i, allocation in enumerate(self.allocations):
            if allocation["allocation_id"] == allocation_id:
                self.allocations[i]["completed"] = completed
                
                if completion_time:
                    self.allocations[i]["completion_time"] = completion_time.isoformat()
                
                if completion_quality is not None:
                    self.allocations[i]["completion_quality"] = completion_quality
                
                if on_time is not None:
                    self.allocations[i]["on_time"] = on_time
                
                if user_satisfaction is not None:
                    self.allocations[i]["user_satisfaction"] = user_satisfaction
                
                if feedback:
                    self.allocations[i]["feedback"] = feedback
                
                self.save_history()
                return True
        
        return False
    
    def get_user_allocations(self, user_id: str) -> List[Dict]:
        """
        Get all allocations for a specific user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of allocation records for the user
        """
        return [a for a in self.allocations if a["user_id"] == user_id]
    
    def get_task_allocations(self, task_id: str) -> List[Dict]:
        """
        Get all allocations for a specific task
        
        Args:
            task_id: ID of the task
            
        Returns:
            List of allocation records for the task
        """
        return [a for a in self.allocations if a["task_id"] == task_id]
    
    def get_allocation_by_id(self, allocation_id: str) -> Optional[Dict]:
        """
        Get an allocation by ID
        
        Args:
            allocation_id: ID of the allocation
            
        Returns:
            Allocation record or None if not found
        """
        for allocation in self.allocations:
            if allocation["allocation_id"] == allocation_id:
                return allocation
        
        return None
    
    def get_recent_allocations(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent allocations
        
        Args:
            limit: Maximum number of allocations to return
            
        Returns:
            List of recent allocation records
        """
        sorted_allocations = sorted(
            self.allocations, 
            key=lambda a: a["allocated_at"], 
            reverse=True
        )
        
        return sorted_allocations[:limit]
    
    def get_allocation_statistics(self) -> Dict:
        """
        Get statistics about allocations
        
        Returns:
            Dictionary with allocation statistics
        """
        total_allocations = len(self.allocations)
        completed_allocations = sum(1 for a in self.allocations if a.get("completed", False))
        on_time_allocations = sum(1 for a in self.allocations if a.get("on_time", False))
        
        avg_quality = 0
        avg_satisfaction = 0
        quality_count = 0
        satisfaction_count = 0
        
        for a in self.allocations:
            if a.get("completion_quality") is not None:
                avg_quality += a["completion_quality"]
                quality_count += 1
            
            if a.get("user_satisfaction") is not None:
                avg_satisfaction += a["user_satisfaction"]
                satisfaction_count += 1
        
        if quality_count > 0:
            avg_quality /= quality_count
        
        if satisfaction_count > 0:
            avg_satisfaction /= satisfaction_count
        
        return {
            "total_allocations": total_allocations,
            "completed_allocations": completed_allocations,
            "completion_rate": completed_allocations / total_allocations if total_allocations > 0 else 0,
            "on_time_rate": on_time_allocations / completed_allocations if completed_allocations > 0 else 0,
            "avg_quality": avg_quality,
            "avg_satisfaction": avg_satisfaction
        }


class UserPreferenceModel:
    """Model for learning and predicting user preferences"""
    
    def __init__(self, model_file: str = "user_preference_model.pkl"):
        """
        Initialize the user preference model
        
        Args:
            model_file: Path to the model file
        """
        self.model_file = model_file
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self) -> None:
        """Load the model from file if it exists"""
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data["model"]
                    self.feature_names = model_data["feature_names"]
            except Exception as e:
                print(f"Error loading user preference model: {str(e)}")
                self.model = None
                self.feature_names = None
    
    def save_model(self) -> None:
        """Save the model to file"""
        if self.model and self.feature_names:
            try:
                with open(self.model_file, 'wb') as f:
                    pickle.dump({
                        "model": self.model,
                        "feature_names": self.feature_names
                    }, f)
            except Exception as e:
                print(f"Error saving user preference model: {str(e)}")
    
    def prepare_data(self, allocation_history: List[Dict]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare data for training the model
        
        Args:
            allocation_history: List of allocation records
            
        Returns:
            Tuple of (features DataFrame, target array)
        """
        # Filter allocations with satisfaction ratings
        rated_allocations = [a for a in allocation_history if a.get("user_satisfaction") is not None]
        
        if not rated_allocations:
            raise ValueError("No rated allocations available for training")
        
        # Extract features
        data = []
        for alloc in rated_allocations:
            # Basic task features
            task_features = {
                "task_priority": alloc["priority"],
                "estimated_duration_minutes": alloc["estimated_duration_minutes"],
                "user_id": alloc["user_id"],
                "user_workload": alloc["user_workload"]
            }
            
            # Extract skill match features
            for skill in set(alloc["required_skills"].keys()) | set(alloc["user_skills"].keys()):
                req_level = alloc["required_skills"].get(skill, 0)
                user_level = alloc["user_skills"].get(skill, 0)
                task_features[f"skill_match_{skill}"] = user_level - req_level
            
            # Extract preference match features
            if "preferred_task_types" in alloc["user_preferences"]:
                task_title_lower = alloc["task_title"].lower()
                task_desc_lower = alloc["task_description"].lower()
                
                for pref_type in alloc["user_preferences"]["preferred_task_types"]:
                    # Check if preference type appears in task title or description
                    in_title = 1 if pref_type.lower() in task_title_lower else 0
                    in_desc = 1 if pref_type.lower() in task_desc_lower else 0
                    task_features[f"pref_match_{pref_type}"] = in_title + in_desc
            
            data.append(task_features)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Extract target variable
        y = np.array([alloc["user_satisfaction"] for alloc in rated_allocations])
        
        # Store feature names
        self.feature_names = list(df.columns)
        
        return df, y
    
    def train(self, allocation_history: List[Dict]) -> None:
        """
        Train the model on allocation history
        
        Args:
            allocation_history: List of allocation records
        """
        try:
            # Prepare data
            X, y = self.prepare_data(allocation_history)
            
            if len(X) < 10:
                print("Not enough data for training (need at least 10 rated allocations)")
                return
            
            # Identify categorical and numerical columns
            categorical_cols = [col for col in X.columns if col == "user_id"]
            numerical_cols = [col for col in X.columns if col != "user_id"]
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ],
                remainder='passthrough'
            )
            
            # Create and train the model
            self.model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
            ])
            
            # Split data for training and validation
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = self.model.score(X_train, y_train)
            val_score = self.model.score(X_val, y_val)
            
            print(f"Model trained successfully. Train R²: {train_score:.4f}, Validation R²: {val_score:.4f}")
            
            # Save the model
            self.save_model()
            
        except Exception as e:
            print(f"Error training user preference model: {str(e)}")
    
    def predict_satisfaction(self, task: Task, user: UserProfile) -> float:
        """
        Predict user satisfaction for a task allocation
        
        Args:
            task: The task to be allocated
            user: The user to allocate the task to
            
        Returns:
            Predicted satisfaction score (1-5)
        """
        if not self.model or not self.feature_names:
            # If no model is available, return a neutral score
            return 3.0
        
        try:
            # Prepare features
            features = {
                "task_priority": task.priority,
                "estimated_duration_minutes": task.estimated_duration.total_seconds() / 60,
                "user_id": user.user_id,
                "user_workload": user.current_workload
            }
            
            # Extract skill match features
            for skill in set(task.required_skills.keys()) | set(user.skills.keys()):
                req_level = task.required_skills.get(skill, 0)
                user_level = user.skills.get(skill, 0)
                features[f"skill_match_{skill}"] = user_level - req_level
            
            # Extract preference match features
            if "preferred_task_types" in user.preferences:
                task_title_lower = task.title.lower()
                task_desc_lower = task.description.lower()
                
                for pref_type in user.preferences["preferred_task_types"]:
                    # Check if preference type appears in task title or description
                    in_title = 1 if pref_type.lower() in task_title_lower else 0
                    in_desc = 1 if pref_type.lower() in task_desc_lower else 0
                    features[f"pref_match_{pref_type}"] = in_title + in_desc
            
            # Create DataFrame with single row
            df = pd.DataFrame([features])
            
            # Ensure all feature columns from training are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0
            
            # Reorder columns to match training data
            df = df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            
            # Clip prediction to valid range
            return max(1.0, min(5.0, prediction))
            
        except Exception as e:
            print(f"Error predicting user satisfaction: {str(e)}")
            return 3.0  # Return neutral score on error


class PersonalizedTaskMatcher:
    """Personalized task matching that learns from past allocations"""
    
    def __init__(self, 
                 task_matching_engine: TaskMatchingEngine, # Added engine parameter
                 allocation_history: Optional[AllocationHistory] = None,
                 preference_model: Optional[UserPreferenceModel] = None):
        """
        Initialize the personalized task matcher
        
        Args:
            task_matching_engine: The core LLM-based matching engine
            allocation_history: Allocation history object
            preference_model: User preference model
        """
        self.task_matching_engine = task_matching_engine # Store the engine
        self.allocation_history = allocation_history or AllocationHistory()
        self.preference_model = preference_model or UserPreferenceModel()
        self.user_performance_cache = {}
        self.user_preference_cache = {}
        self.update_caches()
    
    def update_caches(self) -> None:
        """Update the performance and preference caches from allocation history"""
        # Reset caches
        self.user_performance_cache = defaultdict(lambda: {
            "completion_rate": 0.0,
            "on_time_rate": 0.0,
            "avg_quality": 0.0,
            "count": 0
        })
        
        self.user_preference_cache = defaultdict(lambda: {
            "preferred_skills": defaultdict(int),
            "avoided_skills": defaultdict(int),
            "preferred_priorities": defaultdict(int),
            "count": 0
        })
        
        # Process all allocations
        for allocation in self.allocation_history.allocations:
            user_id = allocation["user_id"]
            
            # Update performance cache
            if allocation.get("completed") is not None:
                self.user_performance_cache[user_id]["count"] += 1
                self.user_performance_cache[user_id]["completion_rate"] += 1 if allocation["completed"] else 0
                
                if allocation.get("on_time") is not None:
                    self.user_performance_cache[user_id]["on_time_rate"] += 1 if allocation["on_time"] else 0
                
                if allocation.get("completion_quality") is not None:
                    self.user_performance_cache[user_id]["avg_quality"] += allocation["completion_quality"]
            
            # Update preference cache
            self.user_preference_cache[user_id]["count"] += 1
            
            # Track skills based on satisfaction
            if allocation.get("user_satisfaction") is not None:
                satisfaction = allocation["user_satisfaction"]
                
                for skill, level in allocation["required_skills"].items():
                    if satisfaction >= 4:  # High satisfaction
                        self.user_preference_cache[user_id]["preferred_skills"][skill] += 1
                    elif satisfaction <= 2:  # Low satisfaction
                        self.user_preference_cache[user_id]["avoided_skills"][skill] += 1
                
                # Track priority preferences
                priority = allocation["priority"]
                if satisfaction >= 4:  # High satisfaction
                    self.user_preference_cache[user_id]["preferred_priorities"][priority] += 1
        
        # Calculate averages
        for user_id, perf in self.user_performance_cache.items():
            if perf["count"] > 0:
                perf["completion_rate"] /= perf["count"]
                perf["on_time_rate"] /= perf["count"]
                perf["avg_quality"] /= perf["count"]
    
    def get_user_performance(self, user_id: str) -> Dict:
        """
        Get performance metrics for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with performance metrics
        """
        return dict(self.user_performance_cache[user_id])
    
    def get_user_preferences(self, user_id: str) -> Dict:
        """
        Get learned preferences for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with learned preferences
        """
        return dict(self.user_preference_cache[user_id])
    
    def calculate_personalized_score(self, task: Task, user: UserProfile) -> float:
        """
        Calculate a personalized score for a task-user match
        
        Args:
            task: The task to be allocated
            user: The user to allocate the task to
            
        Returns:
            Personalized score (0-1)
        """
        # Base score components
        skill_score = 0.0
        preference_score = 0.0
        
        # Calculate skill match score
        total_skills = len(task.required_skills)
        if total_skills > 0:
            skill_matches = 0
            for skill, required_level in task.required_skills.items():
                user_level = user.skills.get(skill, 0)
                if user_level >= required_level:
                    skill_matches += 1
            
            skill_score = skill_matches / total_skills
        else:
            skill_score = 1.0  # No skills required
        
        # Calculate preference score based on learned preferences
        user_prefs = self.user_preference_cache[user.user_id]
        if user_prefs["count"] > 0:
            # Check if task has preferred skills
            pref_skill_score = 0.0
            for skill in task.required_skills:
                if skill in user_prefs["preferred_skills"]:
                    pref_skill_score += user_prefs["preferred_skills"][skill] / user_prefs["count"]
                if skill in user_prefs["avoided_skills"]:
                    pref_skill_score -= user_prefs["avoided_skills"][skill] / user_prefs["count"]
            
            # Normalize to 0-1 range
            pref_skill_score = max(0.0, min(1.0, (pref_skill_score + 1) / 2))
            
            # Check if task has preferred priority
            pref_priority_score = 0.0
            if task.priority in user_prefs["preferred_priorities"]:
                pref_priority_score = user_prefs["preferred_priorities"][task.priority] / user_prefs["count"]
            
            preference_score = (pref_skill_score + pref_priority_score) / 2
        else:
            # Fall back to explicit preferences if no learned preferences
            if "preferred_task_types" in user.preferences:
                task_title_lower = task.title.lower()
                task_desc_lower = task.description.lower()
                
                for pref_type in user.preferences["preferred_task_types"]:
                    if (pref_type.lower() in task_title_lower or 
                        pref_type.lower() in task_desc_lower):
                        preference_score += 0.2  # Add 0.2 for each preference match
                
                preference_score = min(1.0, preference_score)  # Cap at 1.0
        
        # Calculate final personalized score with weights
        # Adjusted weights: more emphasis on skill, less on satisfaction prediction
        final_score = (
            skill_score * 0.7 + 
            preference_score * 0.3
            # satisfaction_score * 0.1 - Removed satisfaction prediction influence for now
        )
        
        return final_score
    
    def match_task_to_users(self, task: Task, users: List[UserProfile]) -> Dict:
        """
        Match a task to the best available user using LLM engine first, then personalization.
        
        Args:
            task: The task to be allocated
            users: List of available users
            
        Returns:
            Dictionary with matching results
        """
        if not users:
            return {
                "best_match_user_id": None,
                "confidence_score": 0.0,
                "allocation_reason": "No users available for this task.",
                "alternative_users": []
            }
            
        # 1. Get base matching from LLM engine
        try:
            llm_allocation_result = self.task_matching_engine.match_task_to_users(task, users)
            if not llm_allocation_result or not llm_allocation_result.user_id:
                 return {
                    "best_match_user_id": None,
                    "confidence_score": 0.0,
                    "allocation_reason": "LLM matching engine did not find a suitable user.",
                    "alternative_users": []
                }
        except Exception as e:
            print(f"Error calling TaskMatchingEngine: {e}")
            return {
                "best_match_user_id": None,
                "confidence_score": 0.0,
                "allocation_reason": f"Error during LLM matching: {e}",
                "alternative_users": []
            }

        # 2. Calculate personalized scores for LLM suggestions
        candidates = []
        # Add best match
        best_match_user = next((u for u in users if u.user_id == llm_allocation_result.user_id), None)
        if best_match_user:
            personalized_score = self.calculate_personalized_score(task, best_match_user)
            combined_score = (llm_allocation_result.confidence_score * 0.7) + (personalized_score * 0.3)
            candidates.append({
                "user": best_match_user,
                "llm_score": llm_allocation_result.confidence_score,
                "personalized_score": personalized_score,
                "combined_score": combined_score,
                "llm_reason": llm_allocation_result.allocation_reason
            })

        # Add alternatives
        for alt in llm_allocation_result.alternative_users:
            alt_user = next((u for u in users if u.user_id == alt["user_id"]), None)
            if alt_user:
                personalized_score = self.calculate_personalized_score(task, alt_user)
                combined_score = (alt["confidence_score"] * 0.7) + (personalized_score * 0.3)
                candidates.append({
                    "user": alt_user,
                    "llm_score": alt["confidence_score"],
                    "personalized_score": personalized_score,
                    "combined_score": combined_score,
                    "llm_reason": alt.get("reason", "Alternative suggestion by LLM.")
                })
                
        if not candidates:
            return {
                "best_match_user_id": None,
                "confidence_score": 0.0,
                "allocation_reason": "LLM suggestions could not be processed.",
                "alternative_users": []
            }

        # 3. Re-rank candidates based on combined score
        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        
        best_candidate = candidates[0]
        alternative_candidates = candidates[1:4]

        # 4. Generate enhanced allocation reason
        enhanced_reason = best_candidate["llm_reason"]
        if best_candidate["personalized_score"] > 0.6:
            enhanced_reason += f" Additionally, personalization score ({best_candidate['personalized_score']:.2f}) indicates a good fit based on historical data or preferences."
        elif best_candidate["personalized_score"] < 0.4 and self.user_preference_cache[best_candidate["user"].user_id]["count"] > 0:
             enhanced_reason += f" However, personalization score ({best_candidate['personalized_score']:.2f}) suggests potential misalignment with past preferences."

        # 5. Format output
        final_alternatives = []
        for alt in alternative_candidates:
            alt_reason = alt['llm_reason']
            if alt['personalized_score'] > 0.6:
                 alt_reason += f" (Personalization Score: {alt['personalized_score']:.2f})"
            final_alternatives.append({
                "user_id": alt["user"].user_id,
                "confidence_score": alt["combined_score"], # Report combined score
                "reason": alt_reason
            })

        return {
            "best_match_user_id": best_candidate["user"].user_id,
            "confidence_score": best_candidate["combined_score"], # Report combined score
            "allocation_reason": enhanced_reason,
            "alternative_users": final_alternatives
        }
    
    def record_allocation(self, task: Task, user: UserProfile, allocation_result: Dict) -> None:
        """
        Record a task allocation in the history
        
        Args:
            task: The allocated task
            user: The user the task was allocated to
            allocation_result: The allocation result from match_task_to_users
        """
        # Create a TaskAllocation object
        allocation = TaskAllocation(
            task_id=task.task_id,
            user_id=user.user_id,
            confidence_score=allocation_result["confidence_score"],
            allocation_reason=allocation_result["allocation_reason"],
            alternative_users=allocation_result["alternative_users"]
        )
        
        # Add to history
        self.allocation_history.add_allocation(task, user, allocation)
        
        # Update caches
        self.update_caches()
    
    def record_allocation_outcome(self, 
                                allocation_id: str, 
                                completed: bool = True,
                                completion_time: Optional[datetime] = None,
                                completion_quality: Optional[float] = None,
                                on_time: Optional[bool] = None,
                                user_satisfaction: Optional[float] = None,
                                feedback: Optional[str] = None) -> bool:
        """
        Record the outcome of an allocation
        
        Args:
            allocation_id: ID of the allocation
            completed: Whether the task was completed
            completion_time: When the task was completed
            completion_quality: Quality rating of the completed task (1-5)
            on_time: Whether the task was completed on time
            user_satisfaction: User satisfaction rating (1-5)
            feedback: Feedback text
            
        Returns:
            True if the allocation was found and updated, False otherwise
        """
        result = self.allocation_history.update_allocation_outcome(
            allocation_id, completed, completion_time, completion_quality,
            on_time, user_satisfaction, feedback
        )
        
        if result:
            # Update caches
            self.update_caches()
            
            # Retrain preference model if we have new satisfaction data
            if user_satisfaction is not None:
                self.preference_model.train(self.allocation_history.allocations)
        
        return result
    
    def get_personalized_recommendations(self, user: UserProfile, tasks: List[Task], limit: int = 5) -> List[Dict]:
        """
        Get personalized task recommendations for a user
        
        Args:
            user: The user to get recommendations for
            tasks: List of available tasks
            limit: Maximum number of recommendations to return
            
        Returns:
            List of recommended tasks with scores
        """
        if not tasks:
            return []
        
        # Calculate personalized scores for each task
        task_scores = []
        for task in tasks:
            personalized_score = self.calculate_personalized_score(task, user)
            task_scores.append((task, personalized_score))
        
        # Sort tasks by score in descending order
        task_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Format recommendations
        recommendations = []
        for task, score in task_scores[:limit]:
            recommendations.append({
                "task_id": task.task_id,
                "title": task.title,
                "score": score,
                "deadline": task.deadline.isoformat(),
                "priority": task.priority
            })
        
        return recommendations
    
    def get_user_insights(self, user_id: str) -> Dict:
        """
        Get insights about a user's preferences and performance
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with user insights
        """
        # Get user allocations
        user_allocations = self.allocation_history.get_user_allocations(user_id)
        
        if not user_allocations:
            return {
                "message": "No allocation history available for this user",
                "top_skills": [],
                "preferred_task_types": [],
                "performance_trends": {},
                "satisfaction_factors": []
            }
        
        # Analyze top skills
        skill_counts = defaultdict(int)
        for alloc in user_allocations:
            for skill in alloc["required_skills"]:
                skill_counts[skill] += 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Analyze preferred task types
        task_words = defaultdict(int)
        for alloc in user_allocations:
            # Fix: Check if user_satisfaction is None before comparison
            satisfaction = alloc.get("user_satisfaction")
            if satisfaction is not None and satisfaction >= 4:  # High satisfaction
                # Extract words from title and description
                words = (alloc["task_title"] + " " + alloc["task_description"]).lower().split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        task_words[word] += 1
        
        preferred_task_types = sorted(task_words.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyze performance trends
        completed_allocations = [a for a in user_allocations if a.get("completed", False)]
        performance_trends = {}
        
        if completed_allocations:
            # Group by month
            monthly_data = defaultdict(lambda: {"count": 0, "on_time": 0, "quality": 0})
            
            for alloc in completed_allocations:
                if "allocated_at" in alloc:
                    date = datetime.fromisoformat(alloc["allocated_at"]).strftime("%Y-%m")
                    monthly_data[date]["count"] += 1
                    
                    if alloc.get("on_time", False):
                        monthly_data[date]["on_time"] += 1
                    
                    if alloc.get("completion_quality") is not None:
                        monthly_data[date]["quality"] += alloc["completion_quality"]
            
            # Calculate monthly averages
            for month, data in monthly_data.items():
                if data["count"] > 0:
                    performance_trends[month] = {
                        "count": data["count"],
                        "on_time_rate": data["on_time"] / data["count"],
                        "avg_quality": data["quality"] / data["count"]
                    }
        
        # Analyze satisfaction factors
        satisfaction_factors = []
        
        # Check if high priority correlates with satisfaction
        high_priority_satisfaction = [
            a.get("user_satisfaction", 0) for a in user_allocations 
            if a.get("priority", 0) >= 4 and a.get("user_satisfaction") is not None
        ]
        
        low_priority_satisfaction = [
            a.get("user_satisfaction", 0) for a in user_allocations 
            if a.get("priority", 0) <= 2 and a.get("user_satisfaction") is not None
        ]
        
        if high_priority_satisfaction and low_priority_satisfaction:
            avg_high = sum(high_priority_satisfaction) / len(high_priority_satisfaction)
            avg_low = sum(low_priority_satisfaction) / len(low_priority_satisfaction)
            
            if avg_high > avg_low + 0.5:
                satisfaction_factors.append("Prefers high-priority tasks")
            elif avg_low > avg_high + 0.5:
                satisfaction_factors.append("Prefers low-priority tasks")
        
        # Check if certain skills correlate with satisfaction
        skill_satisfaction = defaultdict(list)
        for alloc in user_allocations:
            if alloc.get("user_satisfaction") is not None:
                for skill in alloc["required_skills"]:
                    skill_satisfaction[skill].append(alloc["user_satisfaction"])
        
        for skill, ratings in skill_satisfaction.items():
            if len(ratings) >= 3:  # Need at least 3 data points
                avg_rating = sum(ratings) / len(ratings)
                if avg_rating >= 4.0:
                    satisfaction_factors.append(f"High satisfaction with {skill} tasks")
                elif avg_rating <= 2.5:
                    satisfaction_factors.append(f"Low satisfaction with {skill} tasks")
        
        return {
            "top_skills": [{"skill": s, "count": c} for s, c in top_skills],
            "preferred_task_types": [{"word": w, "count": c} for w, c in preferred_task_types],
            "performance_trends": performance_trends,
            "satisfaction_factors": satisfaction_factors
        }


# Example usage
if __name__ == "__main__":
    from task_matching_algorithm import Task, UserProfile, TaskAllocation
    
    # Create sample users
    users = [
        UserProfile(
            user_id="user-123",
            name="John Doe",
            skills={"Python": 5, "JavaScript": 3, "React": 2},
            availability=[
                {"start": "2025-04-03T09:00:00", "end": "2025-04-03T17:00:00"},
                {"start": "2025-04-04T09:00:00", "end": "2025-04-04T17:00:00"}
            ],
            preferences={"preferred_task_types": ["backend", "data-processing"]},
            performance_metrics={"task_completion_rate": 0.95, "on_time_rate": 0.9},
            current_workload=0.5
        ),
        UserProfile(
            user_id="user-124",
            name="Jane Smith",
            skills={"Python": 4, "JavaScript": 5, "React": 5},
            availability=[
                {"start": "2025-04-03T13:00:00", "end": "2025-04-03T18:00:00"},
                {"start": "2025-04-04T09:00:00", "end": "2025-04-04T18:00:00"}
            ],
            preferences={"preferred_task_types": ["frontend", "ui-design"]},
            performance_metrics={"task_completion_rate": 0.98, "on_time_rate": 0.85},
            current_workload=0.7
        ),
        UserProfile(
            user_id="user-125",
            name="Alex Johnson",
            skills={"Python": 3, "JavaScript": 4, "React": 4, "Data Analysis": 5},
            availability=[
                {"start": "2025-04-03T09:00:00", "end": "2025-04-03T15:00:00"},
                {"start": "2025-04-04T09:00:00", "end": "2025-04-04T15:00:00"}
            ],
            preferences={"preferred_task_types": ["data-analysis", "visualization"]},
            performance_metrics={"task_completion_rate": 0.92, "on_time_rate": 0.95},
            current_workload=0.3
        )
    ]
    
    # Create sample tasks
    tasks = [
        Task(
            task_id="task-789",
            title="Implement login feature",
            description="Create login functionality with JWT authentication",
            required_skills={"Python": 3, "JavaScript": 3},
            priority=4,
            deadline=datetime.fromisoformat("2025-04-10T17:00:00"),
            estimated_duration=timedelta(hours=4)
        ),
        Task(
            task_id="task-790",
            title="Design user dashboard",
            description="Create UI design for the main user dashboard",
            required_skills={"React": 4, "JavaScript": 3},
            priority=3,
            deadline=datetime.fromisoformat("2025-04-08T17:00:00"),
            estimated_duration=timedelta(hours=6)
        ),
        Task(
            task_id="task-791",
            title="Data analysis for Q1 report",
            description="Analyze user engagement data for Q1 2025 report",
            required_skills={"Python": 3, "Data Analysis": 4},
            priority=5,
            deadline=datetime.fromisoformat("2025-04-15T17:00:00"),
            estimated_duration=timedelta(hours=8)
        )
    ]
    
    # Initialize personalized task matcher
    matcher = PersonalizedTaskMatcher()
    
    # Create some sample allocation history
    for i in range(10):
        # Randomly select a user and task
        import random
        user = random.choice(users)
        task = random.choice(tasks)
        
        # Create a sample allocation result
        allocation_result = {
            "best_match_user_id": user.user_id,
            "confidence_score": random.uniform(0.7, 0.95),
            "allocation_reason": f"Sample allocation {i+1}",
            "alternative_users": []
        }
        
        # Record the allocation
        matcher.record_allocation(task, user, allocation_result)
        
        # Record a random outcome
        allocation_id = matcher.allocation_history.allocations[-1]["allocation_id"]
        completed = random.random() > 0.2  # 80% chance of completion
        
        completion_time = None
        completion_quality = None
        on_time = None
        user_satisfaction = None
        
        if completed:
            completion_time = datetime.now()
            completion_quality = random.uniform(3.0, 5.0)
            on_time = random.random() > 0.3  # 70% chance of on-time
            user_satisfaction = random.uniform(2.5, 5.0)
        
        matcher.record_allocation_outcome(
            allocation_id,
            completed=completed,
            completion_time=completion_time,
            completion_quality=completion_quality,
            on_time=on_time,
            user_satisfaction=user_satisfaction
        )
    
    # Train the preference model
    matcher.preference_model.train(matcher.allocation_history.allocations)
    
    # Test personalized matching
    print("Personalized Task Matching:")
    for task in tasks:
        print(f"\nMatching task: {task.title}")
        result = matcher.match_task_to_users(task, users)
        
        matched_user = next((u for u in users if u.user_id == result["best_match_user_id"]), None)
        if matched_user:
            print(f"Matched to: {matched_user.name}")
            print(f"Confidence score: {result['confidence_score']:.2f}")
            print(f"Reason: {result['allocation_reason']}")
            
            if result["alternative_users"]:
                print("Alternative users:")
                for alt in result["alternative_users"]:
                    alt_user = next((u for u in users if u.user_id == alt["user_id"]), None)
                    if alt_user:
                        print(f"- {alt_user.name} (score: {alt['confidence_score']:.2f})")
    
    # Test personalized recommendations
    print("\nPersonalized Recommendations:")
    for user in users:
        print(f"\nRecommendations for {user.name}:")
        recommendations = matcher.get_personalized_recommendations(user, tasks)
        
        for rec in recommendations:
            print(f"- {rec['title']} (score: {rec['score']:.2f})")
    
    # Test user insights
    print("\nUser Insights:")
    for user in users:
        print(f"\nInsights for {user.name}:")
        insights = matcher.get_user_insights(user.user_id)
        
        if insights.get("top_skills"):
            print("Top skills:")
            for skill in insights["top_skills"]:
                print(f"- {skill['skill']}: {skill['count']} tasks")
        
        if insights.get("satisfaction_factors"):
            print("Satisfaction factors:")
            for factor in insights["satisfaction_factors"]:
                print(f"- {factor}")
