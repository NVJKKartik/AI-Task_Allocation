"""Automated task allocation system"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pytz
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
import sys
import os
import numpy as np

from task_matching_algorithm import Task, UserProfile, TaskAllocation, TaskMatchingEngine
from personalization import PersonalizedTaskMatcher, AllocationHistory
from availability_management import AvailabilityManager

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add project root to path

class AutomatedAllocator:
    """Handles automated task allocation with feedback system"""
    
    def __init__(self, 
                 personalization_engine: PersonalizedTaskMatcher,
                 task_matching_engine: TaskMatchingEngine,
                 availability_manager: Optional[AvailabilityManager] = None):
        """
        Initialize the automated allocator
        
        Args:
            personalization_engine: The personalization engine instance
            task_matching_engine: The core LLM matching engine instance
            availability_manager: The availability manager instance
        """
        self.personalization_engine = personalization_engine
        self.task_matching_engine = task_matching_engine
        self.availability_manager = availability_manager or AvailabilityManager()
        self.llm = ChatOpenAI(temperature=0)
    
    def find_unallocated_tasks(self, tasks: Dict[str, Task], allocations: Dict[str, Dict]) -> List[Task]:
        """Find tasks that haven't been allocated yet"""
        return [
            task for task in tasks.values()
            if task.status != "ASSIGNED" and not any(
                alloc["task_id"] == task.task_id and alloc["status"] == "ALLOCATED"
                for alloc in allocations.values()
            )
        ]
    
    def get_eligible_users(self, task: Task, users: List[UserProfile]) -> List[UserProfile]:
        """Get users who are eligible for a task based on skills and workload"""
        eligible_users = []
        for user in users:
            if user.current_workload >= 1.0:  # Skip fully loaded users
                continue
                
            # Calculate skill match percentage
            skill_matches = 0
            total_skills = len(task.required_skills)
            for skill, required_level in task.required_skills.items():
                user_level = user.skills.get(skill, 0)
                if user_level >= required_level * 0.6:  # 60% of required level
                    skill_matches += 1
            
            skill_match_percentage = skill_matches / total_skills if total_skills > 0 else 1.0
            
            # Include user if they match at least 50% of skills
            if skill_match_percentage >= 0.5:
                eligible_users.append(user)
        
        return eligible_users
    
    def generate_allocation_plan(self, tasks: Dict[str, Task], users: List[UserProfile], allocations: Dict[str, Dict]) -> List[Dict]:
        """Generate a plan for allocating tasks to users"""
        unallocated_tasks = self.find_unallocated_tasks(tasks, allocations)
        if not unallocated_tasks:
            return []
        
        # Sort tasks by priority and deadline
        sorted_tasks = sorted(
            unallocated_tasks,
            key=lambda t: (t.priority, t.deadline),
            reverse=True
        )
        
        allocation_plan = []
        for task in sorted_tasks:
            eligible_users = self.get_eligible_users(task, users)
            if not eligible_users:
                continue
            
            # Use personalization engine for matching
            allocation = self.personalization_engine.match_task_to_users(task, eligible_users)
            
            if allocation["best_match_user_id"]:
                matched_user = next(u for u in users if u.user_id == allocation["best_match_user_id"])
                allocation_plan.append({
                    "task": task,
                    "user": matched_user,
                    "confidence_score": allocation["confidence_score"],
                    "allocation_reason": allocation["allocation_reason"],
                    "feedback": None,
                    "adjusted": False
                })
                # Temporarily increase user's workload for next allocations
                matched_user.current_workload = min(1.0, matched_user.current_workload + 0.2)
        
        return allocation_plan
    
    def adjust_allocation(self, allocation: Dict, users: List[UserProfile]) -> Dict:
        """Adjust an allocation based on feedback"""
        # Get alternative users
        alternative_users = [
            user for user in users
            if user.user_id != allocation["user"].user_id and
            user.current_workload < 1.0
        ]
        
        if not alternative_users:
            return allocation
        
        # Use LLM to analyze and suggest better matches
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze the task and suggest better user matches based on:
            1. Task requirements and user skills
            2. Current workload distribution
            3. Historical performance
            
            Output format:
            {{
                "suggested_users": [
                    {{
                        "user_id": "user_id",
                        "reason": "matching reason"
                    }}
                ],
                "adjustment_reason": "why this adjustment is better"
            }}"""),
            ("user", f"""Task: {allocation['task'].title}
            Current User: {allocation['user'].name}
            Required Skills: {allocation['task'].required_skills}
            Alternative Users: {[{'id': u.user_id, 'name': u.name, 'skills': u.skills} for u in alternative_users]}""")
        ])
        
        analysis_parser = JsonOutputParser()
        analysis_chain = analysis_prompt | self.llm | analysis_parser
        
        try:
            analysis = analysis_chain.invoke({})
            if analysis["suggested_users"]:
                # Update the allocation with the best suggested user
                best_suggestion = analysis["suggested_users"][0]
                new_user = next(
                    u for u in users 
                    if u.user_id == best_suggestion["user_id"]
                )
                
                # Update workload
                allocation["user"].current_workload = max(0.0, allocation["user"].current_workload - 0.2)
                new_user.current_workload = min(1.0, new_user.current_workload + 0.2)
                
                # Update allocation
                allocation["user"] = new_user
                allocation["allocation_reason"] = best_suggestion["reason"]
                allocation["adjusted"] = True
                allocation["feedback"] = "adjust"
                
                return allocation
        except Exception as e:
            print(f"Error analyzing alternatives: {str(e)}")
        
        return allocation
    
    def implement_allocations(self, allocation_plan: List[Dict], allocations: Dict[str, Dict]) -> None:
        """Implement approved allocations"""
        for alloc in allocation_plan:
            if alloc["feedback"] == "approve":
                allocation_id = f"alloc-{len(allocations) + 1000}"
                allocations[allocation_id] = {
                    "task_id": alloc["task"].task_id,
                    "user_id": alloc["user"].user_id,
                    "allocated_at": datetime.now(pytz.UTC),
                    "status": "ALLOCATED",
                    "confidence_score": alloc["confidence_score"],
                    "allocation_reason": alloc["allocation_reason"]
                }
                
                # Update task status
                alloc["task"].status = "ASSIGNED"
                
                # Record in personalization engine
                allocation_data = {
                    "best_match_user_id": alloc["user"].user_id,
                    "confidence_score": alloc["confidence_score"],
                    "allocation_reason": alloc["allocation_reason"],
                    "alternative_users": []
                }
                self.personalization_engine.record_allocation(
                    alloc["task"], alloc["user"], allocation_data
                )
    
    def get_allocation_suggestions(self, task: Task, users: List[UserProfile]) -> Dict:
        """Get allocation suggestions for a task"""
        # Get eligible users
        eligible_users = self.get_eligible_users(task, users)
        if not eligible_users:
            return None
        
        # Use personalization engine (which now uses the LLM engine first)
        allocation = self.personalization_engine.match_task_to_users(task, eligible_users)
        
        # Find the full user profile for the best match ID
        matched_user_profile = None
        if allocation["best_match_user_id"]:
            matched_user_profile = next((u for u in users if u.user_id == allocation["best_match_user_id"]), None)

        if matched_user_profile:
            return {
                "best_match": matched_user_profile, # Return the full profile
                "confidence_score": allocation["confidence_score"],
                "allocation_reason": allocation["allocation_reason"],
                "alternative_users": allocation["alternative_users"]
            }
        
        return None # Return None if no best match ID or profile found 