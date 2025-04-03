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
import json # Import json

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
    
    def get_eligible_users(self, task: Task, users: List[UserProfile], simulated_now: Optional[datetime] = None) -> List[UserProfile]:
        """Find users who meet basic criteria (skills, availability, workload)."""
        
        if simulated_now is None:
            simulated_now = datetime.now(pytz.UTC)
        
        eligible = []
        for user in users:
            # 1. Skill Check
            has_required_skills = True
            if task.required_skills:
                for skill, level in task.required_skills.items():
                    if user.skills.get(skill, 0) < level:
                        has_required_skills = False
                        break
            if not has_required_skills:
                continue
            
            # 2. Workload Check (simple threshold)
            if user.current_workload >= 1.0:
                continue

            # --- Modified Availability Check --- #
            required_duration = task.estimated_duration
            
            # Check deadline first
            if task.deadline.astimezone(pytz.UTC) <= simulated_now:
                continue
                
            # Get the specific user's availability object
            user_availability = self.availability_manager.get_user_availability(user.user_id)
            
            # Calculate max days to search ahead
            days_diff = (task.deadline.astimezone(pytz.UTC) - simulated_now).days
            max_days = max(1, min(days_diff + 1, 30)) # Search at least today, up to 30 days

            # Check if there is *any* slot available between now and the deadline
            next_slot = user_availability.find_next_available_slot(
                start_time=simulated_now, 
                duration=required_duration,
                max_days_ahead=max_days # Use max_days_ahead
            )
            
            if next_slot is None:
                 # print(f"DEBUG: User {user.user_id} has NO slot for task {task.task_id} duration {required_duration} before deadline {task.deadline}") # Optional Debug
                 continue # Skip user if no suitable slot found before deadline
            # --- End Modified Availability Check --- #
            
            # print(f"DEBUG: User {user.user_id} IS eligible for task {task.task_id}") # Optional Debug
            eligible.append(user)
            
        return eligible
    
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
    
    def get_allocation_suggestions(self, task: Task, users: List[UserProfile], simulated_now: Optional[datetime] = None) -> Dict:
        """Get allocation suggestions for a task, considering eligibility with simulated time."""
        
        if simulated_now is None:
            simulated_now = datetime.now(pytz.UTC)
            
        # Get eligible users based on skills AND availability starting from simulated_now
        eligible_users = self.get_eligible_users(task, users, simulated_now)
        if not eligible_users:
            return None
        
        # Use personalization engine (which now uses the LLM engine first)
        # Pass eligible users to the personalization matcher
        allocation = self.personalization_engine.match_task_to_users(task, eligible_users)
        
        # Find the full user profile for the best match ID from the original users list
        matched_user_profile = None
        if allocation["best_match_user_id"]:
            # Ensure we look in the original 'users' list passed to the function
            matched_user_profile = next((u for u in users if u.user_id == allocation["best_match_user_id"]), None)

        if matched_user_profile:
            return {
                "best_match": matched_user_profile, # Return the full profile
                "confidence_score": allocation["confidence_score"],
                "allocation_reason": allocation["allocation_reason"],
                "alternative_users": allocation["alternative_users"]
            }
        
        return None # Return None if no best match ID or profile found 

    def _get_availability_summary(self, user_id: str, start_time: datetime, lookahead_days: int = 7) -> str:
        """Generates a brief summary of user availability."""
        try:
            user_availability = self.availability_manager.get_user_availability(user_id)
            end_time = start_time + timedelta(days=lookahead_days)
            available_slots = user_availability.get_available_slots(start_time, end_time)
            busy_slots = user_availability.get_busy_slots(start_time, end_time)
            
            if not available_slots and not busy_slots:
                return "Availability not specified, assume generally available during standard hours."
            
            summary = "Available slots: " + ", ".join([f"{s.start_time.strftime('%a %H:%M')}-{s.end_time.strftime('%H:%M')}" for s in available_slots[:3]])
            if len(available_slots) > 3:
                summary += "..."
            summary += " | Busy slots: " + ", ".join([f"{s.start_time.strftime('%a %H:%M')}-{s.end_time.strftime('%H:%M')}" for s in busy_slots[:2]])
            if len(busy_slots) > 2:
                 summary += "..."
            return summary
        except Exception:
            return "Could not retrieve detailed availability."

    def create_llm_allocation_plan(self, 
                                 tasks: List[Task], 
                                 users: List[UserProfile], 
                                 simulated_now: datetime) -> List[Dict]:
        """Generate a holistic allocation plan using LLM reasoning."""
        
        if not tasks or not users:
            return []

        # Prepare simplified user data for the prompt
        users_prompt_data = []
        for user in users:
            availability_summary = self._get_availability_summary(user.user_id, simulated_now)
            users_prompt_data.append({
                "user_id": user.user_id,
                "name": user.name,
                "skills": user.skills,
                "current_workload": user.current_workload,
                "availability_summary": availability_summary
                # Add preferences/performance summary if desired later
            })

        # Prepare task data
        tasks_prompt_data = [t.to_dict() for t in tasks]

        # Define the LLM prompt for planning
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Task Allocation Planner. Your goal is to create an optimal plan assigning the given tasks to the available users.
            
            Guidelines:
            1.  **Prioritize:** Assign higher priority tasks and those with earlier deadlines first.
            2.  **Match Skills:** Match tasks to users with relevant skills and sufficient proficiency. Consider the task context (like in the TaskMatchingEngine prompt).
            3.  **Check Availability & Workload:** Use the provided availability summary and current workload. A user might handle multiple tasks if they have the capacity and time before respective deadlines.
            4.  **Balance Load:** Aim for a reasonably balanced workload across users, but prioritize getting tasks assigned effectively over perfect balance.
            5.  **Assign if Possible:** Try to assign every task, but it's okay to leave a task unassigned if no suitable user is found.
            6.  **Reasoning:** Provide a concise reason for each assignment based on the above factors.

            Input:
            - List of tasks with id, title, required_skills, priority, deadline, estimated_duration_minutes.
            - List of users with id, name, skills, current_workload, availability_summary.
            - Current Time: {current_time}

            Output ONLY a JSON list of allocation objects, where each object represents ONE task assignment:
            [ 
                {{
                    "task_id": "task-id-string", 
                    "user_id": "user-id-string", 
                    "reason": "Concise reason for this specific assignment (skills, availability, priority etc.)." 
                }},
                # ... more allocations ...
            ]
            If a task cannot be assigned, do not include it in the output list.
            """),
            ("human", "Tasks:\n{tasks_json}\n\nUsers:\n{users_json}\n\nCurrent Time: {current_time}")
        ])

        planning_chain = planning_prompt | self.llm | JsonOutputParser()

        try:
            print("--- Calling LLM for Allocation Plan --- ") # Debug
            plan_result = planning_chain.invoke({
                "tasks_json": json.dumps(tasks_prompt_data, default=str),
                "users_json": json.dumps(users_prompt_data, default=str),
                "current_time": simulated_now.isoformat()
            })
            
            # Validate the result structure (should be a list of dicts)
            if isinstance(plan_result, list):
                 # Basic validation of list items
                 validated_plan = [
                     item for item in plan_result 
                     if isinstance(item, dict) and 
                        'task_id' in item and 'user_id' in item and 'reason' in item
                 ]
                 print(f"--- LLM Allocation Plan Generated ({len(validated_plan)} assignments) --- ") # Debug
                 return validated_plan
            else:
                print(f"LLM plan result was not a list: {type(plan_result)}")
                return []

        except Exception as e:
            print(f"Error creating LLM allocation plan: {e}")
            return [] 