"""Allocation board visualization"""

from typing import Dict, List
import streamlit as st
from task_matching_algorithm import Task, UserProfile

class AllocationBoard:
    """Visualizes task allocations and user workloads"""
    
    def __init__(self):
        self.workload_levels = {
            "low": (0.0, 0.4),
            "medium": (0.4, 0.7),
            "high": (0.7, 1.0)
        }
    
    def get_workload_level(self, workload: float) -> str:
        """Get the workload level category for a user"""
        for level, (min_val, max_val) in self.workload_levels.items():
            if min_val <= workload < max_val:
                return level
        return "high"  # Default to high if workload is 1.0
    
    def display_allocation_board(self, users: List[UserProfile], tasks: Dict[str, Task], allocations: Dict[str, Dict]) -> None:
        """Display the allocation board with current task and user status"""
        st.subheader("Task Allocation Board")
        
        # Group users by workload level
        workload_groups = {
            "low": [],
            "medium": [],
            "high": []
        }
        
        for user in users:
            workload_level = self.get_workload_level(user.current_workload)
            workload_groups[workload_level].append(user)
        
        # Display users by workload level
        for level, user_group in workload_groups.items():
            if user_group:
                st.subheader(f"{level.capitalize()} Workload Users")
                for user in user_group:
                    with st.expander(f"{user.name} (Workload: {user.current_workload:.1%})"):
                        # Show user's allocated tasks
                        user_allocations = [
                            alloc for alloc in allocations.values()
                            if alloc["user_id"] == user.user_id and alloc["status"] == "ALLOCATED"
                        ]
                        
                        if user_allocations:
                            st.write("**Allocated Tasks:**")
                            for alloc in user_allocations:
                                task = tasks.get(alloc["task_id"])
                                if task:
                                    st.write(f"- {task.title} (Status: {task.status})")
                                    st.write(f"  Confidence: {alloc['confidence_score']:.2%}")
                                    st.write(f"  Reason: {alloc['allocation_reason']}")
                        else:
                            st.info("No tasks allocated")
                        
                        # Show user's skills and preferences
                        st.write("**Skills:**")
                        for skill, level in user.skills.items():
                            st.write(f"- {skill}: {'⭐' * level}")
                        
                        st.write("**Preferences:**")
                        st.write(f"- Task Types: {', '.join(user.preferences.get('preferred_task_types', []))}")
                        st.write(f"- Complexity: {user.preferences.get('task_complexity_preference', 'Not specified')}")
        
        # Display unallocated tasks
        unallocated_tasks = [
            task for task in tasks.values()
            if task.status != "ASSIGNED" and not any(
                alloc["task_id"] == task.task_id and alloc["status"] == "ALLOCATED"
                for alloc in allocations.values()
            )
        ]
        
        if unallocated_tasks:
            st.subheader("Unallocated Tasks")
            for task in unallocated_tasks:
                with st.expander(f"{task.title} (Priority: {task.priority})"):
                    st.write(f"Status: {task.status}")
                    st.write(f"Required Skills: {', '.join(task.required_skills.keys())}")
                    st.write(f"Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}")
        else:
            st.info("No unallocated tasks") 