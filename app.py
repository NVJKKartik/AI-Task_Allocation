"""
User Interface for AI Task Allocation Agent

This module implements a Streamlit-based user interface for the AI task allocation agent.
It provides a way to interact with the task matching algorithm and availability management system.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the Python path
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz
import calendar
from typing import List, Dict, Any, Optional
import uuid
from enum import Enum
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from automated_allocation import AutomatedAllocator
from personalization import PersonalizedTaskMatcher, AllocationHistory
from task_matching_algorithm import Task as TaskModel, UserProfile, TaskMatchingEngine
from availability_management import (
    TimeSlot, 
    RecurringTimeSlot, 
    UserAvailability, 
    AvailabilityManager
)
from task_automation import TaskAnalyzer
from smart_availability import SmartAvailabilityManager
from communication_manager import CommunicationManager
from learning_system import LearningSystem
from progress_tracking import ProgressTracker
from allocation_board import AllocationBoard
from ai_agent import TaskAllocationAgent
from sample_data import get_sample_data

# Set page configuration
st.set_page_config(
    page_title="AI Task Allocation Agent",
    page_icon="✅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
load_dotenv()

# Initialize session state for storing data
if 'users' not in st.session_state:
    st.session_state.users = {}
if 'tasks' not in st.session_state:
    st.session_state.tasks = {}
if 'allocations' not in st.session_state:
    st.session_state.allocations = {}
if 'availability_manager' not in st.session_state:
    st.session_state.availability_manager = AvailabilityManager()
if 'task_matching_engine' not in st.session_state:
    st.session_state.task_matching_engine = TaskMatchingEngine()
if 'personalization_engine' not in st.session_state:
    st.session_state.personalization_engine = PersonalizedTaskMatcher(
        task_matching_engine=st.session_state.task_matching_engine
    )
if 'task_analyzer' not in st.session_state:
    st.session_state.task_analyzer = TaskAnalyzer()
if 'smart_availability_manager' not in st.session_state:
    st.session_state.smart_availability_manager = SmartAvailabilityManager()
if 'communication_manager' not in st.session_state:
    st.session_state.communication_manager = CommunicationManager()
if 'learning_system' not in st.session_state:
    st.session_state.learning_system = LearningSystem()
if 'progress_tracker' not in st.session_state:
    st.session_state.progress_tracker = ProgressTracker()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
# Initialize Simulated Time
if 'simulated_now' not in st.session_state:
    st.session_state.simulated_now = datetime.now(pytz.UTC)

# Initialize components
personalization_engine = st.session_state.personalization_engine
task_matching_engine = st.session_state.task_matching_engine
availability_manager = st.session_state.availability_manager
automated_allocator = AutomatedAllocator(
    personalization_engine=personalization_engine,
    task_matching_engine=task_matching_engine,
    availability_manager=availability_manager
)
allocation_board = AllocationBoard()

# Initialize AI agent
if 'ai_agent' not in st.session_state:
    st.session_state.ai_agent = TaskAllocationAgent()

# Function to load sample data
def load_sample_data():
    # Call the function from the new module
    sample_users, sample_tasks, availability_manager = get_sample_data()
    
    # Update session state
    st.session_state.users = sample_users
    st.session_state.tasks = sample_tasks
    st.session_state.availability_manager = availability_manager
    
    # Re-initialize personalization engine (optional, clears learned data)
    st.session_state.personalization_engine = PersonalizedTaskMatcher(
        task_matching_engine=st.session_state.task_matching_engine,
        allocation_history=AllocationHistory() # Start fresh history
    )
    
    # Reset allocations
    st.session_state.allocations = {}

    st.success(f"Generated {len(sample_users)} users and {len(sample_tasks)} tasks.")

# Function to match a task to users
def match_task_to_users(task_id):
    if task_id not in st.session_state.tasks:
        st.error(f"Task {task_id} not found")
        return None
    
    task = st.session_state.tasks[task_id]
    users = list(st.session_state.users.values())
    
    if not users:
        st.error("No users available for matching")
        return None
    
    try:
        engine = TaskMatchingEngine()
        allocation = engine.match_task_to_users(task, users)
        
        # Store the allocation
        st.session_state.allocations[task_id] = allocation
        
        return allocation
    except Exception as e:
        st.error(f"Error matching task: {str(e)}")
        return None

# Function to create a radar chart for skill comparison
def create_skill_radar_chart(task, user):
    # Get all unique skills from task and user
    all_skills = set(task.required_skills.keys()) | set(user.skills.keys())
    
    # Create data for the radar chart
    categories = list(all_skills)
    task_values = [task.required_skills.get(skill, 0) for skill in categories]
    user_values = [user.skills.get(skill, 0) for skill in categories]
    
    # Create the radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=task_values,
        theta=categories,
        fill='toself',
        name='Required Skills'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=user_values,
        theta=categories,
        fill='toself',
        name='User Skills'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 5]
            )
        ),
        showlegend=True,
        title="Skill Comparison"
    )
    
    return fig

# Function to create a calendar heatmap
def create_calendar_heatmap(user_id: str, year: int, month: int):
    """Create a calendar heatmap showing user availability"""
    # Get user's availability
    user_availability = st.session_state.availability_manager.get_user_availability(user_id)
    
    # Get the first and last day of the month
    first_day = datetime(year, month, 1, tzinfo=pytz.UTC)
    last_day = datetime(year, month, calendar.monthrange(year, month)[1], 23, 59, 59, tzinfo=pytz.UTC)
    
    # Get all available and busy slots for the month
    available_slots = user_availability.get_available_slots(first_day, last_day)
    busy_slots = user_availability.get_busy_slots(first_day, last_day)
    
    # Create a DataFrame for the heatmap
    days = []
    availability = []
    
    # For each day in the month
    for day in range(1, calendar.monthrange(year, month)[1] + 1):
        current_date = datetime(year, month, day, tzinfo=pytz.UTC)
        end_of_day = datetime(year, month, day, 23, 59, 59, tzinfo=pytz.UTC)
        
        # Check if user is available at any point during the day
        is_available = False
        for slot in available_slots:
            if slot.overlaps(TimeSlot(current_date, end_of_day, "TEMP")):
                is_available = True
                break
        
        days.append(day)
        availability.append(1 if is_available else 0)
    
    # Create the heatmap
    df = pd.DataFrame({
        'Day': days,
        'Availability': availability
    })
    
    fig = px.imshow(
        df.pivot_table(index='Day', values='Availability', aggfunc='mean').values.reshape(1, -1),
        x=days,
        color_continuous_scale='RdYlGn',
        labels=dict(x="Day of Month", color="Availability"),
        title=f"Availability Calendar - {calendar.month_name[month]} {year}"
    )
    
    fig.update_layout(height=200)
    
    return fig

def show_smart_availability():
    """Show smart availability features"""
    st.header("Smart Availability")
    
    # User selection
    user_id = st.selectbox("Select User", list(st.session_state.users.keys()))
    if user_id:
        user = st.session_state.users[user_id]
        
        # Optimal working hours
        st.subheader("Optimal Working Hours")
        historical_data = st.session_state.progress_tracker.task_history.get(user_id, [])
        optimal_hours = st.session_state.smart_availability_manager.suggest_optimal_hours(user_id, historical_data)
        st.json({
            "Start Time": optimal_hours.start_time.strftime("%H:%M"),
            "End Time": optimal_hours.end_time.strftime("%H:%M"),
            "Timezone": optimal_hours.timezone,
            "Work Days": [calendar.day_name[d] for d in optimal_hours.work_days]
        })
        
        # Calendar integration
        st.subheader("Calendar Integration")
        calendar_id = st.text_input("Enter Calendar ID (optional)")
        if calendar_id:
            if st.button("Sync Calendar"):
                events = st.session_state.smart_availability_manager.sync_with_calendar(user_id, calendar_id)
                st.json(events)
        
        # Availability schedule
        st.subheader("Availability Schedule")
        start_date = st.date_input("Start Date")
        end_date = st.date_input("End Date")
        if start_date and end_date:
            schedule = st.session_state.smart_availability_manager.generate_availability_schedule(
                user_id,
                datetime.combine(start_date, datetime.min.time()),
                datetime.combine(end_date, datetime.max.time())
            )
            st.json(schedule)

def show_communication():
    """Show communication features"""
    st.header("Communication")
    
    # Notification settings
    st.subheader("Notification Settings")
    notification_type = st.selectbox(
        "Notification Type",
        ["email", "slack", "discord", "push"]
    )
    
    # Task assignment notification
    st.subheader("Task Assignment")
    task_id = st.selectbox("Select Task", list(st.session_state.tasks.keys()))
    user_id = st.selectbox("Select User", list(st.session_state.users.keys()))
    if st.button("Send Assignment Notification"):
        if task_id and user_id:
            success = st.session_state.communication_manager.send_task_assignment_notification(
                user_id,
                st.session_state.tasks[task_id]
            )
            if success:
                st.success("Notification sent successfully!")
            else:
                st.error("Failed to send notification")
    
    # Deadline reminders
    st.subheader("Deadline Reminders")
    days_until_deadline = st.slider("Days Until Deadline", 1, 30, 7)
    if st.button("Send Deadline Reminder"):
        if task_id and user_id:
            success = st.session_state.communication_manager.send_deadline_reminder(
                user_id,
                st.session_state.tasks[task_id],
                days_until_deadline
            )
            if success:
                st.success("Reminder sent successfully!")
        else:
                st.error("Failed to send reminder")

def show_learning():
    """Show learning and adaptation features"""
    st.header("Learning and Adaptation")
    
    # User selection
    user_id = st.selectbox("Select User", list(st.session_state.users.keys()))
    if user_id:
        # User preferences
        st.subheader("User Preferences")
        preferences = st.session_state.learning_system.user_preferences.get(user_id)
        if preferences:
            st.json({
                "Preferred Task Types": preferences.preferred_task_types,
                "Skill Preferences": preferences.skill_preferences,
                "Task Complexity Preference": preferences.task_complexity_preference,
                "Collaboration Preference": preferences.collaboration_preference
            })
        
        # Task recommendations
        st.subheader("Task Recommendations")
        if st.button("Get Recommendations"):
            recommendations = st.session_state.learning_system.get_recommendations(
                user_id,
                list(st.session_state.tasks.values())
            )
            for task in recommendations:
                st.json(task)
        
            if recommendations:
                try:
                    for task in recommendations:
                        st.markdown(f"#### Task ID: {task.task_id}")
                        st.write(f"**Title:** {task.title}")
                        st.write(f"**Description:** {task.description}")
                        st.write(f"**Required Skills:** {task.required_skills}")
                        st.write(f"**Priority:** {task.priority}")
                        st.write(f"**Deadline:** {task.deadline.strftime('%Y-%m-%d %H:%M')}")
                        
                        # Add YouTube and technology links
                        st.markdown("### Learning Resources")
                        for skill in task.required_skills:
                            skill_url = skill.replace(" ", "+")
                            st.write(f"- [Learn {skill} on YouTube](https://www.youtube.com/results?search_query={skill_url})")
                            st.write(f"- [Documentation for {skill}](https://www.google.com/search?q={skill_url}+documentation)")
                        
                        st.markdown("---")
                except TypeError:
                    st.error("Error displaying recommendations. Data may not be JSON serializable.")
        # User clustering
        st.subheader("User Clustering")
        if st.button("Cluster Users"):
            clusters = st.session_state.learning_system.cluster_users_by_preferences()
            st.json(clusters)

def show_progress_tracking():
    """Show progress tracking features"""
    st.header("Progress Tracking")
    
    # User selection
    user_id = st.selectbox("Select User", list(st.session_state.users.keys()))
    if user_id:
        # Performance metrics
        st.subheader("Performance Metrics")
        metrics = st.session_state.progress_tracker.get_performance_report(user_id)
        if metrics:
            st.json(metrics)
        
        # Progress visualization
        st.subheader("Progress Visualization")
        if st.button("Generate Visualization"):
            output_path = f"progress_{user_id}.png"
            success = st.session_state.progress_tracker.generate_progress_visualization(
                user_id,
                output_path
            )
            if success:
                st.image(output_path)
            else:
                st.error("Failed to generate visualization")
        
        # Milestone tracking
        st.subheader("Milestone Tracking")
        milestones = st.session_state.progress_tracker.milestones.get(user_id, [])
        if milestones:
            st.json(milestones)

def show_user_management():
    """Show user management interface"""
    st.header("User Management")
    
    # Load sample data button
    if st.button("Load Sample Data"):
        load_sample_data()
    
    # Add new user
    st.subheader("Add New User")
    
    user_id = st.text_input("User ID (e.g., user-123)")
    name = st.text_input("Name")
    
    # Skills input
    st.text("Skills")
    st.caption("Enter skills and proficiency levels in natural language (e.g., 'Expert in Python, intermediate JavaScript, beginner in React')")
    skills_text = st.text_area("Skills Description", height=100)
    
    # Preferences input
    st.text("Preferences")
    st.caption("Describe task preferences in natural language (e.g., 'I prefer backend development and data processing tasks')")
    preferences_text = st.text_area("Preferences Description", height=100)
    
    if st.button("Add User"):
        if not user_id or not name:
            st.error("User ID and Name are required")
            return
            
        try:
            # Process skills using LLM
            llm = ChatOpenAI(temperature=0)
            skills_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract skills and their levels from the text. Convert proficiency descriptions to numbers:
                1: Beginner
                2: Basic
                3: Intermediate
                4: Advanced
                5: Expert
                
                Output format should be a JSON object with skill names as keys and proficiency levels as values.
                
                Example input: Expert in Python, intermediate JavaScript, beginner in React
                Example output: {{"Python": 5, "JavaScript": 3, "React": 1}}"""),
                ("user", "{input}")
            ])
            
            # Process preferences using LLM
            prefs_prompt = ChatPromptTemplate.from_messages([
                ("system", """Extract task type preferences from the text.
                Output format should be a JSON object with a "preferred_task_types" array.
                
                Example input: I prefer backend development and data processing tasks
                Example output: {{"preferred_task_types": ["backend", "data-processing"]}}"""),
                ("user", "{input}")
            ])
            
            # Parse the outputs
            skills_parser = JsonOutputParser()
            prefs_parser = JsonOutputParser()
            
            # Get skills and preferences from LLM
            skills_chain = skills_prompt | llm | skills_parser
            prefs_chain = prefs_prompt | llm | prefs_parser
            
            skills = skills_chain.invoke({"input": skills_text}) if skills_text else {"General": 3}
            preferences = prefs_chain.invoke({"input": preferences_text}) if preferences_text else {"preferred_task_types": ["general"]}
            
            # Create user profile
            user = UserProfile(
                user_id=user_id,
                name=name,
                skills=skills,
                availability=[],  # Empty availability to be set later
                preferences=preferences,
                performance_metrics={"task_completion_rate": 0.0, "on_time_rate": 0.0},
                current_workload=0.0
            )
            
            st.session_state.users[user_id] = user
            st.success("User added successfully!")
            
        except Exception as e:
            st.error(f"Error adding user: {str(e)}")
    
    # View existing users
    st.subheader("Existing Users")
    if st.session_state.users:
        for user_id, user in st.session_state.users.items():
            with st.expander(f"{user.name} ({user_id})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text("Skills")
                    for skill, level in user.skills.items():
                        st.write(f"• {skill}: {'⭐' * level}")

                with col2:
                    st.text("Preferences")
                    if "preferred_task_types" in user.preferences:
                        for task_type in user.preferences["preferred_task_types"]:
                            st.write(f"• {task_type}")

                if user.performance_metrics:
                    st.text("Performance Metrics")
                    completion_rate = user.performance_metrics.get("task_completion_rate", 0)
                    on_time_rate = user.performance_metrics.get("on_time_rate", 0)
                    st.progress(completion_rate, text=f"Task Completion Rate: {completion_rate:.0%}")
                    st.progress(on_time_rate, text=f"On-time Rate: {on_time_rate:.0%}")
    else:
        st.info("No users added yet. Click 'Load Sample Data' to add sample users.")

def show_task_management():
    """Show task management interface"""
    st.header("Task Management")
    
    # Add new task
    st.subheader("Add New Task")
    
    task_id = st.text_input("Task ID (e.g., task-123)")
    title = st.text_input("Title")
    description = st.text_area("Description (optional)", height=100)
    
    # Task details
    col1, col2 = st.columns(2)
    with col1:
        priority = st.slider("Priority", 1, 5, 3, help="1: Low, 5: High")
        estimated_duration = st.number_input("Estimated Duration (hours)", min_value=1, max_value=40, value=4)
    with col2:
        deadline = st.date_input("Deadline")
        deadline_time = st.time_input("Deadline Time", value=datetime.strptime("17:00", "%H:%M").time())
    
    if st.button("Add Task"):
        if not task_id or not title:
            st.error("Task ID and Title are required")
            return
            
        try:
            # Use LLM to analyze the task and fill in missing details
            llm = ChatOpenAI(temperature=0)
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """Analyze the task title and description to extract or suggest:
                1. Required skills with minimum proficiency levels (1-5 scale)
                2. Task complexity assessment
                3. Suggested task breakdown
                4. Expanded description (if original is too brief)
                5. Any potential risks or dependencies
                
                Output format:
                {{
                    "required_skills": {{"skill_name": minimum_level}},
                    "complexity": "Low|Medium|High",
                    "task_breakdown": ["step1", "step2", ...],
                    "expanded_description": "detailed description",
                    "risks_and_dependencies": ["risk1", "risk2", ...]
                }}"""),
                ("user", "{input}")
            ])
            
            # Parse the outputs
            analysis_parser = JsonOutputParser()
            analysis_chain = analysis_prompt | llm | analysis_parser
            
            # Get task analysis
            analysis = analysis_chain.invoke({
                "input": f"Title: {title}\nDescription: {description or 'No description provided'}"
            })
            
            # Use the expanded description if original is empty or too short
            final_description = description if description and len(description) > 50 else analysis["expanded_description"]
            
            # Create task
            task = TaskModel(
                task_id=task_id,
                title=title,
                description=final_description,
                required_skills=analysis["required_skills"],
                priority=priority,
                deadline=datetime.combine(deadline, deadline_time, tzinfo=pytz.UTC),
                estimated_duration=timedelta(hours=estimated_duration)
            )
            
            st.session_state.tasks[task_id] = task
            
            # Show task analysis
            st.success("Task added successfully!")
            with st.expander("View Task Analysis"):
                st.subheader("Task Details")
                st.write(f"Description: {final_description}")
                
                st.subheader("Required Skills")
                for skill, level in analysis["required_skills"].items():
                    st.write(f"• {skill}: {'⭐' * level}")
                
                st.subheader("Complexity")
                st.write(analysis["complexity"])
                
                st.subheader("Suggested Task Breakdown")
                for i, step in enumerate(analysis["task_breakdown"], 1):
                    st.write(f"{i}. {step}")
                
                st.subheader("Risks and Dependencies")
                for risk in analysis["risks_and_dependencies"]:
                    st.write(f"• {risk}")
            
        except Exception as e:
            st.error(f"Error adding task: {str(e)}")
    
    # View existing tasks
    st.subheader("Existing Tasks")
    if st.session_state.tasks:
        for task_id, task in st.session_state.tasks.items():
            with st.expander(f"{task.title} ({task_id})"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text("Description")
                    st.write(task.description)
                    
                    st.text("Required Skills")
                    for skill, level in task.required_skills.items():
                        st.write(f"• {skill}: {'⭐' * level}")
                    
                    with col2:
                        st.text("Details")
                        st.write(f"• Priority: {'🔥' * task.priority}")
                        st.write(f"• Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}")
                        st.write(f"• Duration: {task.estimated_duration.total_seconds() / 3600:.1f} hours")
    else:
        st.info("No tasks added yet. Click 'Load Sample Data' to add sample tasks.")

class TaskStatus(Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class AllocationStatus(Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"

def show_ai_allocation():
    """Show AI-suggested task allocations generated by the LLM planner."""
    st.title("🤖 AI Task Allocation Plan")
    
    # Initialize session state if needed
    if 'allocations' not in st.session_state:
        st.session_state.allocations = {}
    if 'tasks' not in st.session_state:
        st.session_state.tasks = {}
    if 'users' not in st.session_state:
        st.session_state.users = {}

    # Get unallocated tasks and available users
    unallocated_tasks = [
        task for task in st.session_state.tasks.values()
        if task.status == TaskStatus.PENDING.value # Only suggest for PENDING tasks
    ]
    available_users = list(st.session_state.users.values())

    if not unallocated_tasks:
        st.info("No pending tasks available for AI allocation planning.")
        return
    if not available_users:
        st.warning("No users available to allocate tasks to.")
        return

    # Initialize allocator
    personalization_engine = st.session_state.personalization_engine
    task_matching_engine = st.session_state.task_matching_engine
    availability_manager = st.session_state.availability_manager
    automated_allocator = AutomatedAllocator(
        personalization_engine=personalization_engine,
        task_matching_engine=task_matching_engine,
        availability_manager=availability_manager
    )

    # --- Generate the Plan --- #
    st.write("Generating allocation plan using AI...")
    with st.spinner("LLM is planning allocations..."):
        allocation_plan = automated_allocator.create_llm_allocation_plan(
            tasks=unallocated_tasks,
            users=available_users,
            simulated_now=st.session_state.simulated_now
        )
    
    if not allocation_plan:
        st.warning("The AI planner could not generate any allocations for the pending tasks with the current users.")
        return
    
    st.write(f"### Proposed Allocation Plan ({len(allocation_plan)} assignments)")
    
    # Store the plan in session state to handle approvals
    st.session_state.ai_allocation_plan = allocation_plan
    allocations_to_approve_ui = [] # For the approve all button

    # Display the generated plan
    for planned_alloc in allocation_plan:
        task_id = planned_alloc["task_id"]
        user_id = planned_alloc["user_id"]
        reason = planned_alloc["reason"]
        
        # Get full task and user objects for display
        task = st.session_state.tasks.get(task_id)
        user = st.session_state.users.get(user_id)

        if not task or not user:
            st.error(f"Error displaying plan: Could not find task {task_id} or user {user_id}")
            continue

        # Check if this task was already handled (e.g., by individual approve)
        current_task_status = st.session_state.tasks[task_id].status
        if current_task_status != TaskStatus.PENDING.value:
             continue # Skip displaying already approved/rejected tasks

        with st.expander(f"Assign Task: '{task.title}' TO User: '{user.name}'"):
            st.write(f"**Reason:** {reason}")
            st.divider()
            # Show Task Details
            st.markdown("**Task Details:**")
            st.write(f"  - ID: {task.task_id}")
            st.write(f"  - Priority: {task.priority}/5")
            st.write(f"  - Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M')}")
            st.write(f"  - Duration: {task.estimated_duration.total_seconds() / 3600:.1f} hours")
            st.write(f"  - Skills Req: {task.required_skills}")
            # Show User Details
            st.markdown("**User Details:**")
            st.write(f"  - ID: {user.user_id}")
            st.write(f"  - Workload: {user.current_workload:.1f}")
            st.write(f"  - Skills: {user.skills}")
            
            # Individual approve/reject buttons using a unique key based on task_id
            approve_key = f"approve_{task_id}"
            reject_key = f"reject_{task_id}"
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Approve Assignment", key=approve_key):
                    # Execute this single allocation
                    allocation_id = str(uuid.uuid4())
                    st.session_state.allocations[allocation_id] = {
                        "task_id": task_id,
                        "user_id": user_id,
                        "allocated_at": st.session_state.simulated_now,
                        "status": AllocationStatus.ACCEPTED.value,
                        "confidence_score": 1.0, # Mark as accepted from plan
                        "allocation_reason": reason
                    }
                    task.status = TaskStatus.ASSIGNED.value
                    st.session_state.tasks[task_id] = task
                    user.current_workload = min(1.0, user.current_workload + 0.2)
                    st.session_state.users[user_id] = user
                    st.success(f"Task '{task.title}' assigned to {user.name}.")
                    st.rerun()
            with col2:
                if st.button("Reject Assignment", key=reject_key):
                    # Mark task as rejected (maybe add a reason later?)
                    # For now, just leave it PENDING but remove from UI implicitly on rerun
                    # A more robust way would be to mark it somehow so planner ignores it next time
                    st.warning(f"Rejected assignment of '{task.title}' to {user.name}.")
                    # Remove from the plan in session state so 'Approve All' ignores it
                    st.session_state.ai_allocation_plan = [
                        p for p in st.session_state.ai_allocation_plan 
                        if p["task_id"] != task_id
                    ]
                    st.rerun()
        
        # Add to list for bulk approval if still pending
        if st.session_state.tasks[task_id].status == TaskStatus.PENDING.value:
            allocations_to_approve_ui.append(planned_alloc)

    # --- Approve All Button --- #
    if allocations_to_approve_ui:
        st.write("---")
        if st.button("Approve All Remaining Planned Allocations"):
            approved_count = 0
            # Use the filtered list from the UI loop
            for planned_alloc in allocations_to_approve_ui:
                task_id = planned_alloc["task_id"]
                user_id = planned_alloc["user_id"]
                reason = planned_alloc["reason"]
                
                # Double-check status before approving
                if st.session_state.tasks[task_id].status == TaskStatus.PENDING.value:
                    task = st.session_state.tasks[task_id]
                    user = st.session_state.users[user_id]
                    
                    allocation_id = str(uuid.uuid4())
                    st.session_state.allocations[allocation_id] = {
                         "task_id": task_id,
                         "user_id": user_id,
                         "allocated_at": st.session_state.simulated_now,
                         "status": AllocationStatus.ACCEPTED.value,
                         "confidence_score": 1.0, 
                         "allocation_reason": reason
                     }
                    task.status = TaskStatus.ASSIGNED.value
                    st.session_state.tasks[task_id] = task
                    user.current_workload = min(1.0, user.current_workload + 0.2)
                    st.session_state.users[user_id] = user
                    approved_count += 1
            
            if approved_count > 0:
                st.success(f"{approved_count} allocations approved successfully!")
            else:
                st.info("No remaining pending allocations from the plan were approved.")
            # Clear the plan after bulk approval
            st.session_state.ai_allocation_plan = [] 
            st.rerun()

def show_manual_allocation():
    """Show manual task allocation interface"""
    st.title("Manual Task Allocation")

    if 'tasks' not in st.session_state or not st.session_state.tasks:
        st.warning("No tasks available. Please add tasks first.")
        return
    if 'users' not in st.session_state or not st.session_state.users:
        st.warning("No users available. Please add users first.")
        return

    # Get unallocated tasks
    unallocated_tasks = [
        task for task in st.session_state.tasks.values()
        if task.status == TaskStatus.PENDING.value
    ]

    if not unallocated_tasks:
        st.info("All tasks are currently allocated.")
        return

    # Select Task
    task_options = {task.task_id: f"{task.title} (ID: {task.task_id})" for task in unallocated_tasks}
    selected_task_id = st.selectbox("Select Task to Allocate", options=list(task_options.keys()), format_func=lambda x: task_options[x])

    if selected_task_id:
        task = st.session_state.tasks[selected_task_id]
        st.write("### Task Details")
        st.write(f"**Title:** {task.title}")
        st.write(f"**Description:** {task.description}")
        st.write(f"**Priority:** {task.priority}/5")
        st.write(f"**Deadline:** {task.deadline.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Estimated Duration:** {task.estimated_duration.total_seconds() / 3600:.1f} hours")

        # Select User
        user_options = {user.user_id: f"{user.name} (Workload: {user.current_workload:.1f})" for user in st.session_state.users.values()}
        selected_user_id = st.selectbox("Select User to Assign", options=list(user_options.keys()), format_func=lambda x: user_options[x])

        if selected_user_id:
            user = st.session_state.users[selected_user_id]
            reason = st.text_input("Reason for manual allocation (optional):")

            # Check availability before allowing allocation button
            # Get user availability object
            user_availability = st.session_state.availability_manager.get_user_availability(user.user_id)
            
            # Check if deadline passed
            allocate_button_disabled = False
            availability_message = ""
            if task.deadline.astimezone(pytz.UTC) <= st.session_state.simulated_now:
                availability_message = f"Task deadline ({task.deadline.strftime('%Y-%m-%d')}) has passed relative to simulated time."
                allocate_button_disabled = True
            else:
                # Calculate max days to search ahead
                days_diff = (task.deadline.astimezone(pytz.UTC) - st.session_state.simulated_now).days
                max_days = max(1, min(days_diff + 1, 30))
                
                # Find next available slot
                next_slot = user_availability.find_next_available_slot(
                    start_time=st.session_state.simulated_now, # Use simulated time
                    duration=task.estimated_duration,
                    max_days_ahead=max_days # Use max_days_ahead
                )
                
                if next_slot is None:
                    availability_message = f"{user.name} does not appear to have a suitable free time slot starting from {st.session_state.simulated_now.strftime('%H:%M %Z')} until the deadline for the task's estimated duration ({task.estimated_duration}). Allocation might lead to conflicts."
                    allocate_button_disabled = True
                else:
                    availability_message = f"{user.name} appears available for this task (next slot starts {next_slot.strftime('%Y-%m-%d %H:%M')})."
            
            # Display availability message (use st.warning for unavailability)
            if allocate_button_disabled:
                st.warning(availability_message)
            else:
                st.success(availability_message)

            if st.button("Allocate Task Manually", disabled=allocate_button_disabled):
                # Create allocation record
                allocation_id = str(uuid.uuid4())
                st.session_state.allocations[allocation_id] = {
                    "task_id": task.task_id,
                    "user_id": user.user_id,
                    "allocated_at": datetime.now(pytz.UTC),
                    "status": AllocationStatus.ACCEPTED.value,
                    "confidence_score": 1.0,  # Manual allocation = 100% confidence
                    "allocation_reason": f"Manual allocation: {reason}"
                }

                # Update task status
                task.status = TaskStatus.ASSIGNED.value
                st.session_state.tasks[task.task_id] = task

                # Update user workload
                user.current_workload = min(1.0, user.current_workload + 0.2) # Adjust workload logic as needed
                st.session_state.users[user.user_id] = user

                st.success(f"Task '{task.title}' manually allocated to {user.name}.")
                st.rerun()

def show_allocation_board():
    """Display the task allocation board with controls to update task status."""
    st.title("Allocation Board")

    if 'tasks' not in st.session_state or not st.session_state.tasks:
        st.info("No tasks to display on the board.")
        return

    # Get tasks and allocations
    tasks = st.session_state.tasks
    allocations = st.session_state.allocations
    users = st.session_state.users
    
    # Helper function to find the active allocation for a task
    def find_active_allocation(task_id):
        for alloc_id, alloc in allocations.items():
            if alloc["task_id"] == task_id and alloc["status"] == AllocationStatus.ACCEPTED.value:
                return alloc_id, alloc
        return None, None

    # Create columns for task statuses
    cols = st.columns(len(TaskStatus))
    status_map = {status: i for i, status in enumerate(TaskStatus)}

    # Group tasks by status
    tasks_by_status = {status.value: [] for status in TaskStatus}
    for task_id, task in tasks.items():
        # Ensure task status is a string value from the enum
        status_value = task.status if isinstance(task.status, str) else task.status.value
        if status_value not in tasks_by_status:
             # Handle potential legacy statuses or inconsistencies
             st.warning(f"Task {task.task_id} has unknown status '{status_value}'. Placing in PENDING.")
             status_value = TaskStatus.PENDING.value
        tasks_by_status[status_value].append(task)


    # Display tasks in columns
    for status, col_index in status_map.items():
        with cols[col_index]:
            st.subheader(status.value.replace("_", " "))
            st.write("---")
            # Sort tasks within columns, maybe by priority or deadline? Optional.
            # sorted_tasks = sorted(tasks_by_status[status.value], key=lambda t: t.priority, reverse=True)
            sorted_tasks = tasks_by_status[status.value] 
            
            for task in sorted_tasks:
                with st.container(border=True):
                    st.write(f"**{task.title}**")
                    st.caption(f"ID: {task.task_id}")
                    st.write(f"Prio: {task.priority} | Due: {task.deadline.strftime('%Y-%m-%d')}")
                    
                    # Find allocation details if assigned or in progress
                    assigned_user_name = "Unassigned"
                    user_id = None
                    alloc_id = None
                    if task.status == TaskStatus.ASSIGNED.value or task.status == TaskStatus.IN_PROGRESS.value:
                        alloc_id, allocation = find_active_allocation(task.task_id)
                        if allocation and allocation["user_id"] in users:
                            user_id = allocation["user_id"]
                            assigned_user_name = users[user_id].name
                        elif allocation:
                             assigned_user_name = f"User ID: {allocation['user_id']}"
                             user_id = allocation['user_id'] # Still store ID even if user deleted

                    st.write(f"👤 {assigned_user_name}")

                    # --- Add Action Buttons based on status ---
                    if task.status == TaskStatus.ASSIGNED.value:
                        button_cols = st.columns(2)
                        with button_cols[0]:
                            if st.button("▶️ Start Progress", key=f"start_{task.task_id}", use_container_width=True):
                                task.status = TaskStatus.IN_PROGRESS.value
                                st.session_state.tasks[task.task_id] = task
                                st.success(f"Task '{task.title}' moved to In Progress.")
                                st.rerun()
                        with button_cols[1]:
                             if st.button("❌ Cancel", key=f"cancel_assigned_{task.task_id}", use_container_width=True):
                                task.status = TaskStatus.CANCELLED.value
                                st.session_state.tasks[task.task_id] = task
                                if alloc_id and user_id in st.session_state.users:
                                    st.session_state.allocations[alloc_id]["status"] = AllocationStatus.REJECTED.value # Or maybe CANCELLED if added
                                    user = st.session_state.users[user_id]
                                    user.current_workload = max(0.0, user.current_workload - 0.2) # Decrease workload
                                    st.session_state.users[user_id] = user
                                st.warning(f"Task '{task.title}' cancelled.")
                                st.rerun()

                    elif task.status == TaskStatus.IN_PROGRESS.value:
                        button_cols = st.columns(2)
                        with button_cols[0]:
                            if st.button("✅ Mark Complete", key=f"complete_{task.task_id}", use_container_width=True):
                                task.status = TaskStatus.COMPLETED.value
                                st.session_state.tasks[task.task_id] = task
                                if alloc_id and user_id in st.session_state.users:
                                    st.session_state.allocations[alloc_id]["status"] = AllocationStatus.COMPLETED.value 
                                    st.session_state.allocations[alloc_id]["completed_at"] = st.session_state.simulated_now # Record completion time
                                    user = st.session_state.users[user_id]
                                    user.current_workload = max(0.0, user.current_workload - 0.2) # Decrease workload
                                    st.session_state.users[user_id] = user
                                st.success(f"Task '{task.title}' marked as Completed.")
                                st.rerun()
                        with button_cols[1]:
                             if st.button("❌ Cancel", key=f"cancel_inprogress_{task.task_id}", use_container_width=True):
                                task.status = TaskStatus.CANCELLED.value
                                st.session_state.tasks[task.task_id] = task
                                if alloc_id and user_id in st.session_state.users:
                                    st.session_state.allocations[alloc_id]["status"] = AllocationStatus.REJECTED.value # Or CANCELLED
                                    user = st.session_state.users[user_id]
                                    user.current_workload = max(0.0, user.current_workload - 0.2) # Decrease workload
                                    st.session_state.users[user_id] = user
                                st.warning(f"Task '{task.title}' cancelled.")
                                st.rerun()
                    # --- End Action Buttons ---

def show_agent_chat():
    """Display the AI agent chat interface"""
    st.title("💬 Agent Chat")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask the agent about tasks, users, or allocations..."):
        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.spinner("Agent is thinking..."):
            try:
                # Call the actual agent method, passing the message and the history
                agent_response = st.session_state.ai_agent.process_chat_message(
                    user_message=prompt, 
                    chat_history=st.session_state.chat_history, # Pass history
                    session_state=st.session_state # Pass state for actions
                )
                
                # Add agent response to history and display
                st.session_state.chat_history.append({"role": "assistant", "content": agent_response})
                with st.chat_message("assistant"):
                    st.markdown(agent_response)
            
            except Exception as e:
                error_message = f"Sorry, an error occurred: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                with st.chat_message("assistant"):
                    st.error(error_message)

# Main application
def main():
    """Main application function"""
    # Custom CSS for better styling
    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        .sidebar-text {
            padding: 1rem;
            color: #fff;
            font-size: 1.2rem;
            font-weight: 600;
        }
        div[data-testid="stSidebarNav"] {
            background-color: #1E1E1E;
            padding: 1rem;
            border-radius: 10px;
        }
        div[data-testid="stSidebarNav"] > ul {
            padding-left: 0;
        }
        div[data-testid="stSidebarNav"] span {
            color: #fff;
            font-size: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar navigation with icons
    st.sidebar.markdown('<p class="sidebar-text">🎯 Task Allocation Agent</p>', unsafe_allow_html=True)
    
    # --- Time Simulation Controls --- #
    st.sidebar.divider()
    st.sidebar.subheader("⏳ Time Simulation")
    sim_now_str = st.session_state.simulated_now.strftime("%Y-%m-%d %H:%M %Z")
    st.sidebar.write(f"Current Sim Time: **{sim_now_str}**")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Advance 1 Hr"):
            st.session_state.simulated_now += timedelta(hours=1)
            st.rerun()
        if st.button("Advance 1 Day"):
            st.session_state.simulated_now += timedelta(days=1)
            st.rerun()
    with col2:
        if st.button("Advance 8 Hrs"):
            st.session_state.simulated_now += timedelta(hours=8)
            st.rerun()
        if st.button("Advance 1 Week"):
            st.session_state.simulated_now += timedelta(weeks=1)
            st.rerun()
    if st.sidebar.button("Reset Time"):
        st.session_state.simulated_now = datetime.now(pytz.UTC)
        st.rerun()
    st.sidebar.divider()
    # --- End Time Simulation --- #

    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        [
            "💬 Agent Chat",
            "👤 User Management",
            "📋 Task Management", 
            "📋 Allocation Board",
            "🤖 AI Allocation",
            "✍️ Manual Allocation",
            "📅 Smart Availability",
            # "💬 Communication",
            "📚 Learning",
            # "📊 Progress Tracking"
        ]
    )

    # Strip icons from page names for routing
    page_name = page.split(" ", 1)[1]
    
    if page_name == "Agent Chat":
        show_agent_chat()
    elif page_name == "User Management":
        show_user_management()
    elif page_name == "Task Management":
        show_task_management()
    elif page_name == "Allocation Board":
        show_allocation_board()
    elif page_name == "AI Allocation":
        show_ai_allocation()
    elif page_name == "Manual Allocation":
        show_manual_allocation()
    elif page_name == "Smart Availability":
        show_smart_availability()
    elif page_name == "Communication":
        show_communication()
    elif page_name == "Learning":
        show_learning()
    elif page_name == "Progress Tracking":
        show_progress_tracking()

# Run the app
if __name__ == "__main__":
    main()