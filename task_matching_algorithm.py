"""
Task Matching Algorithm for AI Task Allocation Agent

This module implements the core task matching algorithm using LangChain.
It matches tasks to users based on skills, expertise, availability, and preferences.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

class UserProfile:
    """User profile with skills, availability, and preferences"""
    def __init__(self, user_id: str, name: str, skills: Dict[str, int], 
                 availability: List[Dict], preferences: Dict[str, Any],
                 performance_metrics: Dict[str, float], current_workload: float):
        self.user_id = user_id
        self.name = name
        self.skills = skills  # Dict of skill_name: proficiency_level
        self.availability = availability  # List of time slots
        self.preferences = preferences
        self.performance_metrics = performance_metrics
        self.current_workload = current_workload
    
    def to_dict(self) -> Dict:
        """Convert user profile to dictionary for LLM prompt"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "skills": self.skills,
            "availability": self.availability,
            "preferences": self.preferences,
            "performance_metrics": self.performance_metrics,
            "current_workload": self.current_workload
        }

class Task:
    """Represents a task that needs to be allocated"""
    def __init__(
        self,
        task_id: str,
        title: str,
        description: str,
        required_skills: Dict[str, int],
        priority: int,
        deadline: datetime,
        estimated_duration: timedelta,
        status: str = "PENDING"
    ):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.required_skills = required_skills
        self.priority = priority
        self.deadline = deadline
        self.estimated_duration = estimated_duration
        self.status = status  # PENDING, ASSIGNED, COMPLETED, CANCELLED
    
    def to_dict(self) -> Dict:
        """Convert task to dictionary for LLM prompt"""
        return {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "required_skills": self.required_skills,
            "priority": self.priority,
            "deadline": self.deadline.isoformat(),
            "estimated_duration_minutes": self.estimated_duration.total_seconds() / 60,
            "status": self.status
        }

class TaskAllocation:
    """Result of task allocation with user, confidence score, and reasoning"""
    def __init__(self, task_id: str, user_id: str, confidence_score: float, 
                 allocation_reason: str, alternative_users: List[Dict] = None):
        self.task_id = task_id
        self.user_id = user_id
        self.confidence_score = confidence_score
        self.allocation_reason = allocation_reason
        self.alternative_users = alternative_users or []
        self.allocated_at = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert allocation to dictionary"""
        return {
            "task_id": self.task_id,
            "user_id": self.user_id,
            "confidence_score": self.confidence_score,
            "allocation_reason": self.allocation_reason,
            "alternative_users": self.alternative_users,
            "allocated_at": self.allocated_at.isoformat()
        }

class TaskMatchingEngine:
    """Core task matching engine using LangChain and LLMs"""
    
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """Initialize the task matching engine with specified LLM"""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        print(f"Using OpenAI API key: {api_key[:5]}...{api_key[-4:]}")
        
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            openai_api_key=api_key
        )
        self.setup_output_parser()
        self.setup_prompt_template()
        self.chain = self.prompt | self.llm | self.output_parser
    
    def setup_output_parser(self):
        """Set up the JSON output parser for LLM responses"""
        self.output_parser = JsonOutputParser()
    
    def setup_prompt_template(self):
        """Set up the prompt template for task allocation"""
        template = """
        You are an AI Task Allocation Agent responsible for intelligently matching tasks with the most suitable individuals.
        Analyze the task requirements and user profiles carefully.

        # Task Information:
        {task}
        
        # Available Users:
        {users}
        
        # Current Date and Time:
        {current_datetime}
        
        Your goal is to identify the BEST user for this task based on a holistic evaluation:
        1.  **Skill Relevance (Crucial):** Don't just look for exact keyword matches. Understand the task's *context* and determine if the user's skills are *relevant* even if not explicitly listed. For example, a task requiring 'Python' for 'Data Analysis' is well-suited for someone strong in 'Data Analysis' and proficient in 'Python', even if 'Python' isn't their top skill overall. Similarly, skills like 'Machine Learning' or 'Computer Vision' are highly relevant to a 'face swapping app' task, even if only 'Python' is listed.
        2.  **Skill Proficiency:** Ensure the user meets or exceeds the *minimum required proficiency* for the *relevant* skills.
        3.  **Availability:** Check if the user has sufficient time before the deadline, considering the task's estimated duration.
        4.  **Workload:** Prefer users with lower current workloads, but don't disqualify capable users solely based on a moderate workload if they are a significantly better fit otherwise.
        5.  **Performance & Preferences:** Use historical performance and user preferences as secondary factors to refine choices between otherwise equally qualified candidates.

        Output your decision in the following JSON format ONLY. Provide a confidence score (0.0-1.0) reflecting your certainty in the match, and a clear reasoning justifying your choice based on the criteria above, especially skill relevance.
        {{ 
            "best_match_user_id": "string_or_null",
            "confidence_score": float, 
            "allocation_reason": "string",
            "alternative_users": [
                {{
                    "user_id": "string",
                    "confidence_score": float,
                    "reason": "string"
                }}
            ] 
        }}
        """
        
        self.prompt = ChatPromptTemplate.from_template(template)
    
    def match_task_to_users(self, task: Task, users: List[UserProfile]) -> TaskAllocation:
        """Match a task to the best available user"""
        # Prepare inputs for the LLM
        task_dict = task.to_dict()
        users_dict = [user.to_dict() for user in users]
        current_datetime = datetime.now().isoformat()
        
        # Call the chain
        result = self.chain.invoke({
            "task": json.dumps(task_dict, default=str),
            "users": json.dumps(users_dict, default=str),
            "current_datetime": current_datetime
        })
        
        # Create and return the task allocation
        return TaskAllocation(
            task_id=task.task_id,
            user_id=result["best_match_user_id"],
            confidence_score=float(result["confidence_score"]),
            allocation_reason=result["allocation_reason"],
            alternative_users=result["alternative_users"]
        )

    def explain_allocation(self, allocation: TaskAllocation, task: Task, user: UserProfile) -> str:
        """Generate a human-readable explanation of the task allocation"""
        explanation_prompt = ChatPromptTemplate.from_template("""
            You've matched the task "{task_title}" to {user_name} with a confidence score of {confidence_score}.
            
            Task details:
            {task_description}
            Required skills: {required_skills}
            Priority: {priority}
            Deadline: {deadline}
            
            User details:
            Skills: {user_skills}
            Current workload: {current_workload}
            
            Original allocation reason:
            {allocation_reason}
            
            Please provide a clear, concise explanation of this match in a way that would be helpful for both the task creator and the assigned user to understand.
            """)
        
        explanation_chain = explanation_prompt | self.llm
        
        result = explanation_chain.invoke({
            "task_title": task.title,
            "user_name": user.name,
            "confidence_score": allocation.confidence_score,
            "task_description": task.description,
            "required_skills": json.dumps(task.required_skills),
            "priority": task.priority,
            "deadline": task.deadline.isoformat(),
            "user_skills": json.dumps(user.skills),
            "current_workload": user.current_workload,
            "allocation_reason": allocation.allocation_reason
        })
        
        return result.content

# Example usage
if __name__ == "__main__":
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
    
    # Create a sample task
    task = Task(
        task_id="task-789",
        title="Implement login feature",
        description="Create login functionality with JWT authentication",
        required_skills={"Python": 3, "JavaScript": 3},
        priority=4,
        deadline=datetime.fromisoformat("2025-04-10T17:00:00"),
        estimated_duration=timedelta(hours=4)
    )
    
    try:
        # Initialize the task matching engine
        print("Initializing task matching engine...")
        engine = TaskMatchingEngine()
        
        # Match the task to users
        print("Matching task to users...")
        allocation = engine.match_task_to_users(task, users)
        
        # Find the matched user
        matched_user = next((user for user in users if user.user_id == allocation.user_id), None)
        
        # Generate explanation
        if matched_user:
            print(f"Task '{task.title}' matched to {matched_user.name}")
            print(f"Confidence score: {allocation.confidence_score}")
            print(f"Allocation reason: {allocation.allocation_reason}")
            
            explanation = engine.explain_allocation(allocation, task, matched_user)
            print(f"\nDetailed explanation: {explanation}")
            
            print("\nAlternative matches:")
            for alt in allocation.alternative_users:
                alt_user = next((user for user in users if user.user_id == alt["user_id"]), None)
                if alt_user:
                    print(f"- {alt_user.name} (confidence: {alt['confidence_score']}): {alt['reason']}")
        else:
            print("No suitable user found for the task.")
    except Exception as e:
        print(f"Error during task matching: {str(e)}")
