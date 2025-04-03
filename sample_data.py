# flake8: noqa
"""
Generates sample data for the AI Task Allocation Agent.
"""

from datetime import datetime, timedelta
import random
import pytz
import uuid

# Assuming these classes are importable from their respective modules
# Adjust paths if necessary based on your project structure
from task_matching_algorithm import Task as TaskModel, UserProfile
from availability_management import AvailabilityManager, TimeSlot, RecurringTimeSlot
from typing import Tuple, Dict, Any, List

# --- Configuration ---
NUM_USERS = 10
NUM_TASKS = 30

ALL_SKILLS = [
    "Python", "JavaScript", "React", "Node.js", "TypeScript", "HTML/CSS",
    "Data Analysis", "SQL", "NoSQL", "Machine Learning", "Deep Learning",
    "UI Design", "UX Research", "Graphic Design", "Prototyping",
    "Cloud (AWS)", "Cloud (Azure)", "Cloud (GCP)", "Docker", "Kubernetes",
    "Terraform", "CI/CD", "Project Management", "Agile Methodologies",
    "Communication", "Problem Solving", "Team Leadership", "API Design",
    "System Architecture", "Security Auditing"
]

TASK_CATEGORIES = [
    "Backend Development", "Frontend Development", "Data Science", "UI/UX Design",
    "DevOps/Infrastructure", "Project Management", "Research", "Documentation",
    "Testing/QA"
]

FIRST_NAMES = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Jamie", "Skyler", "Quinn", "Drew"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]

# --- Helper Functions ---

def generate_random_skills(num_skills_range=(3, 8)) -> Dict[str, int]:
    """Generates a random set of skills with proficiency levels."""
    num_skills = random.randint(*num_skills_range)
    skills = random.sample(ALL_SKILLS, num_skills)
    return {skill: random.randint(1, 5) for skill in skills}

def generate_random_preferences() -> Dict[str, Any]:
    """Generates random user preferences."""
    prefs = {}
    if random.random() > 0.3:
        prefs["preferred_task_types"] = random.sample(TASK_CATEGORIES, random.randint(1, 3))
    if random.random() > 0.5:
        prefs["preferred_skills"] = random.sample(ALL_SKILLS, random.randint(1, 4))
    if random.random() > 0.6:
        prefs["task_complexity_preference"] = random.randint(1, 5)
    return prefs

# --- Data Generation Functions ---

def generate_sample_users(num_users: int = NUM_USERS) -> Dict[str, UserProfile]:
    """Generates a dictionary of sample UserProfile objects."""
    users = {}
    used_names = set()
    for i in range(num_users):
        user_id = f"user-{uuid.uuid4()}"
        while True:
            name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
            if name not in used_names:
                used_names.add(name)
                break

        users[user_id] = UserProfile(
            user_id=user_id,
            name=name,
            skills=generate_random_skills(),
            availability=[],  # Availability added separately
            preferences=generate_random_preferences(),
            performance_metrics={
                "task_completion_rate": round(random.uniform(0.75, 0.99), 2),
                "on_time_rate": round(random.uniform(0.70, 0.98), 2)
            },
            current_workload=0.0
        )
    return users

def generate_sample_tasks(num_tasks: int = NUM_TASKS, available_skills: List[str] = ALL_SKILLS) -> Dict[str, TaskModel]:
    """Generates a dictionary of sample TaskModel objects."""
    tasks = {}
    today = datetime.now(pytz.UTC)
    for i in range(num_tasks):
        task_id = f"task-{uuid.uuid4()}"
        category = random.choice(TASK_CATEGORIES)
        required_skills = {}
        if available_skills:
            num_req_skills = random.randint(1, 4)
            req_skills_names = random.sample(available_skills, min(num_req_skills, len(available_skills)))
            required_skills = {skill: random.randint(2, 4) for skill in req_skills_names}

        tasks[task_id] = TaskModel(
            task_id=task_id,
            title=f"{category} Task #{i+1} - {random.choice(['Implement', 'Design', 'Analyze', 'Refactor', 'Test', 'Document'])}",
            description=f"Description for {category} task {i+1}. Requires focus on {', '.join(required_skills.keys()) if required_skills else 'general skills'}.",
            required_skills=required_skills,
            priority=random.randint(1, 5),
            deadline=today + timedelta(days=random.randint(3, 30), hours=random.randint(1, 23)),
            estimated_duration=timedelta(hours=random.randint(1, 20)),
            status="PENDING" # Start all tasks as pending
        )
    return tasks

def setup_sample_availability(users: Dict[str, UserProfile], availability_manager: AvailabilityManager):
    """Adds sample availability slots to the manager for the given users."""
    today = datetime.now(pytz.UTC).date()

    for i, user_id in enumerate(users.keys()):
        user_avail = availability_manager.get_user_availability(user_id)

        # Add recurring weekly availability (e.g., Mon-Fri 9-5)
        start_hour = random.randint(8, 10)
        end_hour = random.randint(16, 18)
        recurring_start = datetime(today.year, today.month, today.day, start_hour, 0, tzinfo=pytz.UTC)
        recurring_end = datetime(today.year, today.month, today.day, end_hour, 0, tzinfo=pytz.UTC)
        user_avail.add_recurring_slot(RecurringTimeSlot(
            start_time=recurring_start,
            end_time=recurring_end,
            recurrence_rule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
            availability_type="AVAILABLE"
        ))

        # Add some specific busy slots
        for _ in range(random.randint(0, 3)): # 0 to 3 busy slots in the next week
            busy_day_offset = random.randint(1, 7)
            busy_start_hour = random.randint(9, 15)
            busy_duration_hours = random.uniform(0.5, 2.0)
            busy_start = datetime.combine(today + timedelta(days=busy_day_offset), datetime.min.time(), tzinfo=pytz.UTC)
            busy_start = busy_start.replace(hour=busy_start_hour)
            busy_end = busy_start + timedelta(hours=busy_duration_hours)
            user_avail.add_time_slot(TimeSlot(
                start_time=busy_start,
                end_time=busy_end,
                availability_type="BUSY"
            ))

        # Maybe add an OOO period for one user
        if i == len(users) // 2: # Pick the middle user for OOO
            ooo_start_offset = random.randint(8, 14)
            ooo_days = random.randint(2, 5)
            ooo_start = datetime.combine(today + timedelta(days=ooo_start_offset), datetime.min.time(), tzinfo=pytz.UTC)
            ooo_end = datetime.combine(today + timedelta(days=ooo_start_offset + ooo_days), datetime.min.time(), tzinfo=pytz.UTC)
            user_avail.add_time_slot(TimeSlot(
                start_time=ooo_start,
                end_time=ooo_end,
                availability_type="OUT_OF_OFFICE"
            ))

def get_sample_data() -> Tuple[Dict[str, UserProfile], Dict[str, TaskModel], AvailabilityManager]:
    """Generates and returns all sample data components."""
    print("Generating sample users...")
    sample_users = generate_sample_users()
    
    # Extract all skills present in the generated users for more relevant tasks
    user_skills = set()
    for user in sample_users.values():
        user_skills.update(user.skills.keys())
    
    print("Generating sample tasks...")
    sample_tasks = generate_sample_tasks(available_skills=list(user_skills))
    
    print("Setting up sample availability...")
    availability_manager = AvailabilityManager()
    setup_sample_availability(sample_users, availability_manager)
    
    print("Sample data generation complete.")
    return sample_users, sample_tasks, availability_manager

if __name__ == "__main__":
    # Example of how to use this module
    users_data, tasks_data, avail_manager = get_sample_data()
    print(f"\nGenerated {len(users_data)} users and {len(tasks_data)} tasks.")
    
    # Print first user and task for verification
    first_user_id = list(users_data.keys())[0]
    print("\nFirst User:")
    print(users_data[first_user_id].__dict__)
    
    first_task_id = list(tasks_data.keys())[0]
    print("\nFirst Task:")
    print(tasks_data[first_task_id].__dict__)
    
    print("\nAvailability for first user:")
    user_avail = avail_manager.get_user_availability(first_user_id)
    print(f"  Time Slots: {len(user_avail.time_slots)}")
    print(f"  Recurring Slots: {len(user_avail.recurring_slots)}") 