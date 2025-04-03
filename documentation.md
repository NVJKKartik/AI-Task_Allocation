# AI Task Allocation Agent Documentation

## Overview

The AI Task Allocation Agent is an intelligent system that efficiently assigns tasks to individuals based on their expertise, availability, and preferences. Built using the LangChain AI framework, this agent leverages machine learning to optimize task allocation, ensuring that tasks are assigned to the most qualified individuals while considering their availability and personal preferences.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Core Components](#core-components)
3. [Installation](#installation)
4. [Usage Guide](#usage-guide)
5. [API Reference](#api-reference)
6. [Machine Learning Model](#machine-learning-model)
7. [Future Enhancements](#future-enhancements)

## System Architecture

The AI Task Allocation Agent follows a modular architecture with the following key components:

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│                    AI Task Allocation Agent                 │
│                                                             │
├─────────────┬─────────────────┬────────────────┬────────────┤
│             │                 │                │            │
│  Task       │  Availability   │ Personalization│   User     │
│  Matching   │  Management     │    Engine      │ Interface  │
│  Algorithm  │    System       │                │ (Streamlit)│
│             │                 │                │            │
└─────────────┴─────────────────┴────────────────┴────────────┘
        │              │                │              │
        ▼              ▼                ▼              ▼
┌─────────────┬─────────────────┬────────────────┬────────────┐
│             │                 │                │            │
│  LangChain  │   Calendar &    │  ML Models &   │ Web-based  │
│  Framework  │  Scheduling     │ User History   │ Dashboard  │
│             │                 │                │            │
└─────────────┴─────────────────┴────────────────┴────────────┘
```

### Data Flow

1. User profiles and tasks are input into the system
2. The availability management system tracks user schedules
3. The task matching algorithm matches tasks to users based on skills and requirements
4. The personalization engine refines matches based on historical data and preferences
5. Results are presented through the user interface
6. Task outcomes are recorded to improve future allocations

## Core Components

### Task Matching Algorithm

The task matching algorithm is the core of the system, responsible for matching tasks to the most suitable users based on their skills, expertise, and other factors.

**Key Features:**
- Skill matching based on required vs. available skills
- Priority-based allocation for urgent tasks
- Workload balancing to prevent overloading team members
- Deadline-aware allocation to ensure timely completion
- Confidence scoring for match quality assessment

**Implementation:**
- Uses LangChain with OpenAI's GPT models for intelligent matching
- Provides detailed explanations for allocation decisions
- Suggests alternative matches with confidence scores

### Availability Management System

The availability management system tracks and manages user availability, ensuring that tasks are only assigned to users who have the time to complete them.

**Key Features:**
- Time slot management with regular and recurring schedules
- Availability checking for specific time periods
- Finding common availability across multiple users
- Calendar view generation with availability percentages
- Persistence through JSON file storage

**Implementation:**
- TimeSlot class for representing single time slots
- RecurringTimeSlot class for recurring schedules using iCalendar RRULE format
- UserAvailability class for managing individual user availability
- AvailabilityManager class for coordinating availability across users

### Personalization Engine

The personalization engine learns from past allocations and adapts task allocation based on user preferences and performance history.

**Key Features:**
- Allocation history tracking for learning from past assignments
- Machine learning model for predicting user satisfaction
- Personalized scoring for task-user matching
- User insights generation for understanding preferences
- Adaptive allocation based on historical performance

**Implementation:**
- AllocationHistory class for tracking task allocations and outcomes
- UserPreferenceModel class using RandomForest regression for preference learning
- PersonalizedTaskMatcher class for intelligent, personalized matching

### User Interface

The user interface provides an intuitive way to interact with the AI task allocation agent, allowing users to manage profiles, tasks, and view allocations.

**Key Features:**
- Dashboard with summary statistics
- User management with skill profiles
- Task creation and management
- Availability scheduling and visualization
- Task allocation with detailed explanations
- Personalized recommendations

**Implementation:**
- Built with Streamlit for rapid development and ease of use
- Interactive visualizations using Plotly
- Responsive design for desktop and mobile use

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ai-task-allocator.git
   cd ai-task-allocator
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with the following content:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Run the application:
   ```bash
   streamlit run code/app.py
   ```

## Usage Guide

### Getting Started

1. **Load Sample Data**: When you first launch the application, click the "Load Sample Data" button in the sidebar to populate the system with sample users and tasks.

2. **Navigate the Interface**: Use the sidebar to navigate between different sections of the application:
   - Dashboard: Overview of users, tasks, and recent allocations
   - Users: Manage user profiles
   - Tasks: Create and manage tasks
   - Availability: Manage user availability
   - Task Allocation: Match tasks to users

### Managing Users

1. **View Existing Users**: Expand user cards to view details about their skills, preferences, and performance metrics.

2. **Add New Users**: Fill out the form at the bottom of the Users page to add new users with their skills, preferences, and initial performance metrics.

### Managing Tasks

1. **View Existing Tasks**: Expand task cards to view details about required skills, priority, deadline, and allocation status.

2. **Add New Tasks**: Fill out the form at the bottom of the Tasks page to add new tasks with required skills, priority, and deadline.

### Managing Availability

1. **Select User**: Choose a user from the dropdown to view and manage their availability.

2. **View Calendar**: The calendar heatmap shows availability percentages for each day of the selected month.

3. **Add Availability**: Use the "Add Availability" section to add single time slots or recurring schedules.

4. **Find Common Availability**: Select multiple users to find times when all selected users are available.

### Allocating Tasks

1. **Select Task**: Choose an unallocated task from the dropdown.

2. **Match Task**: Click the "Match Task to Users" button to have the AI agent find the best match.

3. **Review Allocation**: Review the allocation details, including the confidence score, reason for the match, and alternative matches.

4. **View Skill Comparison**: The radar chart shows how the user's skills compare to the task requirements.

## API Reference

### Task Matching API

```python
from task_matching_algorithm import Task, UserProfile, TaskMatchingEngine

# Create a task
task = Task(
    task_id="task-123",
    title="Implement login feature",
    description="Create login functionality with JWT authentication",
    required_skills={"Python": 3, "JavaScript": 3},
    priority=4,
    deadline=datetime.fromisoformat("2025-04-10T17:00:00"),
    estimated_duration=timedelta(hours=4)
)

# Create a user profile
user = UserProfile(
    user_id="user-456",
    name="John Doe",
    skills={"Python": 5, "JavaScript": 3, "React": 2},
    availability=[
        {"start": "2025-04-03T09:00:00", "end": "2025-04-03T17:00:00"},
        {"start": "2025-04-04T09:00:00", "end": "2025-04-04T17:00:00"}
    ],
    preferences={"preferred_task_types": ["backend", "data-processing"]},
    performance_metrics={"task_completion_rate": 0.95, "on_time_rate": 0.9},
    current_workload=0.5
)

# Match task to users
engine = TaskMatchingEngine()
allocation = engine.match_task_to_users(task, [user])
```

### Availability Management API

```python
from availability_management import TimeSlot, RecurringTimeSlot, AvailabilityManager
from datetime import datetime, timedelta
import pytz

# Create an availability manager
manager = AvailabilityManager()

# Add a user
manager.add_user("user-123")

# Add a time slot
time_slot = TimeSlot(
    start_time=datetime(2025, 4, 3, 9, 0, tzinfo=pytz.UTC),
    end_time=datetime(2025, 4, 3, 17, 0, tzinfo=pytz.UTC),
    availability_type="AVAILABLE"
)
manager.add_time_slot("user-123", time_slot)

# Add a recurring time slot
recurring_slot = RecurringTimeSlot(
    start_time=datetime(2025, 4, 7, 9, 0, tzinfo=pytz.UTC),
    end_time=datetime(2025, 4, 7, 17, 0, tzinfo=pytz.UTC),
    recurrence_rule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    availability_type="AVAILABLE"
)
manager.add_recurring_slot("user-123", recurring_slot)

# Check availability
is_available = manager.is_user_available(
    "user-123",
    datetime(2025, 4, 3, 10, 0, tzinfo=pytz.UTC),
    timedelta(hours=2)
)
```

### Personalization API

```python
from personalization import PersonalizedTaskMatcher
from task_matching_algorithm import Task, UserProfile

# Create a personalized task matcher
matcher = PersonalizedTaskMatcher()

# Match task to users with personalization
result = matcher.match_task_to_users(task, users)

# Record allocation outcome
matcher.record_allocation_outcome(
    allocation_id="alloc-123",
    completed=True,
    completion_time=datetime.now(),
    completion_quality=4.5,
    on_time=True,
    user_satisfaction=4.0,
    feedback="Great work on this task!"
)

# Get personalized recommendations for a user
recommendations = matcher.get_personalized_recommendations(user, tasks)

# Get user insights
insights = matcher.get_user_insights("user-123")
```

## Machine Learning Model

The AI Task Allocation Agent uses machine learning to personalize task allocation based on historical data and user preferences.

### Model Architecture

- **Algorithm**: Random Forest Regressor
- **Features**:
  - Task properties (priority, estimated duration)
  - User properties (skills, workload)
  - Skill match metrics
  - Preference match indicators
- **Target Variable**: User satisfaction rating (1-5 scale)

### Training Process

1. Historical allocation data is collected, including task details, user details, and satisfaction ratings
2. Data is preprocessed with feature engineering to capture skill matches and preference matches
3. Categorical features are one-hot encoded, numerical features are standardized
4. Model is trained using 80% of the data, with 20% held out for validation
5. Model performance is evaluated using R² score

### Prediction Process

1. For a potential task-user match, features are extracted similar to the training process
2. The trained model predicts the expected user satisfaction
3. This prediction is incorporated into the overall personalized matching score

### Model Performance

The model's performance improves as more allocation data becomes available. With sufficient data, the model typically achieves:
- Training R² scores of 0.70-0.85
- Validation R² scores of 0.60-0.80

## Future Enhancements

The AI Task Allocation Agent has several potential areas for future enhancement:

1. **Team Optimization**: Extend the system to optimize task allocation across entire teams, not just individuals

2. **Integration Capabilities**: Add integrations with popular project management tools like Jira, Asana, or Trello

3. **Advanced ML Models**: Implement more sophisticated machine learning models, such as deep learning for sequence prediction

4. **Real-time Updates**: Add real-time notifications and updates for task allocations and availability changes

5. **Mobile Application**: Develop a companion mobile app for on-the-go availability updates and task acceptance

6. **Natural Language Interface**: Add a conversational interface for task creation and allocation

7. **Performance Analytics**: Enhance the analytics capabilities with more detailed performance metrics and visualizations
