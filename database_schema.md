# Task Allocation Agent - Database Schema Design

## Overview

This document outlines the database schema design for the AI Task Allocation Agent. The schema is designed to support the core functionality of task matching, availability management, and personalization.

## Tables

### 1. Users

Stores information about users who can be assigned tasks.

```sql
CREATE TABLE users (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
```

### 2. Skills

Master list of all possible skills in the system.

```sql
CREATE TABLE skills (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    category VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### 3. User_Skills

Maps users to their skills with proficiency levels.

```sql
CREATE TABLE user_skills (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    skill_id VARCHAR(36) NOT NULL,
    proficiency_level INT NOT NULL, -- 1-5 scale
    years_experience FLOAT,
    last_used_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE,
    UNIQUE (user_id, skill_id)
);
```

### 4. User_Preferences

Stores user preferences for task allocation.

```sql
CREATE TABLE user_preferences (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    preference_key VARCHAR(50) NOT NULL,
    preference_value TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, preference_key)
);
```

### 5. Availability

Tracks user availability for task assignment.

```sql
CREATE TABLE availability (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP NOT NULL,
    availability_type ENUM('AVAILABLE', 'BUSY', 'OUT_OF_OFFICE') NOT NULL,
    recurrence_rule TEXT, -- iCalendar RRULE format for recurring availability
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### 6. Tasks

Stores information about tasks to be allocated.

```sql
CREATE TABLE tasks (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    priority INT NOT NULL, -- 1-5 scale
    deadline TIMESTAMP,
    estimated_duration INT, -- in minutes
    status ENUM('PENDING', 'ASSIGNED', 'IN_PROGRESS', 'COMPLETED', 'CANCELLED') NOT NULL DEFAULT 'PENDING',
    created_by VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL
);
```

### 7. Task_Skills

Maps tasks to required skills with minimum proficiency levels.

```sql
CREATE TABLE task_skills (
    id VARCHAR(36) PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    skill_id VARCHAR(36) NOT NULL,
    minimum_proficiency INT NOT NULL, -- 1-5 scale
    importance INT NOT NULL, -- 1-5 scale
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE,
    UNIQUE (task_id, skill_id)
);
```

### 8. Task_Dependencies

Tracks dependencies between tasks.

```sql
CREATE TABLE task_dependencies (
    id VARCHAR(36) PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    dependency_task_id VARCHAR(36) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (dependency_task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    UNIQUE (task_id, dependency_task_id)
);
```

### 9. Task_Allocations

Records task assignments to users.

```sql
CREATE TABLE task_allocations (
    id VARCHAR(36) PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    confidence_score FLOAT NOT NULL, -- AI confidence in match (0-1)
    allocation_reason TEXT,
    status ENUM('PENDING', 'ACCEPTED', 'REJECTED', 'COMPLETED') NOT NULL DEFAULT 'PENDING',
    allocated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    accepted_at TIMESTAMP,
    completed_at TIMESTAMP,
    feedback_score INT, -- User feedback on allocation quality (1-5)
    feedback_comments TEXT,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

### 10. Performance_Metrics

Tracks user performance metrics for task allocation.

```sql
CREATE TABLE performance_metrics (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value FLOAT NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE (user_id, metric_name, calculated_at)
);
```

### 11. Allocation_History

Stores historical data about task allocations for learning.

```sql
CREATE TABLE allocation_history (
    id VARCHAR(36) PRIMARY KEY,
    task_id VARCHAR(36) NOT NULL,
    user_id VARCHAR(36) NOT NULL,
    allocation_score FLOAT NOT NULL,
    actual_completion_time INT, -- in minutes
    quality_rating INT, -- 1-5 scale
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);
```

## Indexes

```sql
-- User Skills Indexes
CREATE INDEX idx_user_skills_user_id ON user_skills(user_id);
CREATE INDEX idx_user_skills_skill_id ON user_skills(skill_id);
CREATE INDEX idx_user_skills_proficiency ON user_skills(proficiency_level);

-- Availability Indexes
CREATE INDEX idx_availability_user_id ON availability(user_id);
CREATE INDEX idx_availability_time_range ON availability(start_time, end_time);

-- Tasks Indexes
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_deadline ON tasks(deadline);
CREATE INDEX idx_tasks_priority ON tasks(priority);

-- Task Skills Indexes
CREATE INDEX idx_task_skills_task_id ON task_skills(task_id);
CREATE INDEX idx_task_skills_skill_id ON task_skills(skill_id);

-- Task Allocations Indexes
CREATE INDEX idx_task_allocations_task_id ON task_allocations(task_id);
CREATE INDEX idx_task_allocations_user_id ON task_allocations(user_id);
CREATE INDEX idx_task_allocations_status ON task_allocations(status);
```

## Relationships

1. Users have many Skills (through User_Skills)
2. Users have many Preferences (through User_Preferences)
3. Users have many Availability slots
4. Users have many Performance_Metrics
5. Tasks require many Skills (through Task_Skills)
6. Tasks have many Dependencies (through Task_Dependencies)
7. Tasks can be allocated to Users (through Task_Allocations)
8. Historical allocations are stored in Allocation_History

## Data Flow

1. When a new task is created, the system records its details in the Tasks table and its required skills in the Task_Skills table.
2. The task matching algorithm queries User_Skills to find users with matching skills at appropriate proficiency levels.
3. The algorithm checks Availability to ensure users have time to complete the task before the deadline.
4. User_Preferences and Performance_Metrics are considered to refine the matching.
5. The best match is recorded in Task_Allocations.
6. After task completion, metrics are updated in Performance_Metrics and the allocation is archived in Allocation_History for future learning.

This schema design supports all the core requirements of the AI task allocation agent while providing flexibility for future enhancements.
