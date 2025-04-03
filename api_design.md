# AI Task Allocation Agent - API Design

## Overview

This document outlines the API design for the AI Task Allocation Agent. The API provides endpoints for managing tasks, users, skills, and allocations, as well as for interacting with the AI-powered task matching engine.

## Base URL

```
/api/v1
```

## Authentication

All API endpoints require authentication using JWT (JSON Web Tokens).

- **Header**: `Authorization: Bearer <token>`
- **Token Expiration**: 24 hours
- **Refresh Token Endpoint**: `/auth/refresh`

## Endpoints

### Authentication

#### Login

```
POST /auth/login
```

Request:
```json
{
  "email": "user@example.com",
  "password": "secure_password"
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "user-123",
    "name": "John Doe",
    "email": "user@example.com"
  }
}
```

#### Refresh Token

```
POST /auth/refresh
```

Request:
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

Response:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Users

#### Get All Users

```
GET /users
```

Query Parameters:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `search`: Search term for name or email
- `skills`: Filter by skill IDs (comma-separated)

Response:
```json
{
  "total": 100,
  "page": 1,
  "limit": 20,
  "users": [
    {
      "id": "user-123",
      "name": "John Doe",
      "email": "john@example.com",
      "skills": [
        {
          "id": "skill-456",
          "name": "Python",
          "proficiency": 5
        }
      ],
      "current_workload": 0.75
    }
  ]
}
```

#### Get User by ID

```
GET /users/{user_id}
```

Response:
```json
{
  "id": "user-123",
  "name": "John Doe",
  "email": "john@example.com",
  "skills": [
    {
      "id": "skill-456",
      "name": "Python",
      "proficiency": 5,
      "years_experience": 3.5
    }
  ],
  "preferences": {
    "preferred_task_types": ["development", "code-review"],
    "preferred_working_hours": {
      "start": "09:00",
      "end": "17:00"
    },
    "max_concurrent_tasks": 3
  },
  "performance_metrics": {
    "task_completion_rate": 0.95,
    "average_quality_rating": 4.8,
    "on_time_completion_rate": 0.92
  },
  "current_workload": 0.75,
  "availability": [
    {
      "start_time": "2025-04-03T09:00:00Z",
      "end_time": "2025-04-03T12:00:00Z",
      "availability_type": "AVAILABLE"
    }
  ]
}
```

#### Create User

```
POST /users
```

Request:
```json
{
  "name": "Jane Smith",
  "email": "jane@example.com",
  "password": "secure_password"
}
```

Response:
```json
{
  "id": "user-124",
  "name": "Jane Smith",
  "email": "jane@example.com",
  "created_at": "2025-04-02T18:00:00Z"
}
```

#### Update User

```
PUT /users/{user_id}
```

Request:
```json
{
  "name": "Jane Smith-Johnson",
  "email": "jane.updated@example.com"
}
```

Response:
```json
{
  "id": "user-124",
  "name": "Jane Smith-Johnson",
  "email": "jane.updated@example.com",
  "updated_at": "2025-04-02T18:30:00Z"
}
```

#### Delete User

```
DELETE /users/{user_id}
```

Response:
```json
{
  "message": "User deleted successfully"
}
```

### Skills

#### Get All Skills

```
GET /skills
```

Query Parameters:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 50)
- `category`: Filter by category
- `search`: Search term for skill name

Response:
```json
{
  "total": 200,
  "page": 1,
  "limit": 50,
  "skills": [
    {
      "id": "skill-456",
      "name": "Python",
      "description": "Python programming language",
      "category": "Programming"
    }
  ]
}
```

#### Get Skill by ID

```
GET /skills/{skill_id}
```

Response:
```json
{
  "id": "skill-456",
  "name": "Python",
  "description": "Python programming language",
  "category": "Programming",
  "related_skills": [
    {
      "id": "skill-457",
      "name": "Django",
      "relationship": "framework"
    }
  ]
}
```

#### Create Skill

```
POST /skills
```

Request:
```json
{
  "name": "React",
  "description": "React JavaScript library",
  "category": "Frontend"
}
```

Response:
```json
{
  "id": "skill-458",
  "name": "React",
  "description": "React JavaScript library",
  "category": "Frontend",
  "created_at": "2025-04-02T19:00:00Z"
}
```

#### Update Skill

```
PUT /skills/{skill_id}
```

Request:
```json
{
  "description": "React JavaScript library for building user interfaces",
  "category": "Frontend Development"
}
```

Response:
```json
{
  "id": "skill-458",
  "name": "React",
  "description": "React JavaScript library for building user interfaces",
  "category": "Frontend Development",
  "updated_at": "2025-04-02T19:15:00Z"
}
```

#### Delete Skill

```
DELETE /skills/{skill_id}
```

Response:
```json
{
  "message": "Skill deleted successfully"
}
```

### User Skills

#### Add Skill to User

```
POST /users/{user_id}/skills
```

Request:
```json
{
  "skill_id": "skill-456",
  "proficiency": 4,
  "years_experience": 2.5
}
```

Response:
```json
{
  "id": "user-skill-789",
  "user_id": "user-123",
  "skill_id": "skill-456",
  "skill_name": "Python",
  "proficiency": 4,
  "years_experience": 2.5,
  "created_at": "2025-04-02T19:30:00Z"
}
```

#### Update User Skill

```
PUT /users/{user_id}/skills/{skill_id}
```

Request:
```json
{
  "proficiency": 5,
  "years_experience": 3.0
}
```

Response:
```json
{
  "id": "user-skill-789",
  "user_id": "user-123",
  "skill_id": "skill-456",
  "skill_name": "Python",
  "proficiency": 5,
  "years_experience": 3.0,
  "updated_at": "2025-04-02T19:45:00Z"
}
```

#### Remove Skill from User

```
DELETE /users/{user_id}/skills/{skill_id}
```

Response:
```json
{
  "message": "Skill removed from user successfully"
}
```

### Tasks

#### Get All Tasks

```
GET /tasks
```

Query Parameters:
- `page`: Page number (default: 1)
- `limit`: Items per page (default: 20)
- `status`: Filter by status (PENDING, ASSIGNED, IN_PROGRESS, COMPLETED, CANCELLED)
- `priority`: Filter by priority (1-5)
- `deadline_before`: Filter by deadline before date
- `deadline_after`: Filter by deadline after date
- `search`: Search term for task title or description

Response:
```json
{
  "total": 150,
  "page": 1,
  "limit": 20,
  "tasks": [
    {
      "id": "task-789",
      "title": "Implement login feature",
      "description": "Create login functionality with JWT authentication",
      "priority": 4,
      "deadline": "2025-04-10T17:00:00Z",
      "estimated_duration": 240,
      "status": "PENDING",
      "created_at": "2025-04-02T14:00:00Z"
    }
  ]
}
```

#### Get Task by ID

```
GET /tasks/{task_id}
```

Response:
```json
{
  "id": "task-789",
  "title": "Implement login feature",
  "description": "Create login functionality with JWT authentication",
  "priority": 4,
  "deadline": "2025-04-10T17:00:00Z",
  "estimated_duration": 240,
  "status": "ASSIGNED",
  "created_by": {
    "id": "user-123",
    "name": "John Doe"
  },
  "required_skills": [
    {
      "id": "skill-456",
      "name": "Python",
      "minimum_proficiency": 3,
      "importance": 5
    },
    {
      "id": "skill-458",
      "name": "React",
      "minimum_proficiency": 4,
      "importance": 4
    }
  ],
  "dependencies": [
    {
      "id": "task-788",
      "title": "Design authentication flow",
      "status": "COMPLETED"
    }
  ],
  "allocation": {
    "id": "allocation-321",
    "user": {
      "id": "user-124",
      "name": "Jane Smith"
    },
    "confidence_score": 0.92,
    "allocation_reason": "Jane has strong Python and React skills with availability matching the task deadline",
    "status": "ACCEPTED",
    "allocated_at": "2025-04-02T15:30:00Z",
    "accepted_at": "2025-04-02T16:00:00Z"
  },
  "created_at": "2025-04-02T14:00:00Z",
  "updated_at": "2025-04-02T16:00:00Z"
}
```

#### Create Task

```
POST /tasks
```

Request:
```json
{
  "title": "Implement user profile page",
  "description": "Create a user profile page with edit functionality",
  "priority": 3,
  "deadline": "2025-04-15T17:00:00Z",
  "estimated_duration": 180,
  "required_skills": [
    {
      "skill_id": "skill-458",
      "minimum_proficiency": 3,
      "importance": 5
    },
    {
      "skill_id": "skill-459",
      "minimum_proficiency": 2,
      "importance": 3
    }
  ],
  "dependencies": ["task-789"]
}
```

Response:
```json
{
  "id": "task-790",
  "title": "Implement user profile page",
  "description": "Create a user profile page with edit functionality",
  "priority": 3,
  "deadline": "2025-04-15T17:00:00Z",
  "estimated_duration": 180,
  "status": "PENDING",
  "created_by": {
    "id": "user-123",
    "name": "John Doe"
  },
  "created_at": "2025-04-02T20:00:00Z"
}
```

#### Update Task

```
PUT /tasks/{task_id}
```

Request:
```json
{
  "title": "Implement user profile page with avatar",
  "priority": 4,
  "deadline": "2025-04-14T17:00:00Z"
}
```

Response:
```json
{
  "id": "task-790",
  "title": "Implement user profile page with avatar",
  "description": "Create a user profile page with edit functionality",
  "priority": 4,
  "deadline": "2025-04-14T17:00:00Z",
  "estimated_duration": 180,
  "status": "PENDING",
  "updated_at": "2025-04-02T20:15:00Z"
}
```

#### Delete Task

```
DELETE /tasks/{task_id}
```

Response:
```json
{
  "message": "Task deleted successfully"
}
```

### Task Allocation

#### Allocate Task

```
POST /tasks/{task_id}/allocate
```

Request:
```json
{
  "auto_allocate": true
}
```

Response:
```json
{
  "id": "allocation-322",
  "task": {
    "id": "task-790",
    "title": "Implement user profile page with avatar"
  },
  "user": {
    "id": "user-125",
    "name": "Alex Johnson"
  },
  "confidence_score": 0.87,
  "allocation_reason": "Alex has excellent React skills and experience with user interfaces. Currently has low workload and availability matches the deadline.",
  "status": "PENDING",
  "allocated_at": "2025-04-02T20:30:00Z"
}
```

#### Manual Allocation

```
POST /tasks/{task_id}/allocate
```

Request:
```json
{
  "user_id": "user-124"
}
```

Response:
```json
{
  "id": "allocation-322",
  "task": {
    "id": "task-790",
    "title": "Implement user profile page with avatar"
  },
  "user": {
    "id": "user-124",
    "name": "Jane Smith"
  },
  "confidence_score": null,
  "allocation_reason": "Manual allocation by administrator",
  "status": "PENDING",
  "allocated_at": "2025-04-02T20:30:00Z"
}
```

#### Get Allocation Suggestions

```
GET /tasks/{task_id}/allocation-suggestions
```

Response:
```json
{
  "task": {
    "id": "task-790",
    "title": "Implement user profile page with avatar"
  },
  "suggestions": [
    {
      "user": {
        "id": "user-125",
        "name": "Alex Johnson"
      },
      "confidence_score": 0.87,
      "allocation_reason": "Alex has excellent React skills and experience with user interfaces. Currently has low workload and availability matches the deadline.",
      "current_workload": 0.3
    },
    {
      "user": {
        "id": "user-124",
        "name": "Jane Smith"
      },
      "confidence_score": 0.75,
      "allocation_reason": "Jane has good React skills but higher current workload.",
      "current_workload": 0.7
    }
  ]
}
```

#### Accept Task Allocation

```
PUT /tasks/{task_id}/allocation/accept
```

Response:
```json
{
  "id": "allocation-322",
  "task": {
    "id": "task-790",
    "title": "Implement user profile page with avatar"
  },
  "user": {
    "id": "user-125",
    "name": "Alex Johnson"
  },
  "status": "ACCEPTED",
  "accepted_at": "2025-04-02T21:00:00Z"
}
```

#### Reject Task Allocation

```
PUT /tasks/{task_id}/allocation/reject
```

Request:
```json
{
  "reason": "Currently working on higher priority tasks"
}
```

Response:
```json
{
  "id": "allocation-322",
  "task": {
    "id": "task-790",
    "title": "Implement user profile page with avatar"
  },
  "user": {
    "id": "user-125",
    "name": "Alex Johnson"
  },
  "status": "REJECTED",
  "rejection_reason": "Currently working on higher priority tasks",
  "rejected_at": "2025-04-02T21:00:00Z"
}
```

#### Complete Task

```
PUT /tasks/{task_id}/complete
```

Request:
```json
{
  "actual_duration": 210,
  "comments": "Completed with additional avatar upload functionality"
}
```

Response:
```json
{
  "id": "task-790",
  "title": "Implement user profile page with avatar",
  "status": "COMPLETED",
  "completed_at": "2025-04-10T15:30:00Z",
  "actual_duration": 210,
  "comments": "Completed with additional avatar upload functionality"
}
```

### Availability

#### Get User Availability

```
GET /users/{user_id}/availability
```

Query Parameters:
- `start_date`: Start date for availability window (default: today)
- `end_date`: End date for availability window (default: 7 days from start)

Response:
```json
{
  "user_id": "user-123",
  "availability": [
    {
      "id": "avail-456",
      "start_time": "2025-04-03T09:00:00Z",
      "end_time": "2025-04-03T12:00:00Z",
      "availability_type": "AVAILABLE"
    },
    {
      "id": "avail-457",
      "start_time": "2025-04-03T13:00:00Z",
      "end_time": "2025-04-03T17:00:00Z",
      "availability_type": "BUSY",
      "recurrence_rule": "FREQ=WEEKLY;BYDAY=MO,WE,FR"
    }
  ]
}
```

#### Add Availability

```
POST /users/{user_id}/availability
```

Request:
```json
{
  "start_time": "2025-04-04T09:00:00Z",
  "end_time": "2025-04-04T17:00:00Z",
  "availability_type": "AVAILABLE",
  "recurrence_rule": "FREQ=WEEKLY;BYDAY=FR"
}
```

Response:
```json
{
  "id": "avail-458",
  "user_id": "user-123",
  "start_time": "2025-04-04T09:00:00Z",
  "end_time": "2025-04-04T17:00:00Z",
  "availability_type": "AVAILABLE",
  "recurrence_rule": "FREQ=WEEKLY;BYDAY=FR",
  "created_at": "2025-04-02T21:30:00Z"
}
```

#### Update Availability

```
PUT /users/{user_id}/availability/{availability_id}
```

Request:
```json
{
  "start_time": "2025-04-04T10:00:00Z",
  "end_time": "2025-04-04T18:00:00Z"
}
```

Response:
```json
{
  "id": "avail-458",
  "user_id": "user-123",
  "start_time": "2025-04-04T10:00:00Z",
  "end_time": "2025-04-04T18:00:00Z",
  "availability_type": "AVAILABLE",
  "recurrence_rule": "FREQ=WEEKLY;BYDAY=FR",
  "updated_at": "2025-04-02T21:45:00Z"
}
```

#### Delete Availability

```
DELETE /users/{user_id}/availability/{availability_id}
```

Response:
```json
{
  "message": "Availability deleted successfully"
}
```

### Analytics

#### Get User Performance Metrics

```
GET /analytics/users/{user_id}/performance
```

Query Parameters:
- `start_date`: Start date for metrics (default: 30 days ago)
- `end_date`: End date for metrics (default: today)

Response:
```json
{
  "user_id": "user-123",
  "name": "John Doe",
  "metrics": {
    "tasks_completed": 15,
    "tasks_in_progress": 3,
    "average_completion_time": 0.92,
    "on_time_completion_rate": 0.87,
    "average_quality_rating": 4.6,
    "skill_utilization": [
      {
        "skill_id": "skill-456",
        "skill_name": "Python",
        "utilization_rate": 0.75
      }
    ]
  },
  "time_period": {
    "start_date": "2025-03-03T00:00:00Z",
    "end_date": "2025-04-02T23:59:59Z"
  }
}
```

#### Get Task Allocation Metrics

```
GET /analytics/tasks/allocation
```

Query Parameters:
- `start_date`: Start date for metrics (default: 30 days ago)
- `end_date`: End date for metrics (default: today)

Response:
```json
{
  "metrics": {
    "total_tasks": 120,
    "allocated_tasks": 105,
    "allocation_acceptance_rate": 0.92,
    "average_allocation_confidence": 0.85,
    "average_feedback_score": 4.3,
    "allocation_distribution": [
      {
        "user_id": "user-123",
        "name": "John Doe",
        "task_count": 25
      }
    ]
  },
  "time_period": {
    "start_date": "2025-03-03T00:00:00Z",
    "end_date": "2025-04-02T23:59:59Z"
  }
}
```

## Error Handling

All endpoints return standard HTTP status codes:

- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

Error Response Format:

```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "The request contains invalid parameters",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    }
  }
}
```

## Rate Limiting

API endpoints are rate-limited to prevent abuse:

- Standard users: 100 requests per minute
- Admin users: 300 requests per minute

Rate limit headers are included in all responses:

- `X-RateLimit-Limit`: Maximum requests per minute
- `X-RateLimit-Remaining`: Remaining requests in the current window
- `X-RateLimit-Reset`: Time when the rate limit resets (Unix timestamp)

## Versioning

The API uses URL versioning (e.g., `/api/v1`). When breaking changes are introduced, a new version will be released while maintaining support for previous versions for a deprecation period.

## Pagination

List endpoints support pagination with consistent parameters:

- `page`: Page number (1-based)
- `limit`: Items per page
- `sort`: Field to sort by
- `order`: Sort order (asc or desc)

Pagination metadata is included in all list responses:

```json
{
  "pagination": {
    "total": 100,
    "page": 2,
    "limit": 20,
    "pages": 5,
    "next": "/api/v1/users?page=3&limit=20",
    "prev": "/api/v1/users?page=1&limit=20"
  },
  "data": [...]
}
```

This API design provides a comprehensive interface for interacting with the AI Task Allocation Agent, supporting all the core functionality required for task matching, availability management, and personalization.
