# AI Task Allocation Agent - Usage Instructions

This guide provides step-by-step instructions for using the AI Task Allocation Agent to efficiently assign tasks to team members based on their skills, availability, and preferences.

## Getting Started

### Accessing the Application

The AI Task Allocation Agent is a web-based application that can be accessed through your browser. The application is currently running at:

```
https://8501-it12uq9ir01e597pix54h-58220894.manus.computer
```

### Initial Setup

When you first access the application, you'll see the dashboard page. To get started quickly:

1. Click the **Load Sample Data** button in the sidebar to populate the system with sample users and tasks
2. Use the sidebar navigation to explore different sections of the application

## Navigation

The application is divided into five main sections:

1. **Dashboard**: Overview of users, tasks, and recent allocations
2. **Users**: Manage user profiles with skills, preferences, and performance metrics
3. **Tasks**: Create and manage tasks with required skills, priorities, and deadlines
4. **Availability**: Manage user availability with calendar views and scheduling tools
5. **Task Allocation**: AI-powered matching of tasks to users with detailed explanations

## Managing Users

### Viewing Users

1. Navigate to the **Users** page using the sidebar
2. Existing users are displayed as expandable cards
3. Click on a user card to view details about their:
   - Skills and proficiency levels
   - Task type preferences
   - Performance metrics
   - Current workload
   - Availability status

### Adding a New User

1. Scroll down to the **Add New User** form
2. Enter the user's name
3. Set skill levels using the sliders (0-5 rating)
4. Select preferred task types from the dropdown
5. Set initial performance metrics and workload
6. Click **Add User** to create the user profile

### Updating User Information

Currently, user information must be updated by removing and re-adding the user. Future versions will support direct editing of user profiles.

## Managing Tasks

### Viewing Tasks

1. Navigate to the **Tasks** page using the sidebar
2. Existing tasks are displayed as expandable cards
3. Click on a task card to view details about:
   - Description and requirements
   - Required skills and proficiency levels
   - Priority and deadline
   - Estimated duration
   - Allocation status

### Adding a New Task

1. Scroll down to the **Add New Task** form
2. Enter the task title and description
3. Set required skill levels using the sliders (0-5 rating)
4. Set the priority level (1-5)
5. Select the deadline date and time
6. Enter the estimated duration in hours
7. Click **Add Task** to create the task

## Managing Availability

### Viewing User Availability

1. Navigate to the **Availability** page using the sidebar
2. Select a user from the dropdown menu
3. View the calendar heatmap showing availability percentages for each day
4. Below the calendar, you can see available and busy time slots for the next 7 days

### Adding Availability

1. Select the user whose availability you want to manage
2. Choose between **Single Time Slot** or **Recurring Time Slot** tabs
3. For a single time slot:
   - Select the date
   - Set start and end times
   - Choose availability type (AVAILABLE, BUSY, OUT_OF_OFFICE)
   - Click **Add Time Slot**
4. For a recurring time slot:
   - Select the start date
   - Set start and end times
   - Choose frequency (Daily, Weekly, Monthly)
   - For weekly recurrence, select days of the week
   - Choose availability type
   - Click **Add Recurring Time Slot**

### Finding Common Availability

1. Select the primary user from the dropdown
2. In the **Find Common Availability** section, select additional users
3. Set the required duration in hours
4. Click **Find Common Availability**
5. The system will find the next time when all selected users are available for the specified duration

## Task Allocation

### Allocating a Task

1. Navigate to the **Task Allocation** page using the sidebar
2. Select an unallocated task from the dropdown
3. Review the task details displayed below
4. Click **Match Task to Users**
5. The AI agent will analyze all users and find the best match based on:
   - Skill matching
   - Availability
   - Workload balancing
   - Performance history
   - User preferences
   - Predicted satisfaction

### Understanding Allocation Results

After matching a task to users, you'll see:

1. **Allocation Details**:
   - The assigned user
   - Confidence score (how confident the AI is about this match)
   - Detailed reason for the allocation

2. **Skill Comparison**:
   - A radar chart comparing the user's skills with the task requirements

3. **Alternative Matches**:
   - Other potential users who could be assigned the task
   - Their confidence scores and brief reasons

### Viewing Existing Allocations

If there are no unallocated tasks, the Task Allocation page will show existing allocations. For each allocation, you can:

1. View the task and assigned user
2. See the confidence score and allocation reason
3. View the skill comparison chart

## Dashboard

The Dashboard provides an overview of the system status:

1. **Summary Statistics**:
   - Total number of users
   - Total number of tasks
   - Number of allocated tasks

2. **Recent Task Allocations**:
   - List of the most recent task allocations
   - Click on any allocation to view details

## Advanced Features

### Personalization

The AI Task Allocation Agent learns from past allocations to improve future matches:

1. **Learning User Preferences**:
   - The system tracks which tasks users complete with high satisfaction
   - It identifies patterns in preferred task types and skills

2. **Performance Tracking**:
   - Completion rates and on-time rates are tracked
   - This information influences future allocations

3. **Satisfaction Prediction**:
   - The AI predicts how satisfied a user will be with a task
   - This prediction is factored into the matching algorithm

### Personalized Recommendations

The system can provide personalized task recommendations for each user based on their skills, preferences, and past performance. This feature is currently available through the API and will be added to the UI in a future update.

## Troubleshooting

### Common Issues

1. **Task Not Appearing for Allocation**:
   - Ensure the task has not already been allocated
   - Verify that the task has all required fields filled out

2. **User Not Available**:
   - Check the user's availability schedule
   - Ensure the user has available time slots that match the task duration

3. **Low Confidence Scores**:
   - This may indicate that no user is an ideal match for the task
   - Consider adding users with the required skills or adjusting the task requirements

### Getting Help

For additional assistance:

1. Refer to the comprehensive documentation in the `/docs` folder
2. Check the API reference for programmatic access
3. Contact the development team for support

## Next Steps

To get the most out of the AI Task Allocation Agent:

1. **Add Real User Data**:
   - Replace sample data with your actual team members and tasks
   - Set up accurate availability schedules

2. **Record Task Outcomes**:
   - Track completion status, quality, and user satisfaction
   - This data improves the personalization features

3. **Regular Updates**:
   - Keep user skills and preferences up to date
   - Update availability schedules regularly

4. **Explore the API**:
   - For advanced users, the API allows integration with other systems
   - See the API reference in the documentation
