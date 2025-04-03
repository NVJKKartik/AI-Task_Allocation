"""AI Agent Core Logic for Task Allocation - Using Function Calling"""

import os
import json
import uuid
from datetime import datetime, timedelta
import pytz
from enum import Enum
import random
from typing import Dict, List, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
# Assuming these modules are in the same directory or accessible
from task_matching_algorithm import Task as TaskModel, UserProfile # Use alias
from task_automation import TaskAnalyzer
from personalization import AllocationHistory # Import for allocation record
from availability_management import AvailabilityManager # Needed for workload?

# --- Enums (Define BEFORE use) --- #
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

# --- Tool Schemas (using Pydantic for clarity) ---

class AddTaskSchema(BaseModel):
    """Tool to add a new task to the system."""
    title: str = Field(..., description="Concise title for the task.")
    description: str = Field(..., description="Detailed description of the task requirements.")
    priority: Optional[int] = Field(default=3, description="Priority from 1 (Low) to 5 (High). Default is 3.")
    deadline_days: Optional[int] = Field(default=7, description="Number of days from now until the deadline. Default is 7.")

class UpdateStatusSchema(BaseModel):
    """Tool to update the status of an existing task."""
    task_id: str = Field(..., description="The unique ID of the task to update (e.g., 'task-uuid-xyz').")
    new_status: str = Field(..., description=f"The new status. Must be one of: {', '.join([s.value for s in TaskStatus])}.") # Now TaskStatus is defined

class AddUserSchema(BaseModel):
    """Tool to add a new user to the system."""
    name: str = Field(..., description="The full name of the user.")
    skills_description: str = Field(..., description="A natural language description of the user's skills and proficiency (e.g., 'Expert Python, intermediate React, beginner Figma').")
    # Add other fields like email if needed later

class AllocateTaskSchema(BaseModel):
    """Tool to manually allocate a task to a specific user."""
    task_id: str = Field(..., description="The unique ID of the task to allocate.")
    user_id: str = Field(..., description="The unique ID of the user to assign the task to.")
    reason: Optional[str] = Field(default="Allocation via chat agent.", description="Optional reason for this manual allocation.")

class GetTaskDetailsSchema(BaseModel):
    """Tool to retrieve details about a specific task."""
    task_id: str = Field(..., description="The unique ID of the task to retrieve details for.")

class GetUserTasksSchema(BaseModel):
    """Tool to list tasks currently assigned to a specific user."""
    user_id: str = Field(..., description="The unique ID of the user whose tasks should be listed.")
    # Optional: Add status filtering later (e.g., only IN_PROGRESS tasks)

class DeleteTaskSchema(BaseModel):
    """Tool to delete an existing task from the system."""
    task_id: str = Field(..., description="The unique ID of the task to delete.")

class GetUserDetailsSchema(BaseModel):
    """Tool to retrieve details about a specific user."""
    user_id: str = Field(..., description="The unique ID of the user (e.g., 'user-uuid-xyz') to retrieve details for.")

class DeleteUserSchema(BaseModel):
    """Tool to delete an existing user from the system."""
    user_id: str = Field(..., description="The unique ID of the user to delete.")

class QueryTasksSchema(BaseModel):
    """Tool to query tasks based on criteria like status. Use this for system-wide queries (not specific to one user)."""
    status: Optional[str] = Field(default=None, description=f"Filter tasks by status. Valid options: {', '.join([s.value for s in TaskStatus])}. If omitted, shows tasks with any status (useful for general queries, but might be long). Often defaults to PENDING if not specified by user for queries like 'what tasks need doing?'.")
    # Add other filters later, like priority or deadline range

class CountUsersSchema(BaseModel):
    """Tool to count the total number of users currently in the system."""
    pass # No arguments needed

# --- Agent Class ---

class TaskAllocationAgent:
    """AI Agent using Function Calling/Tool Use."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.task_analyzer = TaskAnalyzer()
        # Define available tools
        self.tools = [convert_to_openai_tool(func) for func in [
            AddTaskSchema, 
            UpdateStatusSchema,
            AddUserSchema,
            AllocateTaskSchema,
            GetTaskDetailsSchema,
            GetUserTasksSchema,
            DeleteTaskSchema,
            GetUserDetailsSchema,
            DeleteUserSchema,
            QueryTasksSchema,
            CountUsersSchema
        ]]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        # Store mapping from tool name to execution method
        self.tool_map = {
            "AddTaskSchema": self._execute_add_task,
            "UpdateStatusSchema": self._execute_update_status,
            "AddUserSchema": self._execute_add_user,
            "AllocateTaskSchema": self._execute_allocate_task,
            "GetTaskDetailsSchema": self._execute_get_task_details,
            "GetUserTasksSchema": self._execute_get_user_tasks,
            "DeleteTaskSchema": self._execute_delete_task,
            "GetUserDetailsSchema": self._execute_get_user_details,
            "DeleteUserSchema": self._execute_delete_user,
            "QueryTasksSchema": self._execute_query_tasks,
            "CountUsersSchema": self._execute_count_users,
        }
        self._setup_skill_parser_chain() # Add chain for parsing skills

    def _setup_skill_parser_chain(self):
        """Sets up an LLM chain specifically for parsing skill descriptions."""
        skill_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert skill extractor. Analyze the provided text for skills and their proficiency levels (1-5 scale: 1=Beginner, 5=Expert). If proficiency isn't mentioned, estimate it based on context (e.g., 'very good' might be 4 or 5, 'familiar with' might be 2 or 3).
            Output ONLY a valid JSON object mapping skill names (string) to proficiency levels (integer).
            Example Input: 'Expert Python, familiar with React, good at Figma and can use blender'
            Example Output: {{"Python": 5, "React": 2, "Figma": 4, "blender": 3}} # Escaped curly braces
            If no skills are clearly mentioned or parseable, return an empty JSON object {{}}.
            DO NOT include any other text, explanations, or markdown formatting.
            """),
            ("human", "{skill_description}")
        ])
        # Use a separate parser to avoid interfering with the main tool parser
        self.skill_parser_chain = skill_prompt | self.llm | JsonOutputParser()

    def _execute_add_task(self, title: str, description: str, priority: int = 3, deadline_days: int = 7, session_state: Optional[dict] = None) -> str:
        """Executes the logic to add a task."""
        if not session_state:
            return "Error: Session state not provided."
        try:
            analysis = self.task_analyzer.analyze_task(f"Title: {title}\nDescription: {description}")
            task_id = f"task-{uuid.uuid4()}"
            new_task = TaskModel(
                task_id=task_id,
                title=title,
                description=description,
                required_skills=analysis.required_skills,
                priority=priority,
                deadline=datetime.now(pytz.UTC) + timedelta(days=deadline_days),
                estimated_duration=analysis.estimated_duration,
                status=TaskStatus.PENDING.value
            )
            if 'tasks' not in session_state:
                session_state['tasks'] = {}
            session_state['tasks'][task_id] = new_task
            return f"Successfully added task '{title}' (ID: {task_id}) with skills: {analysis.required_skills}."
        except Exception as e:
            print(f"Error executing add_task: {e}")
            return f"Error adding task: {str(e)}"

    def _execute_update_status(self, task_id: str, new_status: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to update task status."""
        if not session_state:
            return "Error: Session state not provided."
        try:
            # Validate status
            valid_statuses = [s.value for s in TaskStatus]
            if new_status not in valid_statuses:
                 return f"Invalid status '{new_status}'. Valid statuses are: {', '.join(valid_statuses)}."
            # Check task existence
            if 'tasks' not in session_state or task_id not in session_state['tasks']:
                return f"Error: Task with ID {task_id} not found."
            # Update status
            session_state['tasks'][task_id].status = new_status
            # TODO: Update user workload if necessary
            return f"Successfully updated task {task_id} status to {new_status}."
        except Exception as e:
            print(f"Error executing update_status: {e}")
            return f"Error updating status: {str(e)}"

    def _execute_add_user(self, name: str, skills_description: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to add a new user."""
        if not session_state:
            return "Error: Session state not provided."
        parsed_skills = {"General": 1} # Default to basic general skill if parsing fails
        try:
            # Parse skills using the dedicated chain
            raw_parsed_output = self.skill_parser_chain.invoke({"skill_description": skills_description})
            print(f"--- Skill Parser Raw Output: {raw_parsed_output} ---") # DEBUG PRINT
            if isinstance(raw_parsed_output, dict) and raw_parsed_output: # Check if it's a non-empty dict
                # Validate structure (simple check: keys are strings, values are ints)
                validated_skills = {k: int(v) for k, v in raw_parsed_output.items() if isinstance(k, str) and isinstance(v, (int, float, str)) and str(v).isdigit()}
                if validated_skills:
                    parsed_skills = validated_skills
                else:
                    print("Skill parser output validation failed, using default.")
            elif isinstance(raw_parsed_output, dict) and not raw_parsed_output: # Handle empty dict {} case
                parsed_skills = {} # No skills found is valid
            else:
                print(f"Skill parser returned unexpected type: {type(raw_parsed_output)}, using default.")

        except Exception as parse_err:
            print(f"Error invoking/parsing skill parser chain: {parse_err}")
            # Keep the default parsed_skills = {"General": 1}
            
        try: 
            user_id = f"user-{uuid.uuid4()}"
            new_user = UserProfile(
                user_id=user_id,
                name=name,
                skills=parsed_skills,
                availability=[],
                preferences={},
                performance_metrics={"task_completion_rate": 0.0, "on_time_rate": 0.0},
                current_workload=0.0
            )
            if 'users' not in session_state:
                session_state['users'] = {}
            session_state['users'][user_id] = new_user
            return f"Successfully added user '{name}' (ID: {user_id}) with skills: {json.dumps(parsed_skills) if parsed_skills else 'None specified'}."
        except Exception as e:
            print(f"Error executing add_user after parsing skills: {e}")
            return f"Error adding user: {str(e)}"

    def _execute_allocate_task(self, task_id: str, user_id: str, reason: str = "Allocation via chat agent.", session_state: Optional[dict] = None) -> str:
        """Executes the logic to manually allocate a task, checking availability."""
        if not session_state:
            return "Error: Session state not provided."
        try:
            # Validate task and user exist
            if 'tasks' not in session_state or task_id not in session_state['tasks']:
                return f"Error: Task with ID {task_id} not found."
            if 'users' not in session_state or user_id not in session_state['users']:
                return f"Error: User with ID {user_id} not found."
            if 'availability_manager' not in session_state:
                return "Error: Availability Manager not found in session state."

            task = session_state['tasks'][task_id]
            user = session_state['users'][user_id]
            availability_manager = session_state['availability_manager']
            simulated_now = session_state.get('simulated_now', datetime.now(pytz.UTC))

            # Check if task is already assigned
            if task.status != TaskStatus.PENDING.value:
                return f"Error: Task {task_id} is not pending (current status: {task.status}). Cannot reallocate directly via chat yet."
            
            # --- Modified AVAILABILITY CHECK --- #
            user_availability = availability_manager.get_user_availability(user_id)
            
            # Check deadline first
            if task.deadline.astimezone(pytz.UTC) <= simulated_now:
                return f"Error: Task {task_id} deadline has passed relative to current simulated time ({simulated_now.strftime('%H:%M %Z')}). Cannot allocate."
            
            # Calculate max days to search ahead
            days_diff = (task.deadline.astimezone(pytz.UTC) - simulated_now).days
            max_days = max(1, min(days_diff + 1, 30))
                
            next_slot = user_availability.find_next_available_slot(
                start_time=simulated_now, 
                duration=task.estimated_duration,
                max_days_ahead=max_days # Use max_days_ahead
            )
            if next_slot is None:
                return f"Error: User {user.name} (ID: {user_id}) does not have a suitable time slot available starting from {simulated_now.strftime('%H:%M %Z')} until the deadline for the task duration ({task.estimated_duration}). Allocation failed."
            # --- END Modified AVAILABILITY CHECK --- #

            # Create allocation record
            allocation_id = str(uuid.uuid4())
            if 'allocations' not in session_state:
                session_state['allocations'] = {}
            session_state['allocations'][allocation_id] = {
                "task_id": task.task_id,
                "user_id": user.user_id,
                "allocated_at": simulated_now, # Use simulated time for allocation record
                "status": AllocationStatus.ACCEPTED.value,
                "confidence_score": 1.0, # Manual allocation
                "allocation_reason": reason
            }
            # This mimics the AllocationHistory structure used elsewhere, store directly in state for now
            # If AllocationHistory class is used centrally, use that: 
            # allocation_record = AllocationHistory().add_allocation(task, user, ...) # Needs adaptation

            # Update task status
            task.status = TaskStatus.ASSIGNED.value
            session_state['tasks'][task_id] = task

            # Update user workload (simple increment for now)
            user.current_workload = min(1.0, user.current_workload + 0.2) # Adjust as needed
            session_state['users'][user_id] = user

            return f"Successfully allocated task '{task.title}' (ID: {task_id}) to user '{user.name}' (ID: {user_id})."
        except Exception as e:
            print(f"Error executing allocate_task for {task_id} to {user_id}: {e}")
            return f"Error allocating task: {str(e)}"

    def _execute_get_task_details(self, task_id: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to retrieve and format task details."""
        if not session_state:
            return "Error: Session state not provided."
        if 'tasks' not in session_state or task_id not in session_state['tasks']:
            return f"Error: Task with ID {task_id} not found."
        
        try:
            task = session_state['tasks'][task_id]
            # Format the details nicely for the chat
            details = f"Task Details for {task_id}:\n"
            details += f"- Title: {task.title}\n"
            details += f"- Description: {task.description}\n"
            details += f"- Status: {task.status}\n"
            details += f"- Priority: {task.priority}\n"
            details += f"- Deadline: {task.deadline.strftime('%Y-%m-%d %H:%M %Z')}\n"
            details += f"- Estimated Duration: {task.estimated_duration}\n"
            details += f"- Required Skills: {json.dumps(task.required_skills)}\n"
            
            # Find assigned user
            assigned_user_name = "None"
            if task.status == TaskStatus.ASSIGNED.value or task.status == TaskStatus.IN_PROGRESS.value:
                 if 'allocations' in session_state and 'users' in session_state:
                     for alloc in session_state['allocations'].values():
                         if alloc["task_id"] == task_id and alloc["status"] == AllocationStatus.ACCEPTED.value:
                             user_id = alloc["user_id"]
                             assigned_user_name = session_state['users'].get(user_id, {}).get("name", f"ID: {user_id}")
                             break
            details += f"- Assigned To: {assigned_user_name}"
            
            return details
        except Exception as e:
            print(f"Error executing get_task_details for {task_id}: {e}")
            return f"Error retrieving details for task {task_id}: {str(e)}"
            
    def _execute_get_user_tasks(self, user_id: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to find and list tasks assigned to a user."""
        if not session_state:
            return "Error: Session state not provided."
        if 'users' not in session_state or user_id not in session_state['users']:
            return f"Error: User with ID {user_id} not found."
        if 'tasks' not in session_state or 'allocations' not in session_state:
            return "Error: Task or allocation data is missing."
        
        user_name = session_state['users'][user_id].name
        assigned_tasks = []
        try:
            for alloc in session_state['allocations'].values():
                if alloc["user_id"] == user_id and alloc["status"] == AllocationStatus.ACCEPTED.value:
                    task_id = alloc["task_id"]
                    if task_id in session_state['tasks']:
                        task = session_state['tasks'][task_id]
                        assigned_tasks.append(f"- {task.title} (ID: {task_id}, Status: {task.status})")
            
            if not assigned_tasks:
                return f"User {user_name} (ID: {user_id}) currently has no assigned tasks."
            else:
                return f"Tasks assigned to {user_name} (ID: {user_id}):\n" + "\n".join(assigned_tasks)
        except Exception as e:
            print(f"Error executing get_user_tasks for {user_id}: {e}")
            return f"Error retrieving tasks for user {user_id}: {str(e)}"

    def _execute_delete_task(self, task_id: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to delete a task."""
        if not session_state:
            return "Error: Session state not provided."
        if 'tasks' not in session_state:
            return "Error: No tasks exist in the system."
        
        if task_id not in session_state['tasks']:
            return f"Error: Task with ID {task_id} not found. Cannot delete."
        
        try:
            deleted_task_title = session_state['tasks'][task_id].title
            del session_state['tasks'][task_id]
            # Optional: Also remove any related allocations
            if 'allocations' in session_state:
                allocs_to_remove = [alloc_id for alloc_id, alloc in session_state['allocations'].items() if alloc["task_id"] == task_id]
                for alloc_id in allocs_to_remove:
                    del session_state['allocations'][alloc_id]
                print(f"Removed {len(allocs_to_remove)} allocations related to deleted task {task_id}")
            
            return f"Successfully deleted task '{deleted_task_title}' (ID: {task_id})."
        except Exception as e:
            print(f"Error executing delete_task for {task_id}: {e}")
            return f"Error deleting task {task_id}: {str(e)}"

    def _execute_get_user_details(self, user_id: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to retrieve and format user details."""
        if not session_state:
            return "Error: Session state not provided."
        if 'users' not in session_state or user_id not in session_state['users']:
            return f"Error: User with ID {user_id} not found."
        
        try:
            user = session_state['users'][user_id]
            details = f"User Details for {user_id}:\n"
            details += f"- Name: {user.name}\n"
            details += f"- Skills: {json.dumps(user.skills)}\n"
            details += f"- Current Workload: {user.current_workload:.1f}\n"
            details += f"- Preferences: {json.dumps(user.preferences)}\n"
            # Add assigned tasks if needed, similar to _execute_get_user_tasks
            return details
        except Exception as e:
            print(f"Error executing get_user_details for {user_id}: {e}")
            return f"Error retrieving details for user {user_id}: {str(e)}"

    def _execute_delete_user(self, user_id: str, session_state: Optional[dict] = None) -> str:
        """Executes the logic to delete a user."""
        if not session_state:
            return "Error: Session state not provided."
        if 'users' not in session_state:
            return "Error: No users exist in the system."
        
        if user_id not in session_state['users']:
            return f"Error: User with ID {user_id} not found. Cannot delete."
        
        try:
            deleted_user_name = session_state['users'][user_id].name
            del session_state['users'][user_id]
            # Optional: Unassign tasks currently assigned to this user?
            # This might be complex - maybe just note they are unassigned now.
            print(f"User {user_id} deleted. Associated tasks may need reallocation.")
            
            return f"Successfully deleted user '{deleted_user_name}' (ID: {user_id})."
        except Exception as e:
            print(f"Error executing delete_user for {user_id}: {e}")
            return f"Error deleting user {user_id}: {str(e)}"

    def _execute_query_tasks(self, status: Optional[str] = None, session_state: Optional[dict] = None) -> str:
        """Executes the logic to find and list tasks based on status."""
        if not session_state:
            return "Error: Session state not provided."
        if 'tasks' not in session_state or not session_state['tasks']:
            return "There are currently no tasks in the system."
        
        tasks_to_list = []
        target_status = status.upper() if status else None # Normalize status
        
        # Validate status if provided
        if target_status and target_status not in [s.value for s in TaskStatus]:
            valid_statuses = [s.value for s in TaskStatus]
            return f"Invalid status '{status}' provided for query. Valid options are: {', '.join(valid_statuses)}."
            
        try:
            for task_id, task in session_state['tasks'].items():
                if target_status is None or task.status == target_status:
                    # Include assigned user info if applicable
                    assigned_user_name = "Unassigned"
                    if task.status in [TaskStatus.ASSIGNED.value, TaskStatus.IN_PROGRESS.value]:
                        if 'allocations' in session_state and 'users' in session_state:
                           for alloc in session_state['allocations'].values():
                                if alloc["task_id"] == task_id and alloc["status"] == AllocationStatus.ACCEPTED.value:
                                    user_id = alloc["user_id"]
                                    assigned_user_name = session_state['users'].get(user_id, {}).get("name", f"ID: {user_id}")
                                    break
                    tasks_to_list.append(f"- {task.title} (ID: {task_id}, Status: {task.status}, Assigned: {assigned_user_name})")
            
            if not tasks_to_list:
                if target_status:
                    return f"No tasks found with status '{target_status}'."
                else:
                    return "No tasks match the query."
            else:
                limit = 20 # Limit output length
                response = f"Found {len(tasks_to_list)} tasks" 
                if target_status:
                     response += f" with status '{target_status}'"
                response += ":\n" + "\n".join(tasks_to_list[:limit])
                if len(tasks_to_list) > limit:
                    response += f"\n... (and {len(tasks_to_list) - limit} more)"
                return response
        except Exception as e:
            print(f"Error executing query_tasks (status={status}): {e}")
            return f"Error querying tasks: {str(e)}"

    def _execute_count_users(self, session_state: Optional[dict] = None) -> str:
        """Executes the logic to count users."""
        if not session_state:
            return "Error: Session state not provided."
        
        num_users = len(session_state.get('users', {}))
        return f"There are currently {num_users} users in the system."

    def process_chat_message(self, user_message: str, chat_history: list, session_state: dict) -> str:
        """Processes message using LLM with Tool Use.
        
        Args:
            user_message: The latest message from the user.
            chat_history: The list of previous messages.
            session_state: The Streamlit session state dictionary.

        Returns:
            The agent's response string.
        """
        
        # Format history for LLM (excluding latest user message which is passed separately)
        history_messages = []
        for msg in chat_history[:-1]: # Exclude latest user message
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                # Check if this assistant message contained tool calls
                # This part needs careful implementation if we want the LLM to see its own tool use
                # For simplicity now, just add content. A more robust solution tracks tool_calls.
                history_messages.append(AIMessage(content=msg["content"]))
            # Add handling for ToolMessages if you store tool results in history

        # Append the latest user message
        history_messages.append(HumanMessage(content=user_message))

        try:
            # --- First LLM Call: Get response or tool calls --- #
            ai_response: AIMessage = self.llm_with_tools.invoke(history_messages)
            history_messages.append(ai_response) # Add AI response (potential tool call) to history

            # Check if the LLM requested tool calls
            if not ai_response.tool_calls:
                # No tool call requested, just return the text response
                return ai_response.content if ai_response.content else "I'm not sure how to respond to that."
            else:
                # --- Tool call(s) requested, Execute Tools --- #
                tool_results_messages: List[ToolMessage] = []
                executed_tool_summaries = [] # Store simple summaries for final prompt

                for tool_call in ai_response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_result_content = "Error: Tool execution failed."
                    if tool_name in self.tool_map:
                        try:
                            args = tool_call["args"]
                            args["session_state"] = session_state 
                            tool_result_content = self.tool_map[tool_name](**args)
                            # Create a simple summary for the final prompt
                            executed_tool_summaries.append(f"Action: {tool_name}, Result: {tool_result_content}")
                        except Exception as e:
                            print(f"Error executing tool {tool_name} with args {tool_call['args']}: {e}")
                            tool_result_content = f"Error executing tool {tool_name}: {str(e)}"
                            executed_tool_summaries.append(f"Action: {tool_name}, Result: Error - {str(e)}")
                    else:
                        print(f"Warning: Unknown tool requested: {tool_name}")
                        tool_result_content = f"Error: Unknown tool '{tool_name}' requested."
                        executed_tool_summaries.append(f"Action: {tool_name}, Result: Error - Unknown tool")
                    
                    tool_results_messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=tool_result_content
                    ))
                
                # --- Tool Results Processing --- #
                # Check if we got a simple success/error message back
                if tool_results_messages:
                    first_tool_content = tool_results_messages[0].content
                    if first_tool_content.startswith("Successfully") or first_tool_content.startswith("Error:"):
                        print("--- Returning direct tool result --- ") # Debug print
                        return first_tool_content
                    
                # --- Second LLM Call: Generate Summary (If result wasn't simple) --- #
                print("--- Calling LLM for final summary --- ") 
                
                # Construct the specific messages for the summarization call:
                # The AI's previous turn (containing the tool_calls) + The results of those calls
                summary_input_messages = [ai_response] + tool_results_messages
                
                # We don't need a complex prompt, just invoke the base LLM 
                # with the AI request and the tool results. It should naturally summarize.
                final_response_message = self.llm.invoke(summary_input_messages)
                
                return final_response_message.content

        except Exception as e:
            print(f"Error in process_chat_message: {e}")
            # Provide a more user-friendly error message if possible
            if "JSON serializable" in str(e):
                 return "Sorry, I encountered an internal error processing the previous action's result (Serialization)."
            # Add specific handling for the OpenAI API error if possible
            if "role 'tool' must be a response" in str(e):
                 return "Sorry, there was an issue structuring the conversation history for the API. Please try again or rephrase."
            return f"Sorry, an error occurred: {str(e)}"

# Example usage (conceptual)
if __name__ == "__main__":
    agent = TaskAllocationAgent()
    mock_session = {
        'tasks': {},
        'users': {},
        'allocations': {}
    }
    chat_hist = []

    def run_turn(user_input, history, session):
        print(f"\nUser: {user_input}")
        history.append({"role": "user", "content": user_input})
        response = agent.process_chat_message(user_input, history, session)
        print(f"Agent: {response}")
        history.append({"role": "assistant", "content": response}) # Assumes response is final text
        # In a real app, if response required tool calls, the agent message might differ

    run_turn("Hello!", chat_hist, mock_session)
    # Add a user and task first
    run_turn("Add user Jane Dev, expert in Python and React", chat_hist, mock_session)
    run_turn("Add task: Refactor auth module. Priority 5, 3 day deadline.", chat_hist, mock_session)

    # Get IDs from state (replace with actuals in real use)
    user_jane_id = list(mock_session['users'].keys())[0] if mock_session['users'] else 'user-unknown'
    task_refactor_id = list(mock_session['tasks'].keys())[0] if mock_session['tasks'] else 'task-unknown'
    
    # Test the query tools
    run_turn(f"What are the details for task {task_refactor_id}?", chat_hist, mock_session)
    run_turn(f"What tasks does {user_jane_id} have?", chat_hist, mock_session)
    
    # Test allocation then query again
    run_turn(f"Assign {task_refactor_id} to {user_jane_id}", chat_hist, mock_session)
    run_turn(f"What tasks does {user_jane_id} have now?", chat_hist, mock_session)

    # Test Delete
    task_id_to_delete = list(mock_session['tasks'].keys())[0] if mock_session['tasks'] else 'task-unknown'
    
    # Test Delete
    run_turn(f"Can you delete task {task_id_to_delete}?", chat_hist, mock_session)
    print(f"\nTasks after delete attempt: {mock_session['tasks']}")
    run_turn(f"Delete task non-existent-task", chat_hist, mock_session)

    # Test User Query & Delete
    user_id_to_query = list(mock_session['users'].keys())[0] if mock_session['users'] else 'user-unknown'
    user_id_to_delete = list(mock_session['users'].keys())[0] if mock_session['users'] else 'user-unknown'
    
    # Test User Query & Delete
    run_turn(f"Tell me about user {user_id_to_query}", chat_hist, mock_session)
    run_turn(f"Remove user {user_id_to_delete}", chat_hist, mock_session)
    print(f"\nUsers after delete attempt: {mock_session['users']}")

    # Test System Queries & Count
    run_turn("How many users are there?", chat_hist, mock_session)
    run_turn("what tasks are pending?", chat_hist, mock_session)
    run_turn("list all completed tasks", chat_hist, mock_session)
    run_turn("show all tasks", chat_hist, mock_session) # Test with no status filter 