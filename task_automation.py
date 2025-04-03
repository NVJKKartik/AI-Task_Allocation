"""
Task Automation Module for AI Task Allocation Agent

This module implements automated task analysis, processing, and management features.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pytz
import requests
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

@dataclass
class TaskAnalysis:
    """Stores the results of automated task analysis"""
    required_skills: Dict[str, int]  # skill: proficiency_level
    estimated_complexity: int  # 1-5
    estimated_duration: timedelta
    task_category: str
    dependencies: List[str]
    keywords: List[str]

class TaskAnalyzer:
    """Analyzes tasks using NLP and pattern recognition"""
    
    def __init__(self):
        """Initialize the task analyzer"""
        self.llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.2,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        self.setup_prompts()
    
    def setup_prompts(self):
        """Set up the prompt templates for analysis"""
        self.analysis_prompt = ChatPromptTemplate.from_template("""
Analyze the following task description:

Task: {description}

Extract the following information and return it ONLY as a JSON object with these exact keys:
- "required_skills": {{ A dictionary of skill names (string) to proficiency levels (integer 1-5). Infer from description. }} 
- "estimated_complexity": {{ An integer from 1 (simple) to 5 (very complex). }}
- "estimated_duration_hours": {{ A float representing the estimated hours needed. }}
- "task_category": {{ A string suggesting a category (e.g., Development, Design, Analysis). }}
- "dependencies": {{ A list of strings naming potential task dependencies. }}
- "keywords": {{ A list of relevant technical keywords (strings). }}

Example Output Structure:
{{"required_skills": {{"Python": 3}}, "estimated_complexity": 2, "estimated_duration_hours": 4.0, "task_category": "Development", "dependencies": ["API v2"], "keywords": ["python", "api"]}}
""")
        
        self.chain = self.analysis_prompt | self.llm | JsonOutputParser()
    
    def analyze_task(self, description: str) -> TaskAnalysis:
        """Analyze a task description and extract key information"""
        try:
            # First try to extract basic information using regex
            duration_match = re.search(r'(\d+)\s*(hour|hr|h)', description.lower())
            estimated_duration = timedelta(hours=float(duration_match.group(1))) if duration_match else timedelta(hours=2)
            
            # Use LLM for deeper analysis
            result = self.chain.invoke({"description": description})
            
            return TaskAnalysis(
                required_skills=result["required_skills"],
                estimated_complexity=result["estimated_complexity"],
                estimated_duration=timedelta(hours=result["estimated_duration_hours"]),
                task_category=result["task_category"],
                dependencies=result["dependencies"],
                keywords=result["keywords"]
            )
        except Exception as e:
            print(f"Error analyzing task: {str(e)}")
            # Return default analysis
            return TaskAnalysis(
                required_skills={},
                estimated_complexity=3,
                estimated_duration=timedelta(hours=2),
                task_category="General",
                dependencies=[],
                keywords=[]
            )

class TaskAutomationManager:
    """Manages automated task processing and management"""
    
    def __init__(self):
        """Initialize the task automation manager"""
        self.analyzer = TaskAnalyzer()
        self.setup_free_apis()
    
    def setup_free_apis(self):
        """Set up connections to free APIs"""
        # Free calendar API (using iCalendar format)
        self.calendar_api = "https://calendar.google.com/calendar/ical/{calendar_id}/public/basic.ics"
        
        # Free email API (using Gmail API with OAuth)
        self.email_api = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
        
        # Free project management API (using Trello API)
        self.trello_api = "https://api.trello.com/1"
    
    def create_task_from_email(self, email_content: str) -> Optional[Dict]:
        """Create a task from email content"""
        try:
            # Extract task information from email
            analysis = self.analyzer.analyze_task(email_content)
            
            # Create task object
            task = {
                "title": self._extract_title(email_content),
                "description": email_content,
                "required_skills": analysis.required_skills,
                "priority": self._determine_priority(email_content),
                "estimated_duration": analysis.estimated_duration,
                "category": analysis.task_category,
                "dependencies": analysis.dependencies,
                "keywords": analysis.keywords
            }
            
            return task
        except Exception as e:
            print(f"Error creating task from email: {str(e)}")
            return None
    
    def _extract_title(self, content: str) -> str:
        """Extract a title from content"""
        # Try to find a subject line or first line
        lines = content.split('\n')
        for line in lines:
            if line.strip() and len(line.strip()) < 100:
                return line.strip()
        return "New Task"
    
    def _determine_priority(self, content: str) -> int:
        """Determine task priority from content"""
        priority_keywords = {
            "urgent": 5,
            "asap": 5,
            "high priority": 4,
            "important": 4,
            "medium priority": 3,
            "normal": 3,
            "low priority": 2,
            "when possible": 2
        }
        
        content_lower = content.lower()
        for keyword, priority in priority_keywords.items():
            if keyword in content_lower:
                return priority
        
        return 3  # Default to medium priority
    
    def sync_with_calendar(self, user_id: str, calendar_id: str) -> List[Dict]:
        """Sync tasks with calendar events"""
        try:
            # Fetch calendar events
            response = requests.get(self.calendar_api.format(calendar_id=calendar_id))
            if response.status_code == 200:
                # Parse iCalendar data
                events = self._parse_icalendar(response.text)
                return events
            return []
        except Exception as e:
            print(f"Error syncing with calendar: {str(e)}")
            return []
    
    def _parse_icalendar(self, ical_data: str) -> List[Dict]:
        """Parse iCalendar data into events"""
        events = []
        current_event = {}
        
        for line in ical_data.split('\n'):
            if line.startswith('BEGIN:VEVENT'):
                current_event = {}
            elif line.startswith('END:VEVENT'):
                events.append(current_event)
            else:
                key, value = line.split(':', 1)
                current_event[key] = value
        
        return events
    
    def monitor_task_progress(self, task_id: str) -> Dict:
        """Monitor and analyze task progress"""
        # This would integrate with your existing task tracking system
        return {
            "completion_percentage": 0,
            "time_spent": timedelta(),
            "remaining_time": timedelta(),
            "blockers": [],
            "milestones": []
        }
    
    def generate_documentation(self, task: Dict) -> str:
        """Generate documentation for a task"""
        template = """
        # Task Documentation
        
        ## Overview
        Title: {title}
        Category: {category}
        Priority: {priority}
        
        ## Description
        {description}
        
        ## Requirements
        - Estimated Duration: {duration}
        - Required Skills: {skills}
        - Dependencies: {dependencies}
        
        ## Keywords
        {keywords}
        """
        
        return template.format(
            title=task["title"],
            category=task["category"],
            priority=task["priority"],
            description=task["description"],
            duration=str(task["estimated_duration"]),
            skills=", ".join(f"{skill} ({level})" for skill, level in task["required_skills"].items()),
            dependencies=", ".join(task["dependencies"]),
            keywords=", ".join(task["keywords"])
        )

# Example usage
if __name__ == "__main__":
    # Test task analysis
    analyzer = TaskAnalyzer()
    sample_task = """
    Create a login page with React and Node.js
    Estimated time: 4 hours
    Requires: React (level 4), Node.js (level 3)
    Dependencies: User authentication API
    """
    
    analysis = analyzer.analyze_task(sample_task)
    print("Task Analysis:")
    print(f"Required Skills: {analysis.required_skills}")
    print(f"Complexity: {analysis.estimated_complexity}")
    print(f"Duration: {analysis.estimated_duration}")
    print(f"Category: {analysis.task_category}")
    print(f"Dependencies: {analysis.dependencies}")
    print(f"Keywords: {analysis.keywords}")
    
    # Test automation manager
    manager = TaskAutomationManager()
    task = manager.create_task_from_email(sample_task)
    if task:
        print("\nGenerated Task:")
        print(json.dumps(task, indent=2, default=str))
        
        print("\nGenerated Documentation:")
        print(manager.generate_documentation(task)) 