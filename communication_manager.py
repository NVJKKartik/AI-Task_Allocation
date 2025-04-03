"""
Communication Manager Module for AI Task Allocation Agent

This module implements automated communication and notification features
including email notifications, status updates, and team coordination.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from dataclasses import dataclass
import pytz

@dataclass
class Notification:
    """Represents a notification to be sent"""
    recipient: str
    subject: str
    message: str
    priority: int  # 1-5, 5 being highest
    notification_type: str  # email, slack, etc.

class CommunicationManager:
    """Manages automated communication and notifications"""
    
    def __init__(self):
        """Initialize the communication manager"""
        self.setup_free_apis()
        self.notification_queue = []
    
    def setup_free_apis(self):
        """Set up connections to free APIs"""
        # Free email service (using Gmail SMTP)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        
        # Free notification service (using ntfy.sh)
        self.ntfy_api = "https://ntfy.sh"
        
        # Free team chat API (using Discord webhooks)
        self.discord_api = "https://discord.com/api/webhooks"
    
    def send_notification(self, notification: Notification) -> bool:
        """Send a notification through the appropriate channel"""
        try:
            if notification.notification_type == "email":
                return self._send_email(notification)
            elif notification.notification_type == "slack":
                return self._send_slack_message(notification)
            elif notification.notification_type == "discord":
                return self._send_discord_message(notification)
            elif notification.notification_type == "push":
                return self._send_push_notification(notification)
            else:
                print(f"Unknown notification type: {notification.notification_type}")
                return False
        except Exception as e:
            print(f"Error sending notification: {str(e)}")
            return False
    
    def _send_email(self, notification: Notification) -> bool:
        """Send an email notification"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = os.getenv('EMAIL_USER')
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            # Add message body
            msg.attach(MIMEText(notification.message, 'plain'))
            
            # Connect to SMTP server
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            
            # Login and send
            server.login(os.getenv('EMAIL_USER'), os.getenv('EMAIL_PASSWORD'))
            server.send_message(msg)
            server.quit()
            
            return True
        except Exception as e:
            print(f"Error sending email: {str(e)}")
            return False
    
    def _send_slack_message(self, notification: Notification) -> bool:
        """Send a Slack message"""
        try:
            # This would use Slack webhook URL
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                return False
            
            payload = {
                "text": f"*{notification.subject}*\n{notification.message}"
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Slack message: {str(e)}")
            return False
    
    def _send_discord_message(self, notification: Notification) -> bool:
        """Send a Discord message"""
        try:
            # This would use Discord webhook URL
            webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
            if not webhook_url:
                return False
            
            payload = {
                "content": f"**{notification.subject}**\n{notification.message}"
            }
            
            response = requests.post(webhook_url, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending Discord message: {str(e)}")
            return False
    
    def _send_push_notification(self, notification: Notification) -> bool:
        """Send a push notification using ntfy.sh"""
        try:
            # This uses the free ntfy.sh service
            topic = notification.recipient.replace("@", "_")  # Create a topic from email
            url = f"{self.ntfy_api}/{topic}"
            
            payload = {
                "topic": topic,
                "message": notification.message,
                "title": notification.subject,
                "priority": notification.priority
            }
            
            response = requests.post(url, json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"Error sending push notification: {str(e)}")
            return False
    
    def send_task_assignment_notification(self, user_id: str, task: Dict) -> bool:
        """Send a notification for task assignment"""
        notification = Notification(
            recipient=user_id,
            subject=f"New Task Assigned: {task['title']}",
            message=f"""
            You have been assigned a new task:
            
            Title: {task['title']}
            Description: {task['description']}
            Priority: {task['priority']}
            Estimated Duration: {task['estimated_duration']}
            
            Please review and acknowledge this task.
            """,
            priority=3,
            notification_type="email"  # Default to email
        )
        
        return self.send_notification(notification)
    
    def send_deadline_reminder(self, user_id: str, task: Dict, days_until_deadline: int) -> bool:
        """Send a deadline reminder"""
        notification = Notification(
            recipient=user_id,
            subject=f"Deadline Reminder: {task['title']}",
            message=f"""
            Reminder: Task "{task['title']}" is due in {days_until_deadline} days.
            
            Current Status: {task.get('status', 'In Progress')}
            Priority: {task['priority']}
            
            Please ensure you're on track to meet the deadline.
            """,
            priority=4 if days_until_deadline <= 2 else 3,
            notification_type="push"  # Use push notification for reminders
        )
        
        return self.send_notification(notification)
    
    def send_status_update(self, user_id: str, task: Dict, update: str) -> bool:
        """Send a status update notification"""
        notification = Notification(
            recipient=user_id,
            subject=f"Status Update: {task['title']}",
            message=f"""
            Status Update for Task "{task['title']}":
            
            {update}
            
            Current Progress: {task.get('progress', '0%')}
            Next Steps: {task.get('next_steps', 'None specified')}
            """,
            priority=2,
            notification_type="slack"  # Use Slack for status updates
        )
        
        return self.send_notification(notification)
    
    def send_team_coordination_message(self, team_id: str, message: str, priority: int = 2) -> bool:
        """Send a team coordination message"""
        notification = Notification(
            recipient=team_id,
            subject="Team Coordination Update",
            message=message,
            priority=priority,
            notification_type="discord"  # Use Discord for team communication
        )
        
        return self.send_notification(notification)

# Example usage
if __name__ == "__main__":
    # Test communication manager
    manager = CommunicationManager()
    
    # Test task assignment notification
    sample_task = {
        "title": "Implement User Authentication",
        "description": "Create a secure login system with JWT tokens",
        "priority": 4,
        "estimated_duration": timedelta(hours=8)
    }
    
    print("Sending task assignment notification...")
    success = manager.send_task_assignment_notification("user@example.com", sample_task)
    print(f"Notification sent: {success}")
    
    # Test deadline reminder
    print("\nSending deadline reminder...")
    success = manager.send_deadline_reminder("user@example.com", sample_task, 2)
    print(f"Reminder sent: {success}")
    
    # Test status update
    print("\nSending status update...")
    success = manager.send_status_update(
        "user@example.com",
        sample_task,
        "Completed JWT token implementation, working on frontend integration"
    )
    print(f"Status update sent: {success}")
    
    # Test team coordination
    print("\nSending team coordination message...")
    success = manager.send_team_coordination_message(
        "team-123",
        "Team meeting scheduled for tomorrow at 10 AM to discuss sprint progress"
    )
    print(f"Team message sent: {success}") 