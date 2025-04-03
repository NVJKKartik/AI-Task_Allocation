"""
Smart Availability Management Module for AI Task Allocation Agent

This module implements intelligent availability management features including
calendar integration, time zone handling, and optimal working hour suggestions.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pytz
import requests
from dateutil import parser
from dateutil.rrule import rrule, WEEKLY, MO, TU, WE, TH, FR
import numpy as np
from dataclasses import dataclass

@dataclass
class WorkingHours:
    """Represents a user's working hours"""
    start_time: datetime
    end_time: datetime
    timezone: str
    work_days: List[int]  # 0=Monday, 6=Sunday

class SmartAvailabilityManager:
    """Manages smart availability features"""
    
    def __init__(self):
        """Initialize the smart availability manager"""
        self.setup_free_apis()
        self.working_hours_cache = {}
    
    def setup_free_apis(self):
        """Set up connections to free APIs"""
        # Free timezone API
        self.timezone_api = "http://worldtimeapi.org/api/timezone"
        
        # Free calendar API (using iCalendar format)
        self.calendar_api = "https://calendar.google.com/calendar/ical/{calendar_id}/public/basic.ics"
    
    def get_timezone_info(self, timezone: str) -> Dict:
        """Get timezone information from free API"""
        try:
            response = requests.get(f"{self.timezone_api}/{timezone}")
            if response.status_code == 200:
                return response.json()
            return {}
        except Exception as e:
            print(f"Error getting timezone info: {str(e)}")
            return {}
    
    def suggest_optimal_hours(self, user_id: str, historical_data: List[Dict]) -> WorkingHours:
        """Suggest optimal working hours based on historical data"""
        try:
            # Analyze historical completion times
            completion_times = []
            for record in historical_data:
                if "completion_time" in record:
                    completion_times.append(parser.parse(record["completion_time"]))
            
            if not completion_times:
                # Default to standard working hours
                return WorkingHours(
                    start_time=datetime.now(pytz.UTC).replace(hour=9, minute=0, second=0),
                    end_time=datetime.now(pytz.UTC).replace(hour=17, minute=0, second=0),
                    timezone="UTC",
                    work_days=[0, 1, 2, 3, 4]  # Monday to Friday
                )
            
            # Calculate most productive hours
            hours = [t.hour for t in completion_times]
            optimal_start = int(np.percentile(hours, 25))  # 25th percentile
            optimal_end = int(np.percentile(hours, 75))    # 75th percentile
            
            # Get most common work days
            days = [t.weekday() for t in completion_times]
            work_days = list(set(days))
            
            return WorkingHours(
                start_time=datetime.now(pytz.UTC).replace(hour=optimal_start, minute=0, second=0),
                end_time=datetime.now(pytz.UTC).replace(hour=optimal_end, minute=0, second=0),
                timezone="UTC",
                work_days=work_days
            )
        except Exception as e:
            print(f"Error suggesting optimal hours: {str(e)}")
            return self.get_default_working_hours()
    
    def get_default_working_hours(self) -> WorkingHours:
        """Get default working hours"""
        return WorkingHours(
            start_time=datetime.now(pytz.UTC).replace(hour=9, minute=0, second=0),
            end_time=datetime.now(pytz.UTC).replace(hour=17, minute=0, second=0),
            timezone="UTC",
            work_days=[0, 1, 2, 3, 4]  # Monday to Friday
        )
    
    def detect_calendar_conflicts(self, user_id: str, start_time: datetime, end_time: datetime) -> List[Dict]:
        """Detect calendar conflicts for a time slot"""
        try:
            # Get user's calendar events
            events = self._get_calendar_events(user_id)
            
            conflicts = []
            for event in events:
                event_start = parser.parse(event.get("DTSTART", ""))
                event_end = parser.parse(event.get("DTEND", ""))
                
                if (start_time < event_end and end_time > event_start):
                    conflicts.append({
                        "title": event.get("SUMMARY", "Unknown Event"),
                        "start_time": event_start,
                        "end_time": event_end
                    })
            
            return conflicts
        except Exception as e:
            print(f"Error detecting calendar conflicts: {str(e)}")
            return []
    
    def _get_calendar_events(self, user_id: str) -> List[Dict]:
        """Get calendar events for a user"""
        try:
            # This would use the user's calendar ID
            calendar_id = self._get_user_calendar_id(user_id)
            if not calendar_id:
                return []
            
            response = requests.get(self.calendar_api.format(calendar_id=calendar_id))
            if response.status_code == 200:
                return self._parse_icalendar(response.text)
            return []
        except Exception as e:
            print(f"Error getting calendar events: {str(e)}")
            return []
    
    def _get_user_calendar_id(self, user_id: str) -> Optional[str]:
        """Get a user's calendar ID"""
        # This would typically come from user settings or database
        return None
    
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
    
    def generate_availability_schedule(self, user_id: str, start_date: datetime, end_date: datetime) -> Dict:
        """Generate an availability schedule for a user"""
        try:
            # Get user's working hours
            working_hours = self.working_hours_cache.get(user_id, self.get_default_working_hours())
            
            # Generate recurring time slots
            slots = []
            for day in working_hours.work_days:
                for dt in rrule(WEEKLY, dtstart=start_date, until=end_date, byweekday=day):
                    slot_start = dt.replace(
                        hour=working_hours.start_time.hour,
                        minute=working_hours.start_time.minute
                    )
                    slot_end = dt.replace(
                        hour=working_hours.end_time.hour,
                        minute=working_hours.end_time.minute
                    )
                    
                    # Check for conflicts
                    conflicts = self.detect_calendar_conflicts(user_id, slot_start, slot_end)
                    
                    slots.append({
                        "start_time": slot_start,
                        "end_time": slot_end,
                        "timezone": working_hours.timezone,
                        "conflicts": conflicts
                    })
            
            return {
                "user_id": user_id,
                "working_hours": {
                    "start": working_hours.start_time.strftime("%H:%M"),
                    "end": working_hours.end_time.strftime("%H:%M"),
                    "timezone": working_hours.timezone,
                    "work_days": [self._get_day_name(d) for d in working_hours.work_days]
                },
                "slots": slots
            }
        except Exception as e:
            print(f"Error generating availability schedule: {str(e)}")
            return {}
    
    def _get_day_name(self, day_num: int) -> str:
        """Get day name from day number"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        return days[day_num]

# Example usage
if __name__ == "__main__":
    # Test smart availability manager
    manager = SmartAvailabilityManager()
    
    # Test timezone info
    timezone_info = manager.get_timezone_info("America/New_York")
    print("Timezone Info:")
    print(json.dumps(timezone_info, indent=2))
    
    # Test optimal hours suggestion
    historical_data = [
        {"completion_time": "2024-03-20T10:30:00Z"},
        {"completion_time": "2024-03-21T11:15:00Z"},
        {"completion_time": "2024-03-22T09:45:00Z"},
        {"completion_time": "2024-03-23T14:20:00Z"}
    ]
    
    optimal_hours = manager.suggest_optimal_hours("user-123", historical_data)
    print("\nOptimal Working Hours:")
    print(f"Start: {optimal_hours.start_time.strftime('%H:%M')}")
    print(f"End: {optimal_hours.end_time.strftime('%H:%M')}")
    print(f"Timezone: {optimal_hours.timezone}")
    print(f"Work Days: {[manager._get_day_name(d) for d in optimal_hours.work_days]}")
    
    # Test availability schedule
    start_date = datetime.now(pytz.UTC)
    end_date = start_date + timedelta(days=7)
    
    schedule = manager.generate_availability_schedule("user-123", start_date, end_date)
    print("\nAvailability Schedule:")
    print(json.dumps(schedule, indent=2, default=str)) 