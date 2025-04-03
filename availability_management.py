"""
Availability Management System for AI Task Allocation Agent

This module implements the availability management system that tracks and updates
user availability in real-time for the AI task allocation agent.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta, date
import calendar
from dateutil import rrule
from dateutil.parser import parse
from dateutil.rrule import rrulestr
import pytz
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TimeSlot:
    """Represents a time slot with start and end times"""
    def __init__(self, start_time: datetime, end_time: datetime, availability_type: str = "AVAILABLE"):
        """
        Initialize a time slot
        
        Args:
            start_time: Start time of the slot
            end_time: End time of the slot
            availability_type: Type of availability (AVAILABLE, BUSY, OUT_OF_OFFICE)
        """
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)
            
        self.start_time = start_time
        self.end_time = end_time
        self.availability_type = availability_type
    
    @property
    def duration(self) -> timedelta:
        """Get the duration of the time slot"""
        return self.end_time - self.start_time
    
    def overlaps(self, other: 'TimeSlot') -> bool:
        """Check if this time slot overlaps with another"""
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)
    
    def contains(self, time: datetime) -> bool:
        """Check if this time slot contains a specific time"""
        # Ensure timezone awareness
        if time.tzinfo is None:
            time = time.replace(tzinfo=pytz.UTC)
        return self.start_time <= time < self.end_time
    
    def to_dict(self) -> Dict:
        """Convert time slot to dictionary"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "availability_type": self.availability_type,
            "duration_minutes": self.duration.total_seconds() / 60
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TimeSlot':
        """Create a time slot from dictionary"""
        return cls(
            start_time=parse(data["start_time"]),
            end_time=parse(data["end_time"]),
            availability_type=data.get("availability_type", "AVAILABLE")
        )
    
    def __str__(self) -> str:
        return f"{self.start_time.strftime('%Y-%m-%d %H:%M')} - {self.end_time.strftime('%H:%M')} ({self.availability_type})"


class RecurringTimeSlot:
    """Represents a recurring time slot with a recurrence rule"""
    def __init__(self, 
                 start_time: datetime, 
                 end_time: datetime, 
                 recurrence_rule: str,
                 availability_type: str = "AVAILABLE"):
        """
        Initialize a recurring time slot
        
        Args:
            start_time: Start time of the first occurrence
            end_time: End time of the first occurrence
            recurrence_rule: iCalendar RRULE string (e.g., "FREQ=WEEKLY;BYDAY=MO,WE,FR")
            availability_type: Type of availability (AVAILABLE, BUSY, OUT_OF_OFFICE)
        """
        if start_time >= end_time:
            raise ValueError("Start time must be before end time")
        
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
        if end_time.tzinfo is None:
            end_time = end_time.replace(tzinfo=pytz.UTC)
            
        self.start_time = start_time
        self.end_time = end_time
        self.recurrence_rule = recurrence_rule
        self.availability_type = availability_type
        
        # Validate the recurrence rule
        try:
            # Create a naive datetime for rrule (dateutil has issues with timezone-aware datetimes)
            naive_start = start_time.replace(tzinfo=None)
            self.rrule_obj = rrulestr(f"DTSTART:{naive_start.strftime('%Y%m%dT%H%M%S')}\nRRULE:{recurrence_rule}")
        except Exception as e:
            raise ValueError(f"Invalid recurrence rule: {str(e)}")
    
    def get_occurrences(self, start_date: datetime, end_date: datetime) -> List[TimeSlot]:
        """
        Get all occurrences of this recurring time slot within a date range
        
        Args:
            start_date: Start date of the range
            end_date: End date of the range
            
        Returns:
            List of TimeSlot objects representing occurrences
        """
        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=pytz.UTC)
            
        # Convert to naive datetimes for rrule
        naive_start = start_date.replace(tzinfo=None)
        naive_end = end_date.replace(tzinfo=None)
            
        # Get all start times within the range
        naive_start_times = list(self.rrule_obj.between(naive_start, naive_end, inc=True))
        
        # Create time slots for each occurrence
        time_slots = []
        for naive_start in naive_start_times:
            # Add timezone info back
            aware_start = naive_start.replace(tzinfo=pytz.UTC)
            
            # Calculate the end time by adding the duration of the original slot
            duration = self.end_time - self.start_time
            aware_end = aware_start + duration
            
            # Create a time slot for this occurrence
            time_slots.append(TimeSlot(
                start_time=aware_start,
                end_time=aware_end,
                availability_type=self.availability_type
            ))
        
        return time_slots
    
    def to_dict(self) -> Dict:
        """Convert recurring time slot to dictionary"""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "recurrence_rule": self.recurrence_rule,
            "availability_type": self.availability_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RecurringTimeSlot':
        """Create a recurring time slot from dictionary"""
        return cls(
            start_time=parse(data["start_time"]),
            end_time=parse(data["end_time"]),
            recurrence_rule=data["recurrence_rule"],
            availability_type=data.get("availability_type", "AVAILABLE")
        )


class UserAvailability:
    """Manages availability for a single user"""
    def __init__(self, user_id: str):
        """
        Initialize user availability
        
        Args:
            user_id: Unique identifier for the user
        """
        self.user_id = user_id
        self.time_slots: List[TimeSlot] = []
        self.recurring_slots: List[RecurringTimeSlot] = []
    
    def add_time_slot(self, time_slot: TimeSlot) -> None:
        """
        Add a time slot to the user's availability
        
        Args:
            time_slot: TimeSlot to add
        """
        self.time_slots.append(time_slot)
    
    def add_recurring_slot(self, recurring_slot: RecurringTimeSlot) -> None:
        """
        Add a recurring time slot to the user's availability
        
        Args:
            recurring_slot: RecurringTimeSlot to add
        """
        self.recurring_slots.append(recurring_slot)
    
    def remove_time_slot(self, slot_index: int) -> None:
        """
        Remove a time slot by index
        
        Args:
            slot_index: Index of the time slot to remove
        """
        if 0 <= slot_index < len(self.time_slots):
            del self.time_slots[slot_index]
        else:
            raise IndexError("Time slot index out of range")
    
    def remove_recurring_slot(self, slot_index: int) -> None:
        """
        Remove a recurring time slot by index
        
        Args:
            slot_index: Index of the recurring time slot to remove
        """
        if 0 <= slot_index < len(self.recurring_slots):
            del self.recurring_slots[slot_index]
        else:
            raise IndexError("Recurring time slot index out of range")
    
    def get_available_slots(self, start_date: datetime, end_date: datetime) -> List[TimeSlot]:
        """
        Get all available time slots within a date range
        
        Args:
            start_date: Start date of the range
            end_date: End date of the range
            
        Returns:
            List of TimeSlot objects representing available times
        """
        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=pytz.UTC)
            
        # Get regular time slots within the range
        regular_slots = [
            slot for slot in self.time_slots 
            if slot.end_time > start_date and slot.start_time < end_date
            and slot.availability_type == "AVAILABLE"
        ]
        
        # Get recurring time slots within the range
        recurring_slots = []
        for recurring_slot in self.recurring_slots:
            if recurring_slot.availability_type == "AVAILABLE":
                recurring_slots.extend(recurring_slot.get_occurrences(start_date, end_date))
        
        # Combine and sort all available slots
        all_slots = regular_slots + recurring_slots
        all_slots.sort(key=lambda x: x.start_time)
        
        return all_slots
    
    def get_busy_slots(self, start_date: datetime, end_date: datetime) -> List[TimeSlot]:
        """
        Get all busy time slots within a date range
        
        Args:
            start_date: Start date of the range
            end_date: End date of the range
            
        Returns:
            List of TimeSlot objects representing busy times
        """
        # Ensure timezone awareness
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=pytz.UTC)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=pytz.UTC)
            
        # Get regular time slots within the range
        regular_slots = [
            slot for slot in self.time_slots 
            if slot.end_time > start_date and slot.start_time < end_date
            and slot.availability_type != "AVAILABLE"
        ]
        
        # Get recurring time slots within the range
        recurring_slots = []
        for recurring_slot in self.recurring_slots:
            if recurring_slot.availability_type != "AVAILABLE":
                recurring_slots.extend(recurring_slot.get_occurrences(start_date, end_date))
        
        # Combine and sort all busy slots
        all_slots = regular_slots + recurring_slots
        all_slots.sort(key=lambda x: x.start_time)
        
        return all_slots
    
    def is_available_for_duration(self, start_time: datetime, duration: timedelta) -> bool:
        """
        Check if the user is available for a specific duration starting at a given time
        
        Args:
            start_time: Start time to check
            duration: Duration needed
            
        Returns:
            True if the user is available, False otherwise
        """
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
            
        end_time = start_time + duration
        
        # Check if there are any busy slots that overlap with the requested time
        busy_slots = self.get_busy_slots(start_time, end_time)
        for slot in busy_slots:
            if slot.overlaps(TimeSlot(start_time, end_time, "TEMP")):
                return False
        
        # Check if there are available slots that cover the entire duration
        available_slots = self.get_available_slots(start_time, end_time)
        
        # If no available slots are defined, assume available by default
        if not available_slots and not self.recurring_slots:
            return True
        
        # Check if any available slot contains the entire duration
        for slot in available_slots:
            if slot.start_time <= start_time and slot.end_time >= end_time:
                return True
        
        return False
    
    def find_next_available_slot(self, 
                                start_time: datetime, 
                                duration: timedelta, 
                                max_days_ahead: int = 14) -> Optional[datetime]:
        """
        Find the next available time slot of the specified duration
        
        Args:
            start_time: Earliest start time to consider
            duration: Duration needed
            max_days_ahead: Maximum number of days to look ahead
            
        Returns:
            Start time of the next available slot, or None if none found
        """
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
            
        end_date = start_time + timedelta(days=max_days_ahead)
        
        # Get all available and busy slots in the date range
        available_slots = self.get_available_slots(start_time, end_date)
        busy_slots = self.get_busy_slots(start_time, end_date)
        
        # If no available slots are defined and no busy slots, assume available from start_time
        if not available_slots and not busy_slots and not self.recurring_slots:
            return start_time
        
        # Check each available slot
        for slot in available_slots:
            # Skip slots that start before our start_time
            actual_start = max(slot.start_time, start_time)
            
            # Check if the slot is long enough
            if actual_start + duration <= slot.end_time:
                # Check if this slot overlaps with any busy slot
                is_valid = True
                for busy_slot in busy_slots:
                    if busy_slot.overlaps(TimeSlot(actual_start, actual_start + duration, "TEMP")):
                        is_valid = False
                        break
                
                if is_valid:
                    return actual_start
        
        return None
    
    def to_dict(self) -> Dict:
        """Convert user availability to dictionary"""
        return {
            "user_id": self.user_id,
            "time_slots": [slot.to_dict() for slot in self.time_slots],
            "recurring_slots": [slot.to_dict() for slot in self.recurring_slots]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'UserAvailability':
        """Create user availability from dictionary"""
        availability = cls(data["user_id"])
        
        for slot_data in data.get("time_slots", []):
            availability.add_time_slot(TimeSlot.from_dict(slot_data))
        
        for slot_data in data.get("recurring_slots", []):
            availability.add_recurring_slot(RecurringTimeSlot.from_dict(slot_data))
        
        return availability


class AvailabilityManager:
    """Manages availability for multiple users"""
    def __init__(self):
        """Initialize the availability manager"""
        self.users: Dict[str, UserAvailability] = {}
    
    def add_user(self, user_id: str) -> None:
        """
        Add a new user to the availability manager
        
        Args:
            user_id: Unique identifier for the user
        """
        if user_id not in self.users:
            self.users[user_id] = UserAvailability(user_id)
    
    def get_user_availability(self, user_id: str) -> UserAvailability:
        """
        Get availability for a specific user
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            UserAvailability object for the user
        """
        if user_id not in self.users:
            self.add_user(user_id)
        
        return self.users[user_id]
    
    def add_time_slot(self, user_id: str, time_slot: TimeSlot) -> None:
        """
        Add a time slot for a user
        
        Args:
            user_id: Unique identifier for the user
            time_slot: TimeSlot to add
        """
        if user_id not in self.users:
            self.add_user(user_id)
        
        self.users[user_id].add_time_slot(time_slot)
    
    def add_recurring_slot(self, user_id: str, recurring_slot: RecurringTimeSlot) -> None:
        """
        Add a recurring time slot for a user
        
        Args:
            user_id: Unique identifier for the user
            recurring_slot: RecurringTimeSlot to add
        """
        if user_id not in self.users:
            self.add_user(user_id)
        
        self.users[user_id].add_recurring_slot(recurring_slot)
    
    def is_user_available(self, user_id: str, start_time: datetime, duration: timedelta) -> bool:
        """
        Check if a user is available for a specific duration
        
        Args:
            user_id: Unique identifier for the user
            start_time: Start time to check
            duration: Duration needed
            
        Returns:
            True if the user is available, False otherwise
        """
        if user_id not in self.users:
            # If user doesn't exist, assume they're available
            return True
        
        return self.users[user_id].is_available_for_duration(start_time, duration)
    
    def find_common_availability(self, 
                               user_ids: List[str], 
                               start_time: datetime,
                               duration: timedelta,
                               max_days_ahead: int = 14) -> Optional[datetime]:
        """
        Find the next time slot when all specified users are available
        
        Args:
            user_ids: List of user IDs to check
            start_time: Earliest start time to consider
            duration: Duration needed
            max_days_ahead: Maximum number of days to look ahead
            
        Returns:
            Start time of the next common available slot, or None if none found
        """
        # Ensure timezone awareness
        if start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=pytz.UTC)
            
        end_date = start_time + timedelta(days=max_days_ahead)
        current_time = start_time
        
        while current_time < end_date:
            all_available = True
            
            for user_id in user_ids:
                if user_id in self.users and not self.users[user_id].is_available_for_duration(current_time, duration):
                    all_available = False
                    
                    # Find the next available time for this user
                    next_available = self.users[user_id].find_next_available_slot(
                        current_time, duration, max_days_ahead
                    )
                    
                    if next_available:
                        # Move current_time to the next available time for this user
                        current_time = next_available
                    else:
                        # No available time found for this user within the range
                        return None
                    
                    break
            
            if all_available:
                return current_time
        
        return None
    
    def get_calendar_view(self, user_id: str, year: int, month: int) -> Dict:
        """
        Generate a calendar view of availability for a specific month
        
        Args:
            user_id: Unique identifier for the user
            year: Year to generate calendar for
            month: Month to generate calendar for (1-12)
            
        Returns:
            Dictionary with calendar data
        """
        # Create start and end dates for the month
        start_date = datetime(year, month, 1, tzinfo=pytz.UTC)
        
        # Get the last day of the month
        _, last_day = calendar.monthrange(year, month)
        end_date = datetime(year, month, last_day, 23, 59, 59, tzinfo=pytz.UTC)
        
        # Get available and busy slots for the month
        if user_id in self.users:
            available_slots = self.users[user_id].get_available_slots(start_date, end_date)
            busy_slots = self.users[user_id].get_busy_slots(start_date, end_date)
        else:
            available_slots = []
            busy_slots = []
        
        # Create calendar data
        calendar_data = {
            "user_id": user_id,
            "year": year,
            "month": month,
            "days": {}
        }
        
        # Populate days with availability information
        for day in range(1, last_day + 1):
            day_start = datetime(year, month, day, tzinfo=pytz.UTC)
            day_end = datetime(year, month, day, 23, 59, 59, tzinfo=pytz.UTC)
            
            # Find slots for this day
            day_available_slots = [
                slot for slot in available_slots 
                if slot.start_time.date() == day_start.date()
            ]
            
            day_busy_slots = [
                slot for slot in busy_slots 
                if slot.start_time.date() == day_start.date()
            ]
            
            # Calculate availability percentage for the day (based on working hours 9-5)
            working_minutes = 8 * 60  # 8 hours = 480 minutes
            available_minutes = sum([
                min((slot.end_time - slot.start_time).total_seconds() / 60, working_minutes)
                for slot in day_available_slots
            ])
            
            busy_minutes = sum([
                min((slot.end_time - slot.start_time).total_seconds() / 60, working_minutes)
                for slot in day_busy_slots
            ])
            
            # Calculate availability percentage
            if working_minutes > 0:
                availability_percentage = max(0, min(100, (working_minutes - busy_minutes) / working_minutes * 100))
            else:
                availability_percentage = 0
            
            # Add day data to calendar
            calendar_data["days"][str(day)] = {
                "available_slots": [slot.to_dict() for slot in day_available_slots],
                "busy_slots": [slot.to_dict() for slot in day_busy_slots],
                "availability_percentage": availability_percentage
            }
        
        return calendar_data
    
    def to_dict(self) -> Dict:
        """Convert availability manager to dictionary"""
        return {
            "users": {
                user_id: user_avail.to_dict() 
                for user_id, user_avail in self.users.items()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AvailabilityManager':
        """Create availability manager from dictionary"""
        manager = cls()
        
        for user_id, user_data in data.get("users", {}).items():
            manager.users[user_id] = UserAvailability.from_dict(user_data)
        
        return manager
    
    def save_to_file(self, filename: str) -> None:
        """
        Save availability data to a JSON file
        
        Args:
            filename: Path to the file to save to
        """
        data = self.to_dict()
        
        # Convert datetime objects to ISO format strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w') as f:
            json.dump(data, f, default=convert_datetime, indent=2)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'AvailabilityManager':
        """
        Load availability data from a JSON file
        
        Args:
            filename: Path to the file to load from
            
        Returns:
            AvailabilityManager object
        """
        with open(filename, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)


# Example usage
if __name__ == "__main__":
    # Create an availability manager
    manager = AvailabilityManager()
    
    # Add some users
    manager.add_user("user-123")  # John Doe
    manager.add_user("user-124")  # Jane Smith
    manager.add_user("user-125")  # Alex Johnson
    
    # Set up availability for John Doe
    john_avail = manager.get_user_availability("user-123")
    
    # Regular working hours (9-5) for specific days
    john_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 3, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 3, 17, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    john_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 4, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 4, 17, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    # Add a meeting (busy slot)
    john_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 3, 14, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 3, 15, 30, tzinfo=pytz.UTC),
        availability_type="BUSY"
    ))
    
    # Add recurring weekly working hours
    john_avail.add_recurring_slot(RecurringTimeSlot(
        start_time=datetime(2025, 4, 7, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 7, 17, 0, tzinfo=pytz.UTC),
        recurrence_rule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        availability_type="AVAILABLE"
    ))
    
    # Set up availability for Jane Smith
    jane_avail = manager.get_user_availability("user-124")
    
    # Regular working hours for specific days
    jane_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 3, 13, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 3, 18, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    jane_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 4, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 4, 18, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    # Add recurring weekly working hours
    jane_avail.add_recurring_slot(RecurringTimeSlot(
        start_time=datetime(2025, 4, 7, 10, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 7, 18, 0, tzinfo=pytz.UTC),
        recurrence_rule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        availability_type="AVAILABLE"
    ))
    
    # Set up availability for Alex Johnson
    alex_avail = manager.get_user_availability("user-125")
    
    # Regular working hours for specific days
    alex_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 3, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 3, 15, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    alex_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 4, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 4, 15, 0, tzinfo=pytz.UTC),
        availability_type="AVAILABLE"
    ))
    
    # Add recurring weekly working hours
    alex_avail.add_recurring_slot(RecurringTimeSlot(
        start_time=datetime(2025, 4, 7, 9, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 7, 15, 0, tzinfo=pytz.UTC),
        recurrence_rule="FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
        availability_type="AVAILABLE"
    ))
    
    # Add out of office time
    alex_avail.add_time_slot(TimeSlot(
        start_time=datetime(2025, 4, 10, 0, 0, tzinfo=pytz.UTC),
        end_time=datetime(2025, 4, 14, 0, 0, tzinfo=pytz.UTC),
        availability_type="OUT_OF_OFFICE"
    ))
    
    # Test availability checks
    print("Availability Tests:")
    
    # Check if John is available for a 2-hour meeting on April 3rd at 10 AM
    is_available = manager.is_user_available(
        "user-123",
        datetime(2025, 4, 3, 10, 0, tzinfo=pytz.UTC),
        timedelta(hours=2)
    )
    print(f"John available on April 3rd at 10 AM for 2 hours: {is_available}")
    
    # Check if John is available during his meeting
    is_available = manager.is_user_available(
        "user-123",
        datetime(2025, 4, 3, 14, 0, tzinfo=pytz.UTC),
        timedelta(hours=1)
    )
    print(f"John available on April 3rd at 2 PM for 1 hour: {is_available}")
    
    # Find common availability for all users
    common_time = manager.find_common_availability(
        ["user-123", "user-124", "user-125"],
        datetime(2025, 4, 3, 9, 0, tzinfo=pytz.UTC),
        timedelta(hours=2)
    )
    
    if common_time:
        print(f"Common availability found: {common_time.strftime('%Y-%m-%d %H:%M')}")
    else:
        print("No common availability found")
    
    # Generate calendar view for John in April 2025
    calendar_view = manager.get_calendar_view("user-123", 2025, 4)
    print("\nCalendar View for John (April 2025):")
    for day, day_data in calendar_view["days"].items():
        print(f"Day {day}: {day_data['availability_percentage']:.1f}% available")
    
    # Save availability data to file
    manager.save_to_file("availability_data.json")
    print("\nAvailability data saved to file")
    
    # Load availability data from file
    loaded_manager = AvailabilityManager.load_from_file("availability_data.json")
    print("Availability data loaded from file")
    
    # Verify loaded data
    is_available = loaded_manager.is_user_available(
        "user-123",
        datetime(2025, 4, 3, 10, 0, tzinfo=pytz.UTC),
        timedelta(hours=2)
    )
    print(f"Loaded data - John available on April 3rd at 10 AM: {is_available}")
