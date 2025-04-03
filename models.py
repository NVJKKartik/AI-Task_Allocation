from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, Enum, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

Base = declarative_base()

class TaskStatus(enum.Enum):
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"

class AllocationStatus(enum.Enum):
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"

class Task(Base):
    __tablename__ = 'tasks'

    id = Column(String(36), primary_key=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    priority = Column(Integer, nullable=False)  # 1-5 scale
    deadline = Column(DateTime)
    estimated_duration = Column(Integer)  # in minutes
    status = Column(Enum(TaskStatus), nullable=False, default=TaskStatus.PENDING)
    created_by = Column(String(36), ForeignKey('users.id', ondelete='SET NULL'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    allocations = relationship("TaskAllocation", back_populates="task")
    skills = relationship("TaskSkill", back_populates="task")
    dependencies = relationship("TaskDependency", 
                              foreign_keys="[TaskDependency.task_id]",
                              back_populates="task")

class TaskAllocation(Base):
    __tablename__ = 'task_allocations'

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    confidence_score = Column(Float, nullable=False)  # AI confidence in match (0-1)
    allocation_reason = Column(Text)
    status = Column(Enum(AllocationStatus), nullable=False, default=AllocationStatus.PENDING)
    allocated_at = Column(DateTime, default=datetime.utcnow)
    accepted_at = Column(DateTime)
    completed_at = Column(DateTime)
    feedback_score = Column(Integer)  # User feedback on allocation quality (1-5)
    feedback_comments = Column(Text)

    # Relationships
    task = relationship("Task", back_populates="allocations")
    user = relationship("User", back_populates="allocations")

class TaskSkill(Base):
    __tablename__ = 'task_skills'

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False)
    skill_id = Column(String(36), ForeignKey('skills.id', ondelete='CASCADE'), nullable=False)
    minimum_proficiency = Column(Integer, nullable=False)  # 1-5 scale
    importance = Column(Integer, nullable=False)  # 1-5 scale
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("Task", back_populates="skills")
    skill = relationship("Skill")

class TaskDependency(Base):
    __tablename__ = 'task_dependencies'

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False)
    dependency_task_id = Column(String(36), ForeignKey('tasks.id', ondelete='CASCADE'), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    task = relationship("Task", foreign_keys=[task_id], back_populates="dependencies")
    dependency_task = relationship("Task", foreign_keys=[dependency_task_id]) 