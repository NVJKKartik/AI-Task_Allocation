from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from .models import Base

# Create database engine
engine = create_engine('sqlite:///agent_task.db', echo=True)

# Create all tables
Base.metadata.create_all(engine)

# Create session factory
session_factory = sessionmaker(bind=engine)
Session = scoped_session(session_factory)

def get_db():
    """Get database session"""
    db = Session()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database"""
    Base.metadata.create_all(engine) 