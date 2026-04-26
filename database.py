from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./constructai.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- MODELS ---

# --- added on april 18th ---------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    email = Column(String, unique=True, index=True)
    password = Column(String)
    role = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
# --------------------------------

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id = Column(Integer, primary_key=True, index=True)
    role_user = Column(String)        # client, architect, etc.
    sender = Column(String)           # user or assistant
    message = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    filepath = Column(String)
    uploaded_by = Column(String)      # which role uploaded
    description = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

class WorkflowRequest(Base):
    __tablename__ = "workflow_requests"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    description = Column(Text)
    request_type = Column(String)
    submitted_by = Column(String)
    current_stage = Column(String)
    status = Column(String, default="pending")
    ai_summary = Column(Text)
    architect_comment = Column(Text)
    engineer_comment = Column(Text)
    contractor_comment = Column(Text)
    pm_comment = Column(Text)
    architect_status = Column(String, default="pending")
    engineer_status = Column(String, default="pending")
    contractor_status = Column(String, default="pending")
    pm_status = Column(String, default="pending")
    architect_attachment = Column(String)
    engineer_attachment = Column(String)
    contractor_attachment = Column(String)
    pm_attachment = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChecklistTemplate(Base):
    __tablename__ = "checklist_templates"
    id = Column(Integer, primary_key=True, index=True)
    work_type = Column(String)
    title = Column(String)
    questions = Column(Text)
    header_fields = Column(Text, default="[]")
    sections_meta = Column(Text, default="[]")
    created_by = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChecklistAssignment(Base):
    __tablename__ = "checklist_assignments"
    id = Column(Integer, primary_key=True, index=True)
    template_id = Column(Integer)
    assigned_to = Column(String)      # contractor, engineer, etc.
    assigned_by = Column(String)      # admin
    project_name = Column(String)
    due_date = Column(String)
    status = Column(String, default="pending")  # pending, submitted
    timestamp = Column(DateTime, default=datetime.utcnow)

class ChecklistResponse(Base):
    __tablename__ = "checklist_responses"
    id = Column(Integer, primary_key=True, index=True)
    assignment_id = Column(Integer)
    submitted_by = Column(String)
    responses = Column(Text)          # JSON string of answers
    ai_summary = Column(Text)
    issues_found = Column(Integer, default=0)
    image_paths = Column(Text, default="[]")
    timestamp = Column(DateTime, default=datetime.utcnow)

def create_tables():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# -----------------------------------------------------------------------------------------

# from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime

# DATABASE_URL = "sqlite:///./constructai.db"

# engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # --- MODELS ---

# class ChatMessage(Base):
#     __tablename__ = "chat_messages"
#     id = Column(Integer, primary_key=True, index=True)
#     role_user = Column(String)        # client, architect, etc.
#     sender = Column(String)           # user or assistant
#     message = Column(Text)
#     timestamp = Column(DateTime, default=datetime.utcnow)


# class Document(Base):
#     __tablename__ = "documents"
#     id = Column(Integer, primary_key=True, index=True)
#     filename = Column(String)
#     filepath = Column(String)
#     uploaded_by = Column(String)      # which role uploaded
#     description = Column(Text)
#     timestamp = Column(DateTime, default=datetime.utcnow)


# class WorkflowRequest(Base):
#     __tablename__ = "workflow_requests"
#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String)
#     description = Column(Text)
#     request_type = Column(String)
#     submitted_by = Column(String)
#     current_stage = Column(String)
#     status = Column(String, default="pending")
#     ai_summary = Column(Text)
#     architect_comment = Column(Text)
#     engineer_comment = Column(Text)
#     contractor_comment = Column(Text)
#     pm_comment = Column(Text)
#     architect_status = Column(String, default="pending")
#     engineer_status = Column(String, default="pending")
#     contractor_status = Column(String, default="pending")
#     pm_status = Column(String, default="pending")
#     architect_attachment = Column(String)
#     engineer_attachment = Column(String)
#     contractor_attachment = Column(String)
#     pm_attachment = Column(String)
#     timestamp = Column(DateTime, default=datetime.utcnow)


# class ChecklistTemplate(Base):
#     __tablename__ = "checklist_templates"
#     id = Column(Integer, primary_key=True, index=True)
#     work_type = Column(String)              # Brickwork, Concrete, Plaster, Tiles, Steel
#     title = Column(String)
#     description = Column(Text, default="")  # NEW
#     questions = Column(Text)                # JSON string of questions
#     created_by = Column(String)
#     source_pdf = Column(String, default="") # NEW
#     timestamp = Column(DateTime, default=datetime.utcnow)


# class ChecklistAssignment(Base):
#     __tablename__ = "checklist_assignments"
#     id = Column(Integer, primary_key=True, index=True)
#     template_id = Column(Integer)
#     assigned_to = Column(String)            # contractor, engineer, etc.
#     assigned_by = Column(String)            # admin
#     project_name = Column(String)
#     due_date = Column(String)
#     status = Column(String, default="pending")  # pending, submitted
#     timestamp = Column(DateTime, default=datetime.utcnow)


# class ChecklistResponse(Base):
#     __tablename__ = "checklist_responses"
#     id = Column(Integer, primary_key=True, index=True)
#     assignment_id = Column(Integer)
#     submitted_by = Column(String)
#     responses = Column(Text)                # JSON string of answers
#     ai_summary = Column(Text)
#     issues_found = Column(Integer, default=0)
#     timestamp = Column(DateTime, default=datetime.utcnow)


# def create_tables():
#     Base.metadata.create_all(bind=engine)


# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
