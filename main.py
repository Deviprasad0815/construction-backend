from dotenv import load_dotenv
import os
load_dotenv()

from fastapi import FastAPI, UploadFile, File, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
import google.generativeai as genai
import aiofiles
import os
from datetime import datetime
# from database import create_tables, get_db, ChatMessage, Document, WorkflowRequest
from database import create_tables, get_db, ChatMessage, Document, WorkflowRequest, ChecklistTemplate, ChecklistAssignment, ChecklistResponse, SessionLocal, User
from fastapi.responses import FileResponse
import json
import base64

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from io import BytesIO
from fastapi.responses import StreamingResponse
import requests
from PIL import Image as PILImage

# ---- added on april 18th -----
from passlib.context import CryptContext
from jose import jwt, JWTError
from datetime import timedelta

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "constructai-iitg-2026-secret")
ALGORITHM = "HS256"

PREDEFINED_USERS = [
    {"name": "Admin", "email": "admin@constructai.com", "password": "admin123", "role": "admin"},
    {"name": "Client", "email": "client@constructai.com", "password": "demo123", "role": "client"},
    {"name": "Architect", "email": "architect@constructai.com", "password": "demo123", "role": "architect"},
    {"name": "Engineer", "email": "engineer@constructai.com", "password": "demo123", "role": "engineer"},
    {"name": "Contractor", "email": "contractor@constructai.com", "password": "demo123", "role": "contractor"},
    {"name": "Project Manager", "email": "pm@constructai.com", "password": "demo123", "role": "project_manager"},
]

# --- SETUP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# create_tables()

# ---------- added april 18th-----------
def seed_users():
    db = SessionLocal()
    try:
        for u in PREDEFINED_USERS:
            existing = db.query(User).filter(User.email == u["email"]).first()
            if not existing:
                user = User(
                    name=u["name"],
                    email=u["email"],
                    password=pwd_context.hash(u["password"]),
                    role=u["role"]
                )
                db.add(user)
        db.commit()
    finally:
        db.close()

create_tables()
seed_users()
# -------------------------------

# --- SCHEMAS ---
class ChatRequest(BaseModel):
    role: str
    messages: list
    system: str

class WorkflowCreate(BaseModel):
    title: str
    description: str
    submitted_by: str
    request_type: str = "general"

# --- CHAT ---
@app.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    try:
        # Fetch real documents from database
        docs = db.query(Document).order_by(Document.timestamp.desc()).all()
        doc_list = "\n".join([f"- {d.filename} (uploaded by {d.uploaded_by}, description: {d.description})" for d in docs]) or "No documents uploaded yet."

        last_msg = req.messages[-1]["content"]
        prompt = f"""{req.system}

REAL PROJECT DOCUMENTS CURRENTLY IN THE SYSTEM:
{doc_list}

Only reference documents listed above. Do not mention any other documents.

User question: {last_msg}"""

        response = model.generate_content(prompt)
        reply = response.text

        # Save to database
        db.add(ChatMessage(role_user=req.role, sender="user", message=last_msg))
        db.add(ChatMessage(role_user=req.role, sender="assistant", message=reply))
        db.commit()

        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Error: {str(e)}"}

# --- GET CHAT HISTORY ---
@app.get("/chat/history/{role}")
def get_history(role: str, db: Session = Depends(get_db)):
    messages = db.query(ChatMessage).filter(
        ChatMessage.role_user == role
    ).order_by(ChatMessage.timestamp).all()
    return [{"sender": m.sender, "message": m.message, "time": str(m.timestamp)} for m in messages]

@app.delete("/chat/clear/{role}")
def clear_history(role: str, db: Session = Depends(get_db)):
    db.query(ChatMessage).filter(ChatMessage.role_user == role).delete()
    db.commit()
    return {"message": "Chat history cleared"}

# --- UPLOAD DOCUMENT ---
@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    uploaded_by: str = Form(...),
    description: str = Form(""),
    db: Session = Depends(get_db)
):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    async with aiofiles.open(filepath, 'wb') as f:
        content = await file.read()
        await f.write(content)

    doc = Document(
        filename=file.filename,
        filepath=filepath,
        uploaded_by=uploaded_by,
        description=description
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return {"message": "Uploaded successfully", "id": doc.id, "filename": file.filename}

# --- GET ALL DOCUMENTS ---
@app.get("/documents")
def get_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.timestamp.desc()).all()
    return [{"id": d.id, "filename": d.filename, "uploaded_by": d.uploaded_by, "description": d.description, "time": str(d.timestamp)} for d in docs]

# --- DOWNLOAD DOCUMENT ---
@app.get("/documents/download/{doc_id}")
def download_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return {"error": "Document not found"}
    return FileResponse(
        path=doc.filepath,
        filename=doc.filename,
        media_type='application/octet-stream'
    )

# --- CHAT WITH DOCUMENT ---
@app.post("/documents/ask")
async def ask_document(
    document_id: int = Form(...),
    question: str = Form(...),
    role: str = Form(...),
    db: Session = Depends(get_db)
):
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        return {"reply": "Document not found."}
    
    try:
        async with aiofiles.open(doc.filepath, 'rb') as f:
            content = await f.read()
        
        prompt = f"""You are an AI assistant for a construction project.
A {role} is asking about the document '{doc.filename}'.
Document description: {doc.description}

Question: {question}

Note: Respond as if you have read the document and provide helpful construction-domain insights."""
        
        response = model.generate_content(prompt)
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"Error reading document: {str(e)}"}

# --- WORKFLOW ---

@app.post("/workflow/submit")
async def submit_workflow(req: WorkflowCreate, db: Session = Depends(get_db)):
    # AI generates summary
    try:
        prompt = f"""A {req.submitted_by} submitted a construction change request.
Title: {req.title}
Description: {req.description}
Type: {req.request_type}

Write a 2-3 sentence professional summary of this request for the project team."""
        response = model.generate_content(prompt)
        ai_summary = response.text
    except:
        ai_summary = req.description

    wf = WorkflowRequest(
        title=req.title,
        description=req.description,
        request_type=req.request_type if hasattr(req, 'request_type') else "general",
        submitted_by=req.submitted_by,
        current_stage="architect",
        status="in_progress",
        ai_summary=ai_summary
    )
    db.add(wf)
    db.commit()
    db.refresh(wf)
    return {"message": "Submitted", "id": wf.id, "ai_summary": ai_summary}

# @app.get("/workflow")
# def get_workflow(db: Session = Depends(get_db)):
#     items = db.query(WorkflowRequest).order_by(WorkflowRequest.timestamp.desc()).all()
#     return [{
#         "id": w.id, "title": w.title, "description": w.description,
#         "request_type": w.request_type, "submitted_by": w.submitted_by,
#         "current_stage": w.current_stage, "status": w.status,
#         "ai_summary": w.ai_summary,
#         "architect_comment": w.architect_comment, "architect_status": w.architect_status,
#         "engineer_comment": w.engineer_comment, "engineer_status": w.engineer_status,
#         "contractor_comment": w.contractor_comment, "contractor_status": w.contractor_status,
#         "pm_comment": w.pm_comment, "pm_status": w.pm_status,
#         "time": str(w.timestamp)
#     } for w in items]

@app.get("/workflow")
def get_workflow(db: Session = Depends(get_db)):
    items = db.query(WorkflowRequest).order_by(WorkflowRequest.timestamp.desc()).all()
    return [{
        "id": w.id, "title": w.title, "description": w.description,
        "request_type": w.request_type, "submitted_by": w.submitted_by,
        "current_stage": w.current_stage, "status": w.status,
        "ai_summary": w.ai_summary,
        "architect_comment": w.architect_comment, "architect_status": w.architect_status,
        "architect_attachment": w.architect_attachment,
        "engineer_comment": w.engineer_comment, "engineer_status": w.engineer_status,
        "engineer_attachment": w.engineer_attachment,
        "contractor_comment": w.contractor_comment, "contractor_status": w.contractor_status,
        "contractor_attachment": w.contractor_attachment,
        "pm_comment": w.pm_comment, "pm_status": w.pm_status,
        "pm_attachment": w.pm_attachment,
        "time": str(w.timestamp)
    } for w in items]

# @app.post("/workflow/{wf_id}/approve")
# async def approve_workflow(wf_id: int, role: str = Form(...), comment: str = Form(""), db: Session = Depends(get_db)):
#     wf = db.query(WorkflowRequest).filter(WorkflowRequest.id == wf_id).first()
#     if not wf:
#         return {"error": "Not found"}
#     stages = ["architect", "engineer", "contractor", "project_manager"]
#     setattr(wf, f"{role}_comment", comment)
#     setattr(wf, f"{role}_status", "approved")
#     current_index = stages.index(role) if role in stages else -1
#     if current_index < len(stages) - 1:
#         wf.current_stage = stages[current_index + 1]
#     else:
#         wf.current_stage = "completed"
#         wf.status = "approved"
#     db.commit()
#     return {"message": "Approved", "next_stage": wf.current_stage}

@app.post("/workflow/{wf_id}/approve")
async def approve_workflow(
    wf_id: int,
    role: str = Form(...),
    comment: str = Form(""),
    file: UploadFile = File(None),
    db: Session = Depends(get_db)
):
    wf = db.query(WorkflowRequest).filter(WorkflowRequest.id == wf_id).first()
    if not wf:
        return {"error": "Not found"}

    # Save attachment if provided
    if file and file.filename:
        filename = f"workflow_{wf_id}_{role}_{file.filename}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        setattr(wf, f"{role}_attachment", filename)

    stages = ["architect", "engineer", "contractor", "project_manager"]
    setattr(wf, f"{role}_comment", comment)
    setattr(wf, f"{role}_status", "approved")
    current_index = stages.index(role) if role in stages else -1
    if current_index < len(stages) - 1:
        wf.current_stage = stages[current_index + 1]
    else:
        wf.current_stage = "completed"
        wf.status = "approved"
    db.commit()
    return {"message": "Approved", "next_stage": wf.current_stage}

@app.get("/workflow/attachment/{filename}")
def download_workflow_attachment(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        return {"error": "File not found"}
    return FileResponse(
        path=filepath,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.post("/workflow/{wf_id}/reject")
async def reject_workflow(wf_id: int, role: str = Form(...), comment: str = Form(""), db: Session = Depends(get_db)):
    wf = db.query(WorkflowRequest).filter(WorkflowRequest.id == wf_id).first()
    if not wf:
        return {"error": "Not found"}
    setattr(wf, f"{role}_comment", comment)
    setattr(wf, f"{role}_status", "rejected")
    wf.current_stage = "rejected"
    wf.status = "rejected"
    db.commit()
    return {"message": "Rejected"}

# --- CHECKLIST ---

# Default questions per work type
CHECKLIST_QUESTIONS = {
    "Brickwork": [
        "Check for availability of approved drawings for brickwork?",
        "Check for material quality (bricks, mortar)?",
        "Check for proper alignment and plumb of walls?",
        "Check for mortar ratio as specified?",
        "Check for proper curing of brickwork?",
        "Check for openings (doors/windows) as per drawings?",
        "Check for scaffolding safety?",
    ],
    "Concrete": [
        "Check for availability of approved mix design?",
        "Check for reinforcement placement as per drawings?",
        "Check for shuttering/formwork is properly fixed?",
        "Check for cover blocks placed correctly?",
        "Check for concrete grade as specified?",
        "Check for proper vibration during pouring?",
        "Check for curing done properly?",
    ],
    "Plaster": [
        "Check for surface preparation before plastering?",
        "Check for approved mix ratio for plaster?",
        "Check for thickness of plaster as specified?",
        "Check for proper curing of plaster?",
        "Check for finishing is smooth and even?",
        "Check for cracks or defects in plaster?",
    ],
    "Tiles": [
        "Check for approved tile samples and specifications?",
        "Check for surface level before tiling?",
        "Check for approved adhesive/mortar used?",
        "Check for tile alignment and pattern as per drawing?",
        "Check for grouting done properly?",
        "Check for broken or damaged tiles?",
        "Check for proper cleaning after installation?",
    ],
    "Steel Fabrication": [
        "Check for availability of approved shop drawings for fabrication?",
        "Check for availability of structural steel sections and welding rods?",
        "Check for straightening of members, is it acceptable?",
        "Check for availability of authorised welders for welding operations?",
        "Check for cutting of the sections, is it as specified?",
        "Check for dimensions of assembly, is it OK?",
        "Check for joint preparation, is it OK?",
        "Check for weld quality and thickness, is it as specified?",
        "Check for hole punching, is it as specified?",
        "Check for surface preparation prior to painting?",
        "Check for lifting and transportation arrangements prior to erection?",
        "Check for completion of joint records for inspection?",
    ]
}

@app.post("/checklist/template")
async def create_template(
    work_type: str = Form(...),
    title: str = Form(...),
    created_by: str = Form(...),
    db: Session = Depends(get_db)
):
    questions = CHECKLIST_QUESTIONS.get(work_type, [])
    template = ChecklistTemplate(
        work_type=work_type,
        title=title,
        questions=json.dumps(questions),
        created_by=created_by
    )
    db.add(template)
    db.commit()
    db.refresh(template)
    return {"message": "Template created", "id": template.id, "questions": questions}

@app.get("/checklist/templates")
def get_templates(db: Session = Depends(get_db)):
    templates = db.query(ChecklistTemplate).order_by(ChecklistTemplate.timestamp.desc()).all()
    return [{"id": t.id, "work_type": t.work_type, "title": t.title,
             "questions": json.loads(t.questions), "created_by": t.created_by,
             "time": str(t.timestamp)} for t in templates]

@app.post("/checklist/assign")
async def assign_checklist(
    template_id: int = Form(...),
    assigned_to: str = Form(...),
    project_name: str = Form(...),
    due_date: str = Form(...),
    assigned_by: str = Form("admin"),
    db: Session = Depends(get_db)
):
    assignment = ChecklistAssignment(
        template_id=template_id,
        assigned_to=assigned_to,
        assigned_by=assigned_by,
        project_name=project_name,
        due_date=due_date,
        status="pending"
    )
    db.add(assignment)
    db.commit()
    db.refresh(assignment)
    return {"message": "Assigned successfully", "id": assignment.id}

# @app.get("/checklist/assignments/{role}")
# def get_assignments(role: str, db: Session = Depends(get_db)):
#     assignments = db.query(ChecklistAssignment).filter(
#         ChecklistAssignment.assigned_to == role
#     ).order_by(ChecklistAssignment.timestamp.desc()).all()
#     result = []
#     for a in assignments:
#         template = db.query(ChecklistTemplate).filter(
#             ChecklistTemplate.id == a.template_id
#         ).first()
#         if template:
#             result.append({
#                 "id": a.id,
#                 "template_id": a.template_id,
#                 "work_type": template.work_type,
#                 "title": template.title,
#                 "questions": json.loads(template.questions),
#                 "header_fields": json.loads(template.header_fields or "[]"),
#                 "sections_meta": json.loads(template.sections_meta or "[]"),
#                 "project_name": a.project_name,
#                 "due_date": a.due_date,
#                 "status": a.status
#             })
#     return result

@app.get("/checklist/assignments/{role}")
def get_assignments(role: str, db: Session = Depends(get_db)):
    assignments = db.query(ChecklistAssignment).filter(
        ChecklistAssignment.assigned_to == role
    ).order_by(ChecklistAssignment.timestamp.desc()).all()
    result = []
    for a in assignments:
        template = db.query(ChecklistTemplate).filter(
            ChecklistTemplate.id == a.template_id
        ).first()
        response = db.query(ChecklistResponse).filter(
            ChecklistResponse.assignment_id == a.id
        ).first()
        if template:
            result.append({
                "id": a.id,
                "template_id": a.template_id,
                "work_type": template.work_type,
                "title": template.title,
                "questions": json.loads(template.questions),
                "header_fields": json.loads(template.header_fields or "[]"),
                "sections_meta": json.loads(template.sections_meta or "[]"),
                "project_name": a.project_name,
                "due_date": a.due_date,
                "status": a.status,
                "response": {
                    "responses": response.responses,
                    "ai_summary": response.ai_summary,
                    "issues_found": response.issues_found,
                    "image_paths": json.loads(response.image_paths or "[]"),
                    "timestamp": str(response.timestamp)
                } if response else None
            })
    return result

@app.get("/checklist/assignments/all/list")
def get_all_assignments(db: Session = Depends(get_db)):
    assignments = db.query(ChecklistAssignment).order_by(
        ChecklistAssignment.timestamp.desc()
    ).all()
    result = []
    for a in assignments:
        template = db.query(ChecklistTemplate).filter(
            ChecklistTemplate.id == a.template_id
        ).first()
        response = db.query(ChecklistResponse).filter(
            ChecklistResponse.assignment_id == a.id
        ).first()
        result.append({
            "id": a.id,
            "work_type": template.work_type if template else "Unknown",
            "title": template.title if template else "Unknown",
            "assigned_to": a.assigned_to,
            "project_name": a.project_name,
            "due_date": a.due_date,
            "status": a.status,
            "header_fields": json.loads(template.header_fields or "[]") if template else [],
            "sections_meta": json.loads(template.sections_meta or "[]") if template else [],
            "response": {
                "responses": response.responses,
                "ai_summary": response.ai_summary,
                "issues_found": response.issues_found,
                "image_paths": json.loads(response.image_paths or "[]")
            } if response else None
        })
    return result

# @app.post("/checklist/submit")
# async def submit_checklist(
#     assignment_id: int = Form(...),
#     submitted_by: str = Form(...),
#     responses: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         responses_data = json.loads(responses)
        
#         # Handle both old format (list) and new format (dict with header + questions)
#         if isinstance(responses_data, dict):
#             question_responses = responses_data.get("questions", [])
#             header_responses = responses_data.get("header", [])
#         else:
#             question_responses = responses_data
#             header_responses = []

#         # Count issues (No answers)
#         issues = sum(1 for r in question_responses if r.get("answer") == "No")

#         # AI analysis
#         try:
#             issues_text = "\n".join([
#                 f"- [{r.get('section', '')}] {r['question']}: {r['answer']} {r.get('remark', '')}"
#                 for r in question_responses
#             ])
#             header_text = "\n".join([f"{h['label']}: {h['value']}" for h in header_responses])
            
#             prompt = f"""Construction quality checklist submitted.
# Project Info:
# {header_text}

# Inspection Results:
# {issues_text}

# Issues found: {issues}
# Write a professional 3-4 sentence inspection summary. Highlight any issues."""
#             response = model.generate_content(prompt)
#             ai_summary = response.text
#         except:
#             ai_summary = f"Checklist submitted with {issues} issues found."

#         checklist_response = ChecklistResponse(
#             assignment_id=assignment_id,
#             submitted_by=submitted_by,
#             responses=responses,
#             ai_summary=ai_summary,
#             issues_found=issues
#         )
#         db.add(checklist_response)
        
#         assignment = db.query(ChecklistAssignment).filter(
#             ChecklistAssignment.id == assignment_id
#         ).first()
#         if assignment:
#             assignment.status = "submitted"
#         db.commit()
        
#         return {"message": "Submitted", "issues_found": issues, "ai_summary": ai_summary}
#     except Exception as e:
#         return {"message": "Error", "issues_found": 0, "ai_summary": str(e)}

# ----------- added insted of above april 21st--------------------
@app.post("/checklist/submit")
async def submit_checklist(
    assignment_id: int = Form(...),
    submitted_by: str = Form(...),
    responses: str = Form(...),
    images: list[UploadFile] = File(default=[]),
    db: Session = Depends(get_db)
):
    try:
        responses_data = json.loads(responses)

        if isinstance(responses_data, dict):
            question_responses = responses_data.get("questions", [])
            header_responses = responses_data.get("header", [])
        else:
            question_responses = responses_data
            header_responses = []

        issues = sum(1 for r in question_responses if r.get("answer") == "No")

        # Save uploaded images
        saved_images = []
        for img in images:
            if img and img.filename:
                safe_name = f"checklist_{assignment_id}_{img.filename}"
                filepath = os.path.join(UPLOAD_DIR, safe_name)
                async with aiofiles.open(filepath, 'wb') as f:
                    content = await img.read()
                    await f.write(content)
                saved_images.append(safe_name)

        # AI analysis
        try:
            issues_text = "\n".join([
                f"- [{r.get('section', '')}] {r['question']}: {r['answer']} {r.get('remark', '')}"
                for r in question_responses
            ])
            header_text = "\n".join([f"{h['label']}: {h['value']}" for h in header_responses])
            prompt = f"""Construction quality checklist submitted.
Project Info:
{header_text}

Inspection Results:
{issues_text}

Issues found: {issues}
Write a professional 3-4 sentence inspection summary."""
            response = model.generate_content(prompt)
            ai_summary = response.text
        except:
            ai_summary = f"Checklist submitted with {issues} issues found."

        checklist_response = ChecklistResponse(
            assignment_id=assignment_id,
            submitted_by=submitted_by,
            responses=responses,
            ai_summary=ai_summary,
            issues_found=issues,
            image_paths=json.dumps(saved_images)
        )
        db.add(checklist_response)

        assignment = db.query(ChecklistAssignment).filter(
            ChecklistAssignment.id == assignment_id
        ).first()
        if assignment:
            assignment.status = "submitted"
        db.commit()

        return {
            "message": "Submitted",
            "issues_found": issues,
            "ai_summary": ai_summary,
            "images_saved": len(saved_images)
        }
    except Exception as e:
        return {"message": "Error", "issues_found": 0, "ai_summary": str(e)}
# ----------------------------------------------------------------

# import base64

# --- AI CHECKLIST EXTRACTION ---
# @app.post("/checklist/extract-from-pdf")
# async def extract_checklist_from_pdf(
#     file: UploadFile = File(...),
#     work_type: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     try:
#         content = await file.read()
#         pdf_base64 = base64.b64encode(content).decode('utf-8')
        
#         prompt = """You are an expert construction quality control AI.
# Analyze this construction checklist PDF and extract ALL checklist items.

# Return ONLY a valid JSON object in this exact format:
# {
#   "title": "extracted checklist title",
#   "description": "brief description of what this checklist covers",
#   "suggested_role": "contractor or engineer or architect or project_manager",
#   "questions": [
#     {
#       "id": 1,
#       "question": "full question text exactly as in document",
#       "type": "yes_no",
#       "category": "Safety or Quality or Compliance or General",
#       "critical": true or false
#     }
#   ]
# }

# Rules:
# - Extract EVERY question/check item from the document
# - Keep question text exactly as written
# - type is always "yes_no" for Yes/No questions
# - Mark critical=true for safety-critical items
# - Do not add any text outside the JSON"""

#         response = model.generate_content([
#             {
#                 "role": "user",
#                 "parts": [
#                     {
#                         "inline_data": {
#                             "mime_type": "application/pdf",
#                             "data": pdf_base64
#                         }
#                     },
#                     {"text": prompt}
#                 ]
#             }
#         ])
        
#         raw = response.text.strip()
#         raw = raw.replace("```json", "").replace("```", "").strip()
#         extracted = json.loads(raw)
#         return {"success": True, "data": extracted}
#     except Exception as e:
#         return {"success": False, "error": str(e)}

# --- AI CHECKLIST EXTRACTION ---

@app.post("/checklist/extract-from-pdf")
async def extract_checklist_from_pdf(
    file: UploadFile = File(...),
    work_type: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        content = await file.read()
        pdf_base64 = base64.b64encode(content).decode('utf-8')

        prompt = """You are an expert construction quality control AI.
Carefully read this entire PDF checklist document and extract its complete structure.

This PDF may be any type of construction checklist - do NOT assume any specific format.
Extract exactly what you find in the document.

Return ONLY a valid JSON object like this (no extra text before or after):
{
  "title": "exact title from the document",
  "description": "what this checklist is for in 1-2 sentences",
  "suggested_role": "which role should fill this - contractor or engineer or architect or project_manager",
  "header_fields": [
    {"label": "exact field name from document", "type": "text"}
  ],
  "sections": [
    {
      "name": "exact section name from document, or 'Checklist' if no sections",
      "has_time_fields": false,
      "has_weather_fields": false,
      "extra_fields": ["any extra fields in this section like Start Time, End Time, Weather"],
      "questions": [
        {
          "id": 1,
          "question": "exact question text from document",
          "answer_type": "yes_no_na or yes_no or text or number",
          "has_remarks": true,
          "category": "Safety or Quality or Compliance or General",
          "critical": false
        }
      ],
      "has_signature_block": true,
      "signature_teams": ["exactly what teams are listed for signatures"]
    }
  ]
}

STRICT RULES:
1. Extract ALL header fields at top of document (Project, Client, Date, Location etc - whatever exists)
2. Extract ALL sections (Pre/During/After OR any other section names)
3. Extract EVERY SINGLE question - do not skip any
4. Keep EXACT question text - do not paraphrase
5. answer_type = "yes_no_na" if Yes/No/NA columns exist
6. answer_type = "yes_no" if only Yes/No columns exist  
7. answer_type = "text" if it is a text input field
8. has_remarks = true if there is a Remarks column
9. critical = true if question relates to safety
10. has_time_fields = true if section has Start Time or End Time
11. has_weather_fields = true if section has Weather field
12. signature_teams = exact team names listed (Client Team, PMC Team, Contractor Team etc)
13. If document has NO sections, put everything in one section named "Checklist"
14. Return ONLY the JSON object, absolutely no other text"""

        response = model.generate_content([
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "application/pdf",
                            "data": pdf_base64
                        }
                    },
                    {"text": prompt}
                ]
            }
        ])

        raw = response.text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        
        # Find JSON start and end
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        
        extracted = json.loads(raw)

        # Flatten all questions with section info
        all_questions = []
        qid = 1
        for section in extracted.get("sections", []):
            for q in section.get("questions", []):
                q["id"] = qid
                q["section"] = section["name"]
                all_questions.append(q)
                qid += 1
        extracted["questions"] = all_questions

        total = len(all_questions)
        sections = len(extracted.get("sections", []))
        header_fields = len(extracted.get("header_fields", []))

        return {
            "success": True,
            "data": extracted,
            "stats": {
                "total_questions": total,
                "total_sections": sections,
                "header_fields": header_fields
            }
        }
    except json.JSONDecodeError as e:
        return {"success": False, "error": f"AI returned invalid format. Try again. Detail: {str(e)}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- SAVE CUSTOM CHECKLIST ---
@app.post("/checklist/save-custom")
async def save_custom_checklist(
    work_type: str = Form(...),
    title: str = Form(...),
    description: str = Form(""),
    created_by: str = Form(...),
    questions: str = Form(...),
    header_fields: str = Form("[]"),
    sections_meta: str = Form("[]"),
    db: Session = Depends(get_db)
):
    try:
        questions_data = json.loads(questions)
        template = ChecklistTemplate(
            work_type=work_type,
            title=title,
            questions=json.dumps(questions_data),
            header_fields=header_fields,
            sections_meta=sections_meta,
            created_by=created_by
        )
        db.add(template)
        db.commit()
        db.refresh(template)
        return {"success": True, "id": template.id, "message": "Checklist saved!"}
    except Exception as e:
        return {"success": False, "error": str(e)}

# --- added april 18th --- AUTH ---
class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not pwd_context.verify(req.password, user.password):
        return {"success": False, "message": "Invalid email or password."}
    
    token = jwt.encode(
        {"user_id": user.id, "email": user.email, "role": user.role, "name": user.name},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    return {
        "success": True,
        "token": token,
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "role": user.role
        }
    }

@app.get("/auth/users")
def get_users(db: Session = Depends(get_db)):
    users = db.query(User).all()
    return [{"id": u.id, "name": u.name, "email": u.email, "role": u.role} for u in users]

# ---------------------added on april 21st------------------------------------
@app.post("/checklist/template/duplicate/{template_id}")
async def duplicate_template(
    template_id: int,
    new_title: str = Form(...),
    questions: str = Form(...),
    work_type: str = Form(...),
    db: Session = Depends(get_db)
):
    try:
        original = db.query(ChecklistTemplate).filter(ChecklistTemplate.id == template_id).first()
        if not original:
            return {"success": False, "error": "Template not found"}
        new_template = ChecklistTemplate(
            work_type=work_type,
            title=new_title,
            questions=questions,
            header_fields=original.header_fields or "[]",
            sections_meta=original.sections_meta or "[]",
            created_by="admin"
        )
        db.add(new_template)
        db.commit()
        db.refresh(new_template)
        return {"success": True, "id": new_template.id, "message": "Template duplicated!"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.delete("/checklist/template/{template_id}")
def delete_template(template_id: int, db: Session = Depends(get_db)):
    try:
        template = db.query(ChecklistTemplate).filter(ChecklistTemplate.id == template_id).first()
        if not template:
            return {"success": False, "error": "Not found"}
        db.delete(template)
        db.commit()
        return {"success": True, "message": "Deleted"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/uploads/{filename}")
def serve_image(filename: str):
    filepath = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath)

@app.get("/checklist/report/{assignment_id}")
async def generate_report(assignment_id: int, db: Session = Depends(get_db)):
    try:
        assignment = db.query(ChecklistAssignment).filter(
            ChecklistAssignment.id == assignment_id
        ).first()
        if not assignment:
            raise HTTPException(status_code=404, detail="Assignment not found")

        template = db.query(ChecklistTemplate).filter(
            ChecklistTemplate.id == assignment.template_id
        ).first()

        response = db.query(ChecklistResponse).filter(
            ChecklistResponse.assignment_id == assignment_id
        ).first()
        if not response:
            raise HTTPException(status_code=404, detail="No submission found")

        # Parse data
        try:
            raw = json.loads(response.responses)
            if isinstance(raw, dict):
                question_responses = raw.get("questions", [])
                header_responses = raw.get("header", [])
                signatures = raw.get("signatures", {})
            else:
                question_responses = raw
                header_responses = []
                signatures = {}
        except:
            question_responses = []
            header_responses = []
            signatures = {}

        image_paths = json.loads(response.image_paths or "[]")
        sections_meta = json.loads(template.sections_meta or "[]") if template else []

        # Group questions by section
        section_groups = {}
        for r in question_responses:
            sec = r.get("section") or "Checklist"
            if sec not in section_groups:
                section_groups[sec] = []
            section_groups[sec].append(r)

        total = len(question_responses)
        passed = sum(1 for r in question_responses if r.get("answer") == "Yes")
        failed = sum(1 for r in question_responses if r.get("answer") == "No")
        na = sum(1 for r in question_responses if r.get("answer") == "NA")

        # Build PDF
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm,
            leftMargin=2*cm,
            topMargin=2*cm,
            bottomMargin=2*cm
        )

        styles = getSampleStyleSheet()
        elements = []

        # Custom styles
        title_style = ParagraphStyle('Title', parent=styles['Normal'],
            fontSize=22, fontName='Helvetica-Bold', textColor=colors.HexColor('#1e293b'),
            alignment=TA_CENTER, spaceAfter=6)
        subtitle_style = ParagraphStyle('Subtitle', parent=styles['Normal'],
            fontSize=11, textColor=colors.HexColor('#64748b'), alignment=TA_CENTER, spaceAfter=4)
        section_style = ParagraphStyle('Section', parent=styles['Normal'],
            fontSize=14, fontName='Helvetica-Bold', textColor=colors.HexColor('#f97316'),
            spaceBefore=16, spaceAfter=8)
        heading_style = ParagraphStyle('Heading', parent=styles['Normal'],
            fontSize=12, fontName='Helvetica-Bold', textColor=colors.HexColor('#1e293b'),
            spaceBefore=12, spaceAfter=6)
        normal_style = ParagraphStyle('Normal2', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#334155'), spaceAfter=4)
        small_style = ParagraphStyle('Small', parent=styles['Normal'],
            fontSize=9, textColor=colors.HexColor('#64748b'), spaceAfter=2)
        passed_style = ParagraphStyle('Passed', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#10b981'), spaceAfter=3)
        failed_style = ParagraphStyle('Failed', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#ef4444'), spaceAfter=3)
        na_style = ParagraphStyle('NA', parent=styles['Normal'],
            fontSize=10, textColor=colors.HexColor('#64748b'), spaceAfter=3)

        # ── HEADER ──
        elements.append(Paragraph("🏛 ConstructAI", title_style))
        elements.append(Paragraph("Construction Quality Inspection Report", subtitle_style))
        elements.append(Paragraph("IIT Guwahati · CE499 BTP · 2026", subtitle_style))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#f97316')))
        elements.append(Spacer(1, 12))

        # ── REPORT META ──
        elements.append(Paragraph("Report Details", heading_style))
        meta_data = [
            ["Checklist", template.title if template else "Unknown"],
            ["Work Type", template.work_type if template else "Unknown"],
            ["Assigned To", assignment.assigned_to],
            ["Project Name", assignment.project_name],
            ["Due Date", assignment.due_date],
            ["Submitted By", response.submitted_by],
            ["Submission Date", str(response.timestamp)[:19]],
        ]
        meta_table = Table(meta_data, colWidths=[4*cm, 13*cm])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f8fafc')),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#475569')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 12))

        # ── PROJECT INFO ──
        if header_responses:
            elements.append(Paragraph("Project Information", heading_style))
            header_data = [[h['label'], h['value'] or '—'] for h in header_responses]
            header_table = Table(header_data, colWidths=[4*cm, 13*cm])
            header_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#475569')),
                ('TEXTCOLOR', (1, 0), (1, -1), colors.HexColor('#1e293b')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#f8fafc'), colors.white]),
            ]))
            elements.append(header_table)
            elements.append(Spacer(1, 12))

        # ── SUMMARY ──
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Paragraph("Inspection Summary", heading_style))
        summary_data = [
            ["Total Questions", str(total), "Passed ✅", str(passed)],
            ["Failed ❌", str(failed), "Not Applicable", str(na)],
        ]
        summary_table = Table(summary_data, colWidths=[4*cm, 4*cm, 5*cm, 4*cm])
        summary_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor('#1e293b')),
            ('TEXTCOLOR', (2, 0), (3, 0), colors.HexColor('#10b981')),
            ('TEXTCOLOR', (0, 1), (1, 1), colors.HexColor('#ef4444')),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(summary_table)
        elements.append(Spacer(1, 8))

        if response.ai_summary:
            elements.append(Paragraph("AI Inspection Summary:", heading_style))
            clean_summary = response.ai_summary.replace('**', '')
            elements.append(Paragraph(clean_summary, normal_style))
        elements.append(Spacer(1, 12))

        # ── QUESTIONS BY SECTION ──
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
        elements.append(Paragraph("Inspection Results", heading_style))

        for sec_name, sec_questions in section_groups.items():
            elements.append(Paragraph(f"● {sec_name}", section_style))
            q_data = [["#", "Question", "Answer", "Remark"]]
            for i, r in enumerate(sec_questions):
                answer = r.get("answer", "—")
                if answer == "Yes":
                    answer_text = "✅ Yes"
                elif answer == "No":
                    answer_text = "❌ No"
                else:
                    answer_text = "— NA"
                q_data.append([
                    str(i + 1),
                    Paragraph(r.get("question", ""), normal_style),
                    answer_text,
                    Paragraph(r.get("remark", "") or "—", small_style)
                ])
            q_table = Table(q_data, colWidths=[0.8*cm, 9*cm, 2.5*cm, 4.7*cm])
            q_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e293b')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (2, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            elements.append(q_table)
            elements.append(Spacer(1, 8))

            # Signature block for this section
            sec_meta = next((s for s in sections_meta if s.get("name") == sec_name), None)
            if sec_meta and sec_meta.get("signature_teams"):
                elements.append(Paragraph(f"Signatures — {sec_name}", small_style))
                sig_teams = sec_meta["signature_teams"]
                sig_data = [sig_teams]
                sig_fields = []
                for ti, team in enumerate(sig_teams):
                    sig_block = signatures.get(sec_name, {}).get(str(ti), {})
                    sig_fields.append(
                        f"Name: {sig_block.get('name', '________________')}\n"
                        f"Sign: {sig_block.get('signature', '________________')}\n"
                        f"Date: {sig_block.get('date', '________________')}"
                    )
                sig_data.append(sig_fields)
                col_width = 17*cm / max(len(sig_teams), 1)
                sig_table = Table(sig_data, colWidths=[col_width]*len(sig_teams))
                sig_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#fff7ed')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#f97316')),
                    ('PADDING', (0, 0), (-1, -1), 8),
                ]))
                elements.append(sig_table)
                elements.append(Spacer(1, 12))

        # ── SITE IMAGES ──
        if image_paths:
            elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e2e8f0')))
            elements.append(Paragraph("Site Images", heading_style))
            elements.append(Paragraph(f"{len(image_paths)} image(s) uploaded for this inspection.", small_style))
            elements.append(Spacer(1, 8))

            img_row = []
            for i, img_path in enumerate(image_paths):
                try:
                    full_path = os.path.join(UPLOAD_DIR, img_path)
                    print(f"Trying image: {full_path}")
                    print(f"File exists: {os.path.exists(full_path)}")

                    if os.path.exists(full_path):
                        # Open and convert to RGB (handles PNG, JPEG, WEBP etc)
                        pil_img = PILImage.open(full_path).convert('RGB')
                        pil_img.thumbnail((300, 300), PILImage.LANCZOS)
                
                        img_buffer = BytesIO()
                        pil_img.save(img_buffer, format='JPEG', quality=85)
                        img_buffer.seek(0)
                
                        rl_img = RLImage(img_buffer, width=5.5*cm, height=5.5*cm)
                        img_row.append(rl_img)
                    else:
                        # List all files in uploads to debug
                        all_files = os.listdir(UPLOAD_DIR)
                        print(f"Files in uploads: {all_files}")
                        img_row.append(Paragraph(f"Image {i+1}\nnot found", small_style))
                except Exception as e:
                    print(f"Image error for {img_path}: {str(e)}")
                    img_row.append(Paragraph(f"Image {i+1}\n{str(e)[:30]}", small_style))

                if len(img_row) == 3 or i == len(image_paths) - 1:
                    while len(img_row) < 3:
                        img_row.append(Paragraph("", small_style))
                    img_table = Table([img_row], colWidths=[6*cm, 6*cm, 6*cm])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                        ('PADDING', (0, 0), (-1, -1), 6),
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ]))
                    elements.append(img_table)
                    elements.append(Spacer(1, 8))
                    img_row = []

        # ── FOOTER ──
        elements.append(Spacer(1, 20))
        elements.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor('#f97316')))
        elements.append(Paragraph(
            "Generated by ConstructAI · IIT Guwahati CE499 BTP · 2026",
            ParagraphStyle('Footer', parent=styles['Normal'],
                fontSize=8, textColor=colors.HexColor('#94a3b8'),
                alignment=TA_CENTER, spaceBefore=8)
        ))

        doc.build(elements)
        buffer.seek(0)

        filename = f"inspection_report_{assignment_id}_{assignment.project_name.replace(' ', '_')}.pdf"
        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# ------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ConstructAI backend running with SQLite"}

# ------------------- IF WE DONT WANT DATABASE --------------------

# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# import google.generativeai as genai

# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# genai.configure(api_key="AIzaSyACtvwhWk2z4b2IasKVzWJgnJgSoD5vgSc")
# model = genai.GenerativeModel("gemini-2.5-flash")

# class ChatRequest(BaseModel):
#     role: str
#     messages: list
#     system: str

# @app.post("/chat")
# async def chat(req: ChatRequest):
#     try:
#         last_msg = req.messages[-1]["content"]
#         prompt = f"{req.system}\n\nUser question: {last_msg}"
#         response = model.generate_content(prompt)
#         return {"reply": response.text}
#     except Exception as e:
#         return {"reply": f"Error: {str(e)}"}
# @app.get("/")
# def root():
#     return {"status": "OK"}


## ----------------------------------------------------------

# from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import FileResponse
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# import google.generativeai as genai
# import aiofiles
# import os
# import json
# import re
# import tempfile
# from typing import List, Optional

# from database import (
#     create_tables,
#     get_db,
#     ChatMessage,
#     Document,
#     WorkflowRequest,
#     ChecklistTemplate,
#     ChecklistAssignment,
#     ChecklistResponse,
# )

# try:
#     from pypdf import PdfReader
# except ImportError:
#     PdfReader = None

# # --- SETUP ---
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyAEHzQlFjfzJOV2QOeQ_IViRX5bnm9tqhM")
# GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# genai.configure(api_key=GEMINI_API_KEY)
# model = genai.GenerativeModel(GEMINI_MODEL)

# UPLOAD_DIR = "uploads"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# create_tables()

# # --- SCHEMAS ---
# class ChatRequest(BaseModel):
#     role: str
#     messages: list
#     system: str


# class WorkflowCreate(BaseModel):
#     title: str
#     description: str
#     submitted_by: str
#     request_type: str = "general"


# class ChecklistQuestionIn(BaseModel):
#     order_no: int
#     question: str
#     answer_type: str = "yes_no"
#     required: bool = True
#     remarks_required: bool = False


# class ChecklistTemplateCreate(BaseModel):
#     title: str
#     work_type: str
#     description: Optional[str] = ""
#     questions: List[ChecklistQuestionIn]


# # --- HELPERS ---
# def safe_json_loads(value, fallback):
#     try:
#         return json.loads(value) if value else fallback
#     except Exception:
#         return fallback


# def extract_json_from_text(text: str):
#     text = text.strip()

#     if text.startswith("```"):
#         text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
#         text = re.sub(r"^```", "", text).strip()
#         text = re.sub(r"```$", "", text).strip()

#     start = text.find("{")
#     end = text.rfind("}")

#     if start != -1 and end != -1 and end > start:
#         text = text[start:end + 1]

#     return json.loads(text)


# def extract_pdf_text(upload_file: UploadFile) -> str:
#     if PdfReader is None:
#         raise HTTPException(
#             status_code=500,
#             detail="pypdf is not installed. Run: pip install pypdf"
#         )

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         file_bytes = upload_file.file.read()
#         tmp.write(file_bytes)
#         tmp_path = tmp.name

#     try:
#         reader = PdfReader(tmp_path)
#         text_parts = []

#         for page in reader.pages:
#             try:
#                 page_text = page.extract_text() or ""
#                 if page_text.strip():
#                     text_parts.append(page_text)
#             except Exception:
#                 continue

#         full_text = "\n".join(text_parts).strip()

#         if not full_text:
#             raise HTTPException(
#                 status_code=400,
#                 detail="Could not extract readable text from PDF."
#             )

#         return full_text
#     finally:
#         try:
#             os.remove(tmp_path)
#         except Exception:
#             pass


# def build_checklist_extraction_prompt(pdf_text: str, work_type: str) -> str:
#     return f"""
# You are an expert construction QA/QC assistant.

# Convert the following construction checklist PDF text into structured JSON.

# Rules:
# - Return ONLY valid JSON
# - No markdown
# - No explanation
# - Extract a checklist title
# - Use the given work type unless the document clearly indicates a better matching work type
# - Create short professional question text
# - Remove duplicates
# - If the checklist item is inspection style, use answer_type = "yes_no"
# - If the item needs both result and remarks, use answer_type = "yes_no_remarks"
# - If it expects a written field, use "text"
# - If it expects a numeric value, use "number"

# Return this exact JSON shape:
# {{
#   "title": "string",
#   "work_type": "string",
#   "description": "string",
#   "questions": [
#     {{
#       "question": "string",
#       "answer_type": "yes_no",
#       "required": true,
#       "remarks_required": false
#     }}
#   ]
# }}

# Work type:
# {work_type}

# PDF text:
# \"\"\"
# {pdf_text[:30000]}
# \"\"\"
# """.strip()


# # --- CHAT ---
# @app.post("/chat")
# async def chat(req: ChatRequest, db: Session = Depends(get_db)):
#     try:
#         docs = db.query(Document).order_by(Document.timestamp.desc()).all()
#         doc_list = "\n".join([
#             f"- {d.filename} (uploaded by {d.uploaded_by}, description: {d.description})"
#             for d in docs
#         ]) or "No documents uploaded yet."

#         last_msg = req.messages[-1]["content"]
#         prompt = f"""{req.system}

# REAL PROJECT DOCUMENTS CURRENTLY IN THE SYSTEM:
# {doc_list}

# Only reference documents listed above. Do not mention any other documents.

# User question: {last_msg}"""

#         response = model.generate_content(prompt)
#         reply = response.text

#         db.add(ChatMessage(role_user=req.role, sender="user", message=last_msg))
#         db.add(ChatMessage(role_user=req.role, sender="assistant", message=reply))
#         db.commit()

#         return {"reply": reply}
#     except Exception as e:
#         return {"reply": f"Error: {str(e)}"}


# @app.get("/chat/history/{role}")
# def get_history(role: str, db: Session = Depends(get_db)):
#     messages = db.query(ChatMessage).filter(
#         ChatMessage.role_user == role
#     ).order_by(ChatMessage.timestamp).all()

#     return [
#         {"sender": m.sender, "message": m.message, "time": str(m.timestamp)}
#         for m in messages
#     ]


# @app.delete("/chat/clear/{role}")
# def clear_history(role: str, db: Session = Depends(get_db)):
#     db.query(ChatMessage).filter(ChatMessage.role_user == role).delete()
#     db.commit()
#     return {"message": "Chat history cleared"}


# # --- DOCUMENTS ---
# @app.post("/documents/upload")
# async def upload_document(
#     file: UploadFile = File(...),
#     uploaded_by: str = Form(...),
#     description: str = Form(""),
#     db: Session = Depends(get_db)
# ):
#     filepath = os.path.join(UPLOAD_DIR, file.filename)

#     async with aiofiles.open(filepath, "wb") as f:
#         content = await file.read()
#         await f.write(content)

#     doc = Document(
#         filename=file.filename,
#         filepath=filepath,
#         uploaded_by=uploaded_by,
#         description=description
#     )
#     db.add(doc)
#     db.commit()
#     db.refresh(doc)

#     return {
#         "message": "Uploaded successfully",
#         "id": doc.id,
#         "filename": file.filename
#     }


# @app.get("/documents")
# def get_documents(db: Session = Depends(get_db)):
#     docs = db.query(Document).order_by(Document.timestamp.desc()).all()
#     return [
#         {
#             "id": d.id,
#             "filename": d.filename,
#             "uploaded_by": d.uploaded_by,
#             "description": d.description,
#             "time": str(d.timestamp)
#         }
#         for d in docs
#     ]


# @app.get("/documents/download/{doc_id}")
# def download_document(doc_id: int, db: Session = Depends(get_db)):
#     doc = db.query(Document).filter(Document.id == doc_id).first()
#     if not doc:
#         return {"error": "Document not found"}

#     return FileResponse(
#         path=doc.filepath,
#         filename=doc.filename,
#         media_type="application/octet-stream"
#     )


# @app.post("/documents/ask")
# async def ask_document(
#     document_id: int = Form(...),
#     question: str = Form(...),
#     role: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     doc = db.query(Document).filter(Document.id == document_id).first()
#     if not doc:
#         return {"reply": "Document not found."}

#     try:
#         async with aiofiles.open(doc.filepath, "rb") as f:
#             await f.read()

#         prompt = f"""You are an AI assistant for a construction project.
# A {role} is asking about the document '{doc.filename}'.
# Document description: {doc.description}

# Question: {question}

# Note: Respond as if you have read the document and provide helpful construction-domain insights."""

#         response = model.generate_content(prompt)
#         return {"reply": response.text}
#     except Exception as e:
#         return {"reply": f"Error reading document: {str(e)}"}


# # --- WORKFLOW ---
# @app.post("/workflow/submit")
# async def submit_workflow(req: WorkflowCreate, db: Session = Depends(get_db)):
#     try:
#         prompt = f"""A {req.submitted_by} submitted a construction change request.
# Title: {req.title}
# Description: {req.description}
# Type: {req.request_type}

# Write a 2-3 sentence professional summary of this request for the project team."""
#         response = model.generate_content(prompt)
#         ai_summary = response.text
#     except Exception:
#         ai_summary = req.description

#     wf = WorkflowRequest(
#         title=req.title,
#         description=req.description,
#         request_type=req.request_type,
#         submitted_by=req.submitted_by,
#         current_stage="architect",
#         status="in_progress",
#         ai_summary=ai_summary
#     )
#     db.add(wf)
#     db.commit()
#     db.refresh(wf)

#     return {"message": "Submitted", "id": wf.id, "ai_summary": ai_summary}


# @app.get("/workflow")
# def get_workflow(db: Session = Depends(get_db)):
#     items = db.query(WorkflowRequest).order_by(WorkflowRequest.timestamp.desc()).all()

#     return [{
#         "id": w.id,
#         "title": w.title,
#         "description": w.description,
#         "request_type": w.request_type,
#         "submitted_by": w.submitted_by,
#         "current_stage": w.current_stage,
#         "status": w.status,
#         "ai_summary": w.ai_summary,
#         "architect_comment": w.architect_comment,
#         "architect_status": w.architect_status,
#         "architect_attachment": w.architect_attachment,
#         "engineer_comment": w.engineer_comment,
#         "engineer_status": w.engineer_status,
#         "engineer_attachment": w.engineer_attachment,
#         "contractor_comment": w.contractor_comment,
#         "contractor_status": w.contractor_status,
#         "contractor_attachment": w.contractor_attachment,
#         "pm_comment": w.pm_comment,
#         "pm_status": w.pm_status,
#         "pm_attachment": w.pm_attachment,
#         "time": str(w.timestamp)
#     } for w in items]


# @app.post("/workflow/{wf_id}/approve")
# async def approve_workflow(
#     wf_id: int,
#     role: str = Form(...),
#     comment: str = Form(""),
#     file: UploadFile = File(None),
#     db: Session = Depends(get_db)
# ):
#     wf = db.query(WorkflowRequest).filter(WorkflowRequest.id == wf_id).first()
#     if not wf:
#         return {"error": "Not found"}

#     if file and file.filename:
#         filename = f"workflow_{wf_id}_{role}_{file.filename}"
#         filepath = os.path.join(UPLOAD_DIR, filename)

#         async with aiofiles.open(filepath, "wb") as f:
#             content = await file.read()
#             await f.write(content)

#         setattr(wf, f"{role}_attachment", filename)

#     stages = ["architect", "engineer", "contractor", "project_manager"]
#     setattr(wf, f"{role}_comment", comment)
#     setattr(wf, f"{role}_status", "approved")

#     current_index = stages.index(role) if role in stages else -1
#     if current_index < len(stages) - 1:
#         wf.current_stage = stages[current_index + 1]
#     else:
#         wf.current_stage = "completed"
#         wf.status = "approved"

#     db.commit()
#     return {"message": "Approved", "next_stage": wf.current_stage}


# @app.get("/workflow/attachment/{filename}")
# def download_workflow_attachment(filename: str):
#     filepath = os.path.join(UPLOAD_DIR, filename)
#     if not os.path.exists(filepath):
#         return {"error": "File not found"}

#     return FileResponse(
#         path=filepath,
#         filename=filename,
#         media_type="application/octet-stream"
#     )


# @app.post("/workflow/{wf_id}/reject")
# async def reject_workflow(
#     wf_id: int,
#     role: str = Form(...),
#     comment: str = Form(""),
#     db: Session = Depends(get_db)
# ):
#     wf = db.query(WorkflowRequest).filter(WorkflowRequest.id == wf_id).first()
#     if not wf:
#         return {"error": "Not found"}

#     setattr(wf, f"{role}_comment", comment)
#     setattr(wf, f"{role}_status", "rejected")
#     wf.current_stage = "rejected"
#     wf.status = "rejected"

#     db.commit()
#     return {"message": "Rejected"}


# # --- CHECKLIST: DEFAULT QUESTION BANK ---
# CHECKLIST_QUESTIONS = {
#     "Brickwork": [
#         "Check for availability of approved drawings for brickwork?",
#         "Check for material quality (bricks, mortar)?",
#         "Check for proper alignment and plumb of walls?",
#         "Check for mortar ratio as specified?",
#         "Check for proper curing of brickwork?",
#         "Check for openings (doors/windows) as per drawings?",
#         "Check for scaffolding safety?",
#     ],
#     "Concrete": [
#         "Check for availability of approved mix design?",
#         "Check for reinforcement placement as per drawings?",
#         "Check for shuttering/formwork is properly fixed?",
#         "Check for cover blocks placed correctly?",
#         "Check for concrete grade as specified?",
#         "Check for proper vibration during pouring?",
#         "Check for curing done properly?",
#     ],
#     "Plaster": [
#         "Check for surface preparation before plastering?",
#         "Check for approved mix ratio for plaster?",
#         "Check for thickness of plaster as specified?",
#         "Check for proper curing of plaster?",
#         "Check for finishing is smooth and even?",
#         "Check for cracks or defects in plaster?",
#     ],
#     "Tiles": [
#         "Check for approved tile samples and specifications?",
#         "Check for surface level before tiling?",
#         "Check for approved adhesive/mortar used?",
#         "Check for tile alignment and pattern as per drawing?",
#         "Check for grouting done properly?",
#         "Check for broken or damaged tiles?",
#         "Check for proper cleaning after installation?",
#     ],
#     "Steel Fabrication": [
#         "Check for availability of approved shop drawings for fabrication?",
#         "Check for availability of structural steel sections and welding rods?",
#         "Check for straightening of members, is it acceptable?",
#         "Check for availability of authorised welders for welding operations?",
#         "Check for cutting of the sections, is it as specified?",
#         "Check for dimensions of assembly, is it OK?",
#         "Check for joint preparation, is it OK?",
#         "Check for weld quality and thickness, is it as specified?",
#         "Check for hole punching, is it as specified?",
#         "Check for surface preparation prior to painting?",
#         "Check for lifting and transportation arrangements prior to erection?",
#         "Check for completion of joint records for inspection?",
#     ]
# }


# # --- CHECKLIST: OLD MANUAL TEMPLATE CREATION ---
# @app.post("/checklist/template")
# async def create_template(
#     work_type: str = Form(...),
#     title: str = Form(...),
#     created_by: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     base_questions = CHECKLIST_QUESTIONS.get(work_type, [])
#     questions = []

#     for index, question in enumerate(base_questions, start=1):
#         questions.append({
#             "order_no": index,
#             "question": question,
#             "answer_type": "yes_no",
#             "required": True,
#             "remarks_required": False
#         })

#     template = ChecklistTemplate(
#         work_type=work_type,
#         title=title,
#         description="",
#         questions=json.dumps(questions),
#         created_by=created_by,
#         source_pdf=""
#     )
#     db.add(template)
#     db.commit()
#     db.refresh(template)

#     return {"message": "Template created", "id": template.id, "questions": questions}


# # --- CHECKLIST: AI PDF EXTRACTION ---
# @app.post("/checklist/extract-from-pdf")
# async def extract_checklist_from_pdf(
#     file: UploadFile = File(...),
#     work_type: str = Form(...),
# ):
#     if not file.filename.lower().endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF files are supported.")

#     pdf_text = extract_pdf_text(file)

#     try:
#         prompt = build_checklist_extraction_prompt(pdf_text, work_type)
#         response = model.generate_content(prompt)
#         parsed = extract_json_from_text(response.text)

#         title = parsed.get("title", "Generated Checklist")
#         generated_work_type = parsed.get("work_type", work_type)
#         description = parsed.get("description", "")
#         raw_questions = parsed.get("questions", [])

#         cleaned_questions = []
#         for idx, item in enumerate(raw_questions, start=1):
#             question_text = (item.get("question") or "").strip()
#             if not question_text:
#                 continue

#             cleaned_questions.append({
#                 "order_no": idx,
#                 "question": question_text,
#                 "answer_type": item.get("answer_type", "yes_no"),
#                 "required": bool(item.get("required", True)),
#                 "remarks_required": bool(item.get("remarks_required", False)),
#             })

#         if not cleaned_questions:
#             raise HTTPException(
#                 status_code=400,
#                 detail="No checklist questions could be extracted from this PDF."
#             )

#         return {
#             "title": title,
#             "work_type": generated_work_type,
#             "description": description,
#             "questions": cleaned_questions,
#         }

#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Checklist extraction failed: {str(e)}"
#         )


# # --- CHECKLIST: SAVE AI FINALIZED TEMPLATE ---
# @app.post("/checklist/templates/create-ai")
# async def create_ai_template(payload: ChecklistTemplateCreate, db: Session = Depends(get_db)):
#     cleaned_questions = []

#     for idx, q in enumerate(payload.questions, start=1):
#         question_text = q.question.strip()
#         if not question_text:
#             continue

#         cleaned_questions.append({
#             "order_no": idx,
#             "question": question_text,
#             "answer_type": q.answer_type or "yes_no",
#             "required": q.required,
#             "remarks_required": q.remarks_required,
#         })

#     if not cleaned_questions:
#         raise HTTPException(status_code=400, detail="At least one valid question is required.")

#     template = ChecklistTemplate(
#         work_type=payload.work_type,
#         title=payload.title,
#         description=payload.description or "",
#         questions=json.dumps(cleaned_questions),
#         created_by="admin",
#         source_pdf="ai_generated"
#     )
#     db.add(template)
#     db.commit()
#     db.refresh(template)

#     return {
#         "message": "Checklist template created successfully",
#         "id": template.id
#     }


# # --- CHECKLIST: GET TEMPLATES ---
# @app.get("/checklist/templates")
# def get_templates(db: Session = Depends(get_db)):
#     templates = db.query(ChecklistTemplate).order_by(ChecklistTemplate.timestamp.desc()).all()

#     return [
#         {
#             "id": t.id,
#             "work_type": t.work_type,
#             "title": t.title,
#             "description": t.description or "",
#             "questions": safe_json_loads(t.questions, []),
#             "created_by": t.created_by,
#             "source_pdf": t.source_pdf or "",
#             "question_count": len(safe_json_loads(t.questions, [])),
#             "time": str(t.timestamp),
#         }
#         for t in templates
#     ]


# # --- CHECKLIST: ASSIGN ---
# @app.post("/checklist/assign")
# async def assign_checklist(request: Request, db: Session = Depends(get_db)):
#     content_type = request.headers.get("content-type", "")

#     if "application/json" in content_type:
#         data = await request.json()
#         template_id = data.get("template_id")
#         assigned_to = data.get("assigned_to")
#         project_name = data.get("project_name")
#         due_date = data.get("due_date")
#         assigned_by = data.get("assigned_by", "admin")
#     else:
#         form = await request.form()
#         template_id = form.get("template_id")
#         assigned_to = form.get("assigned_to")
#         project_name = form.get("project_name")
#         due_date = form.get("due_date")
#         assigned_by = form.get("assigned_by", "admin")

#     if not template_id or not assigned_to or not project_name or not due_date:
#         raise HTTPException(status_code=400, detail="Missing required assignment fields.")

#     template = db.query(ChecklistTemplate).filter(ChecklistTemplate.id == int(template_id)).first()
#     if not template:
#         raise HTTPException(status_code=404, detail="Checklist template not found.")

#     assignment = ChecklistAssignment(
#         template_id=int(template_id),
#         assigned_to=assigned_to,
#         assigned_by=assigned_by,
#         project_name=project_name,
#         due_date=due_date,
#         status="pending"
#     )
#     db.add(assignment)
#     db.commit()
#     db.refresh(assignment)

#     return {"message": "Assigned successfully", "id": assignment.id}


# # --- CHECKLIST: ASSIGNMENTS BY ROLE ---
# @app.get("/checklist/assignments/{role}")
# def get_assignments(role: str, db: Session = Depends(get_db)):
#     assignments = db.query(ChecklistAssignment).filter(
#         ChecklistAssignment.assigned_to == role
#     ).order_by(ChecklistAssignment.timestamp.desc()).all()

#     result = []
#     for a in assignments:
#         template = db.query(ChecklistTemplate).filter(ChecklistTemplate.id == a.template_id).first()
#         result.append({
#             "id": a.id,
#             "template_id": a.template_id,
#             "work_type": template.work_type if template else "Unknown",
#             "title": template.title if template else "Unknown",
#             "description": template.description if template else "",
#             "questions": safe_json_loads(template.questions, []) if template else [],
#             "project_name": a.project_name,
#             "due_date": a.due_date,
#             "status": a.status,
#             "time": str(a.timestamp)
#         })

#     return result


# # --- CHECKLIST: ALL ASSIGNMENTS ---
# @app.get("/checklist/assignments/all/list")
# def get_all_assignments(db: Session = Depends(get_db)):
#     assignments = db.query(ChecklistAssignment).order_by(ChecklistAssignment.timestamp.desc()).all()
#     result = []

#     for a in assignments:
#         template = db.query(ChecklistTemplate).filter(ChecklistTemplate.id == a.template_id).first()
#         response = db.query(ChecklistResponse).filter(ChecklistResponse.assignment_id == a.id).first()

#         result.append({
#             "id": a.id,
#             "template_id": a.template_id,
#             "work_type": template.work_type if template else "Unknown",
#             "title": template.title if template else "Unknown",
#             "description": template.description if template else "",
#             "assigned_to": a.assigned_to,
#             "assigned_by": a.assigned_by,
#             "project_name": a.project_name,
#             "due_date": a.due_date,
#             "status": a.status,
#             "response": {
#                 "responses": safe_json_loads(response.responses, []),
#                 "ai_summary": response.ai_summary,
#                 "issues_found": response.issues_found
#             } if response else None,
#             "time": str(a.timestamp)
#         })

#     return result


# # --- CHECKLIST: SUBMIT ---
# @app.post("/checklist/submit")
# async def submit_checklist(
#     assignment_id: int = Form(...),
#     submitted_by: str = Form(...),
#     responses: str = Form(...),
#     db: Session = Depends(get_db)
# ):
#     responses_data = safe_json_loads(responses, [])
#     issues = sum(1 for r in responses_data if str(r.get("answer", "")).strip().lower() == "no")

#     try:
#         issues_text = "\n".join([
#             f"- {r.get('question', '')}: {r.get('answer', '')} {('ISSUE' if str(r.get('answer', '')).strip().lower() == 'no' else '')}"
#             for r in responses_data
#         ])

#         prompt = f"""You are a construction quality inspector AI.
# A {submitted_by} submitted a quality checklist with the following responses:

# {issues_text}

# Issues found: {issues}

# Write a professional 3-4 sentence inspection summary. Highlight any No answers as issues requiring immediate attention. Be concise and actionable."""
#         response = model.generate_content(prompt)
#         ai_summary = response.text
#     except Exception:
#         ai_summary = f"Checklist submitted with {issues} issues found."

#     checklist_response = ChecklistResponse(
#         assignment_id=assignment_id,
#         submitted_by=submitted_by,
#         responses=responses,
#         ai_summary=ai_summary,
#         issues_found=issues
#     )
#     db.add(checklist_response)

#     assignment = db.query(ChecklistAssignment).filter(ChecklistAssignment.id == assignment_id).first()
#     if assignment:
#         assignment.status = "submitted"

#     db.commit()

#     return {
#         "message": "Submitted",
#         "issues_found": issues,
#         "ai_summary": ai_summary
#     }


# @app.get("/")
# def root():
#     return {"status": "ConstructAI backend running with SQLite"}