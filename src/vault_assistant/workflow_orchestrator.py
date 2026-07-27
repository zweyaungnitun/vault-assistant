"""
Dynamic Workflow Orchestrator
Executes multi-step tasks: retrieve → scan → extract → draft → approve
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import sqlite3
import json


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_APPROVAL = "waiting_approval"


class TaskType(Enum):
    RETRIEVE = "retrieve"
    SCAN_PII = "scan_pii"
    EXTRACT_INFO = "extract_info"
    DRAFT_DOCUMENT = "draft_document"
    APPROVE = "approve"
    CUSTOM = "custom"


@dataclass
class TaskStep:
    """A single step in a workflow."""
    id: str
    task_type: TaskType
    description: str
    status: TaskStatus = TaskStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    requires_approval: bool = False
    approved_by: Optional[str] = None


@dataclass
class Workflow:
    """A complete workflow with multiple steps."""
    id: str
    name: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    current_step_index: int = 0
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """Orchestrates multi-step agentic workflows."""
    
    def __init__(self, conn: sqlite3.Connection, client=None, model: str = "llama3:8b"):
        self.conn = conn
        self.client = client
        self.model = model
        self._init_schema()
        self.step_handlers: Dict[TaskType, Callable] = self._register_handlers()
    
    def _init_schema(self):
        """Initialize workflow tables."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflows (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                current_step_index INTEGER DEFAULT 0,
                created_by TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_steps (
                id TEXT PRIMARY KEY,
                workflow_id TEXT NOT NULL,
                step_index INTEGER NOT NULL,
                task_type TEXT NOT NULL,
                description TEXT,
                status TEXT DEFAULT 'pending',
                input_data JSON,
                output_data JSON,
                error TEXT,
                requires_approval BOOLEAN DEFAULT 0,
                approved_by TEXT,
                approved_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id)
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_id TEXT NOT NULL,
                step_id TEXT NOT NULL,
                approver TEXT NOT NULL,
                decision TEXT NOT NULL,
                comments TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (workflow_id) REFERENCES workflows(id),
                FOREIGN KEY (step_id) REFERENCES workflow_steps(id)
            )
        """)
        
        self.conn.commit()
    
    def _register_handlers(self) -> Dict[TaskType, Callable]:
        """Register handlers for each task type."""
        return {
            TaskType.RETRIEVE: self._handle_retrieve,
            TaskType.SCAN_PII: self._handle_scan_pii,
            TaskType.EXTRACT_INFO: self._handle_extract_info,
            TaskType.DRAFT_DOCUMENT: self._handle_draft_document,
            TaskType.APPROVE: self._handle_approval_step,
            TaskType.CUSTOM: self._handle_custom,
        }
    
    def create_workflow(self, name: str, description: str, 
                       steps: List[Dict[str, Any]], 
                       created_by: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None) -> Workflow:
        """Create a new workflow."""
        import uuid
        
        workflow_id = str(uuid.uuid4())
        workflow = Workflow(
            id=workflow_id,
            name=name,
            description=description,
            created_by=created_by,
            metadata=metadata or {}
        )
        
        # Create steps
        for idx, step_def in enumerate(steps):
            step = TaskStep(
                id=f"{workflow_id}_step_{idx}",
                task_type=TaskType(step_def.get('type', 'custom')),
                description=step_def.get('description', ''),
                input_data=step_def.get('input_data', {}),
                requires_approval=step_def.get('requires_approval', False)
            )
            workflow.steps.append(step)
        
        # Store in database
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO workflows (id, name, description, status, created_by, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (workflow_id, name, description, 'pending', created_by, json.dumps(metadata or {})))
        
        for idx, step in enumerate(workflow.steps):
            cursor.execute("""
                INSERT INTO workflow_steps (
                    id, workflow_id, step_index, task_type, description, 
                    status, input_data, requires_approval
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                step.id, workflow_id, idx, step.task_type.value, step.description,
                'pending', json.dumps(step.input_data), step.requires_approval
            ))
        
        self.conn.commit()
        return workflow
    
    def execute_workflow(self, workflow_id: str, 
                        context: Optional[Dict[str, Any]] = None) -> Workflow:
        """Execute a workflow step by step."""
        workflow = self.load_workflow(workflow_id)
        
        if workflow.status == TaskStatus.COMPLETED:
            return workflow
        
        workflow.status = TaskStatus.RUNNING
        context = context or {}
        
        while workflow.current_step_index < len(workflow.steps):
            step = workflow.steps[workflow.current_step_index]
            
            if step.status == TaskStatus.COMPLETED:
                workflow.current_step_index += 1
                continue
            
            # Execute step
            step.status = TaskStatus.RUNNING
            self._update_step(step)
            
            handler = self.step_handlers.get(step.task_type)
            if handler:
                try:
                    # Merge context with step input
                    step_input = {**context, **step.input_data}
                    output = handler(step, step_input, workflow)
                    step.output_data = output
                    step.status = TaskStatus.COMPLETED
                    
                    # Update context with output for next steps
                    context.update(output)
                    
                except Exception as e:
                    step.status = TaskStatus.FAILED
                    step.error = str(e)
                    workflow.status = TaskStatus.FAILED
                    self._update_step(step)
                    self._update_workflow(workflow)
                    raise
            else:
                step.status = TaskStatus.FAILED
                step.error = f"No handler for task type: {step.task_type}"
                workflow.status = TaskStatus.FAILED
                self._update_step(step)
                self._update_workflow(workflow)
                raise ValueError(f"No handler for task type: {step.task_type}")
            
            self._update_step(step)
            
            # Check if approval needed
            if step.requires_approval and step.status == TaskStatus.COMPLETED:
                workflow.status = TaskStatus.WAITING_APPROVAL
                self._update_workflow(workflow)
                return workflow
            
            workflow.current_step_index += 1
        
        # All steps completed
        workflow.status = TaskStatus.COMPLETED
        workflow.metadata['completed_at'] = str(sqlite3.connect(':memory:').execute('SELECT CURRENT_TIMESTAMP').fetchone()[0])
        self._update_workflow(workflow)
        
        return workflow
    
    def approve_step(self, workflow_id: str, step_id: str, 
                    approver: str, approved: bool, 
                    comments: Optional[str] = None) -> bool:
        """Approve or reject a workflow step."""
        cursor = self.conn.cursor()
        
        # Record approval
        cursor.execute("""
            INSERT INTO workflow_approvals (workflow_id, step_id, approver, decision, comments)
            VALUES (?, ?, ?, ?, ?)
        """, (workflow_id, step_id, approver, 'approved' if approved else 'rejected', comments))
        
        # Update step
        if approved:
            cursor.execute("""
                UPDATE workflow_steps
                SET approved_by = ?, approved_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (approver, step_id))
            
            # Resume workflow
            workflow = self.load_workflow(workflow_id)
            workflow.status = TaskStatus.RUNNING
            self._update_workflow(workflow)
            return True
        else:
            cursor.execute("""
                UPDATE workflow_steps
                SET status = 'failed', error = ?
                WHERE id = ?
            """, (f"Rejected by {approver}: {comments or 'No reason provided'}", step_id))
            
            workflow = self.load_workflow(workflow_id)
            workflow.status = TaskStatus.FAILED
            self._update_workflow(workflow)
            return False
    
    def _handle_retrieve(self, step: TaskStep, input_data: Dict, 
                        workflow: Workflow) -> Dict[str, Any]:
        """Handle retrieval step."""
        from .agents import answer_question_agentic
        
        question = input_data.get('question', '')
        if not question:
            return {'error': 'No question provided'}
        
        # This would integrate with your existing retrieval
        result = answer_question_agentic(
            self.conn, None, self.client, question, None
        )
        
        return {
            'retrieved_documents': result.get('sources', []),
            'answer': result.get('answer', ''),
            'confidence': result.get('confidence', 0.0)
        }
    
    def _handle_scan_pii(self, step: TaskStep, input_data: Dict,
                        workflow: Workflow) -> Dict[str, Any]:
        """Handle PII scanning step."""
        from .pii_scanner import PIIScanner
        
        text = input_data.get('text', '')
        if not text:
            return {'pii_found': [], 'is_clean': True}
        
        scanner = PIIScanner(self.conn, self.client, self.model)
        results = scanner.scan_text(text)
        
        return {
            'pii_found': results.get('entities', []),
            'is_clean': len(results.get('entities', [])) == 0,
            'risk_score': results.get('risk_score', 0.0)
        }
    
    def _handle_extract_info(self, step: TaskStep, input_data: Dict,
                            workflow: Workflow) -> Dict[str, Any]:
        """Handle information extraction step."""
        if not self.client:
            return {'extracted_data': {}}
        
        text = input_data.get('text', '')
        schema = input_data.get('schema', {})
        
        prompt = f"""Extract information from this text according to the schema:
Schema: {json.dumps(schema)}

Text: {text[:1000]}

Extracted data (JSON):"""
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 500}
        )
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response['response'], re.DOTALL)
            if json_match:
                extracted = json.loads(json_match.group())
                return {'extracted_data': extracted}
        except:
            pass
        
        return {'extracted_data': {}, 'raw_response': response['response']}
    
    def _handle_draft_document(self, step: TaskStep, input_data: Dict,
                              workflow: Workflow) -> Dict[str, Any]:
        """Handle document drafting step."""
        if not self.client:
            return {'draft': ''}
        
        template = input_data.get('template', '')
        data = input_data.get('data', {})
        style = input_data.get('style', 'professional')
        
        prompt = f"""Draft a document using this template and data.
Style: {style}

Template: {template}

Data: {json.dumps(data)}

Draft:"""
        
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={"temperature": 0.3, "num_predict": 1000}
        )
        
        return {
            'draft': response['response'],
            'word_count': len(response['response'].split())
        }
    
    def _handle_approval_step(self, step: TaskStep, input_data: Dict,
                             workflow: Workflow) -> Dict[str, Any]:
        """Handle approval step (placeholder - actual approval is external)."""
        return {'status': 'awaiting_approval'}
    
    def _handle_custom(self, step: TaskStep, input_data: Dict,
                      workflow: Workflow) -> Dict[str, Any]:
        """Handle custom task type."""
        # Allow custom logic via input_data['handler']
        custom_handler = input_data.get('handler')
        if callable(custom_handler):
            return custom_handler(step, input_data, workflow)
        
        return {'message': 'Custom step executed'}
    
    def _update_step(self, step: TaskStep):
        """Update step in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE workflow_steps
            SET status = ?, output_data = ?, error = ?, approved_by = ?
            WHERE id = ?
        """, (
            step.status.value,
            json.dumps(step.output_data),
            step.error,
            step.approved_by,
            step.id
        ))
        self.conn.commit()
    
    def _update_workflow(self, workflow: Workflow):
        """Update workflow in database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE workflows
            SET status = ?, current_step_index = ?, metadata = ?, updated_at = CURRENT_TIMESTAMP, completed_at = ?
            WHERE id = ?
        """, (
            workflow.status.value,
            workflow.current_step_index,
            json.dumps(workflow.metadata),
            workflow.metadata.get('completed_at'),
            workflow.id
        ))
        self.conn.commit()
    
    def load_workflow(self, workflow_id: str) -> Workflow:
        """Load workflow from database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, status, current_step_index, created_by, metadata
            FROM workflows
            WHERE id = ?
        """, (workflow_id,))
        
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = Workflow(
            id=row[0],
            name=row[1],
            description=row[2],
            status=TaskStatus(row[3]),
            current_step_index=row[4],
            created_by=row[5],
            metadata=json.loads(row[6]) if row[6] else {}
        )
        
        # Load steps
        cursor.execute("""
            SELECT id, task_type, description, status, input_data, output_data, 
                   error, requires_approval, approved_by
            FROM workflow_steps
            WHERE workflow_id = ?
            ORDER BY step_index
        """, (workflow_id,))
        
        for step_row in cursor.fetchall():
            step = TaskStep(
                id=step_row[0],
                task_type=TaskType(step_row[1]),
                description=step_row[2],
                status=TaskStatus(step_row[3]),
                input_data=json.loads(step_row[4]) if step_row[4] else {},
                output_data=json.loads(step_row[5]) if step_row[5] else {},
                error=step_row[6],
                requires_approval=step_row[7],
                approved_by=step_row[8]
            )
            workflow.steps.append(step)
        
        return workflow
    
    def get_workflows(self, status: Optional[TaskStatus] = None,
                     created_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of workflows."""
        cursor = self.conn.cursor()
        
        query = "SELECT id, name, description, status, created_by, created_at FROM workflows WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if created_by:
            query += " AND created_by = ?"
            params.append(created_by)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        
        return [{
            'id': row[0],
            'name': row[1],
            'description': row[2],
            'status': row[3],
            'created_by': row[4],
            'created_at': row[5]
        } for row in cursor.fetchall()]
