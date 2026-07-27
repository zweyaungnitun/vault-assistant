"""
Contradiction Detective Agent
Automatically flags logical inconsistencies across documents.
"""
from typing import List, Dict, Any, Tuple, Optional
import sqlite3
from dataclasses import dataclass


@dataclass
class Claim:
    """Represents an extracted claim from text."""
    text: str
    subject: str
    predicate: str
    obj: str
    confidence: float
    source_document: str
    source_chunk_id: Optional[str] = None


@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    claim1: Claim
    claim2: Claim
    contradiction_type: str  # 'direct', 'temporal', 'numerical', 'logical'
    severity: str  # 'critical', 'high', 'medium', 'low'
    explanation: str
    confidence: float


class ContradictionDetector:
    """Detects contradictions and inconsistencies across documents."""
    
    def __init__(self, conn: sqlite3.Connection, client=None, model: str = "llama3:8b"):
        self.conn = conn
        self.client = client
        self.model = model
    
    def extract_claims(self, text: str, source_document: str, 
                      chunk_id: Optional[str] = None) -> List[Claim]:
        """Extract structured claims from text using small model."""
        if not self.client:
            return []
        
        prompt = f"""Extract factual claims from this text. For each claim, identify:
- Subject (who/what)
- Predicate (action/state)
- Object (target/value)
- Confidence (0.0-1.0)

Format as JSON array: [{{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}]

Text: {text[:500]}

Claims:"""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 500}
            )
            
            # Parse response (simplified - in production use proper JSON parsing)
            import json
            import re
            
            json_match = re.search(r'\[.*\]', response['response'], re.DOTALL)
            if json_match:
                claims_data = json.loads(json_match.group())
                return [
                    Claim(
                        text="",
                        subject=c.get('subject', ''),
                        predicate=c.get('predicate', ''),
                        obj=c.get('object', ''),
                        confidence=c.get('confidence', 0.5),
                        source_document=source_document,
                        source_chunk_id=chunk_id
                    )
                    for c in claims_data
                ]
        except Exception as e:
            print(f"Error extracting claims: {e}")
        
        return []
    
    def check_contradiction(self, claim1: Claim, claim2: Claim) -> Optional[Contradiction]:
        """Check if two claims contradict each other."""
        if not self.client:
            return None
        
        # Skip if subjects don't match
        if claim1.subject.lower() != claim2.subject.lower():
            return None
        
        prompt = f"""Do these two claims contradict each other?

Claim 1: {claim1.subject} {claim1.predicate} {claim1.obj}
Source: {claim1.source_document}

Claim 2: {claim2.subject} {claim2.predicate} {claim2.obj}
Source: {claim2.source_document}

Answer with:
1. YES/NO
2. Contradiction type: direct/temporal/numerical/logical
3. Severity: critical/high/medium/low
4. Brief explanation

Response:"""
        
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.0, "num_predict": 200}
            )
            
            resp_text = response['response'].strip()
            lines = resp_text.split('\n')
            
            if len(lines) >= 1 and lines[0].upper().startswith('YES'):
                contradiction_type = 'direct'
                severity = 'medium'
                explanation = ''
                
                for line in lines[1:]:
                    if 'type:' in line.lower():
                        contradiction_type = line.split(':')[1].strip()
                    elif 'severity:' in line.lower():
                        severity = line.split(':')[1].strip()
                    elif 'explanation:' in line.lower() or line.strip() and not any(x in line.lower() for x in ['type:', 'severity:']):
                        explanation = line.strip()
                
                # Calculate confidence based on input claims
                confidence = min(claim1.confidence, claim2.confidence) * 0.9
                
                return Contradiction(
                    claim1=claim1,
                    claim2=claim2,
                    contradiction_type=contradiction_type,
                    severity=severity,
                    explanation=explanation,
                    confidence=confidence
                )
        except Exception as e:
            print(f"Error checking contradiction: {e}")
        
        return None
    
    def scan_documents(self, document_ids: Optional[List[str]] = None) -> List[Contradiction]:
        """Scan documents for contradictions."""
        cursor = self.conn.cursor()
        
        # Get all chunks
        query = "SELECT id, document_id, content FROM chunks"
        if document_ids:
            placeholders = ','.join(['?' for _ in document_ids])
            query += f" WHERE document_id IN ({placeholders})"
            cursor.execute(query, document_ids)
        else:
            cursor.execute(query)
        
        all_claims = []
        
        # Extract claims from all chunks
        for row in cursor.fetchall():
            chunk_id, doc_id, content = row
            claims = self.extract_claims(content, doc_id, str(chunk_id))
            all_claims.extend(claims)
        
        # Check all pairs for contradictions
        contradictions = []
        for i, claim1 in enumerate(all_claims):
            for claim2 in all_claims[i+1:]:
                contradiction = self.check_contradiction(claim1, claim2)
                if contradiction:
                    contradictions.append(contradiction)
        
        # Sort by severity and confidence
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        contradictions.sort(key=lambda c: (severity_order.get(c.severity, 4), -c.confidence))
        
        return contradictions
    
    def store_contradiction(self, contradiction: Contradiction) -> int:
        """Store detected contradiction in database."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contradictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                claim1_subject TEXT,
                claim1_predicate TEXT,
                claim1_object TEXT,
                claim1_source TEXT,
                claim2_subject TEXT,
                claim2_predicate TEXT,
                claim2_object TEXT,
                claim2_source TEXT,
                contradiction_type TEXT,
                severity TEXT,
                explanation TEXT,
                confidence REAL,
                status TEXT DEFAULT 'open',
                resolved_at TIMESTAMP,
                resolved_by TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            INSERT INTO contradictions (
                claim1_subject, claim1_predicate, claim1_object, claim1_source,
                claim2_subject, claim2_predicate, claim2_object, claim2_source,
                contradiction_type, severity, explanation, confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            contradiction.claim1.subject,
            contradiction.claim1.predicate,
            contradiction.claim1.obj,
            contradiction.claim1.source_document,
            contradiction.claim2.subject,
            contradiction.claim2.predicate,
            contradiction.claim2.obj,
            contradiction.claim2.source_document,
            contradiction.contradiction_type,
            contradiction.severity,
            contradiction.explanation,
            contradiction.confidence
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_open_contradictions(self, severity: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get unresolved contradictions."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT id, claim1_subject, claim1_predicate, claim1_object, claim1_source,
                   claim2_subject, claim2_predicate, claim2_object, claim2_source,
                   contradiction_type, severity, explanation, confidence, created_at
            FROM contradictions
            WHERE status = 'open'
        """
        
        params = []
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        query += " ORDER BY CASE severity WHEN 'critical' THEN 1 WHEN 'high' THEN 2 WHEN 'medium' THEN 3 ELSE 4 END, confidence DESC"
        
        cursor.execute(query, params)
        
        return [{
            'id': row[0],
            'claim1': {
                'subject': row[1],
                'predicate': row[2],
                'object': row[3],
                'source': row[4]
            },
            'claim2': {
                'subject': row[5],
                'predicate': row[6],
                'object': row[7],
                'source': row[8]
            },
            'contradiction_type': row[9],
            'severity': row[10],
            'explanation': row[11],
            'confidence': row[12],
            'created_at': row[13]
        } for row in cursor.fetchall()]
    
    def resolve_contradiction(self, contradiction_id: int, resolved_by: str, 
                             resolution_notes: Optional[str] = None) -> bool:
        """Mark a contradiction as resolved."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            UPDATE contradictions
            SET status = 'resolved',
                resolved_at = CURRENT_TIMESTAMP,
                resolved_by = ?
            WHERE id = ?
        """, (resolved_by, contradiction_id))
        
        if resolution_notes:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contradiction_resolutions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    contradiction_id INTEGER,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (contradiction_id) REFERENCES contradictions(id)
                )
            """)
            
            cursor.execute("""
                INSERT INTO contradiction_resolutions (contradiction_id, notes)
                VALUES (?, ?)
            """, (contradiction_id, resolution_notes))
        
        self.conn.commit()
        return cursor.rowcount > 0
