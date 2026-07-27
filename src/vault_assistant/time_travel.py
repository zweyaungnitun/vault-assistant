"""
Time-Travel Query Engine
Tracks policy evolution and detects version conflicts for compliance audits.
"""
from datetime import datetime
from typing import List, Dict, Any, Optional
import sqlite3


class TimeTravelEngine:
    """Manages document versioning and historical queries."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()
    
    def _init_schema(self):
        """Initialize version tracking tables."""
        cursor = self.conn.cursor()
        
        # Document versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                content TEXT,
                metadata JSON,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1,
                UNIQUE(document_id, version)
            )
        """)
        
        # Version relationships (for tracking changes)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS version_changes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                from_version_id INTEGER,
                to_version_id INTEGER,
                change_type TEXT, -- 'modified', 'deleted', 'split', 'merged'
                changed_fields JSON,
                detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_version_id) REFERENCES document_versions(id),
                FOREIGN KEY (to_version_id) REFERENCES document_versions(id)
            )
        """)
        
        # Policy timeline for audit trails
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS policy_timeline (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                policy_id TEXT NOT NULL,
                effective_date DATE,
                expiration_date DATE,
                superseded_by TEXT,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.commit()
    
    def add_version(self, document_id: str, content: str, metadata: Dict[str, Any], 
                    is_active: bool = True) -> int:
        """Add a new version of a document."""
        import hashlib
        
        cursor = self.conn.cursor()
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Get current max version
        cursor.execute(
            "SELECT COALESCE(MAX(version), 0) FROM document_versions WHERE document_id = ?",
            (document_id,)
        )
        current_version = cursor.fetchone()[0]
        new_version = current_version + 1
        
        # Deactivate previous version if this is the new active one
        if is_active and new_version > 1:
            cursor.execute(
                "UPDATE document_versions SET is_active = 0 WHERE document_id = ?",
                (document_id,)
            )
        
        cursor.execute("""
            INSERT INTO document_versions (document_id, version, content_hash, content, metadata, is_active)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (document_id, new_version, content_hash, content, str(metadata), is_active))
        
        version_id = cursor.lastrowid
        self.conn.commit()
        
        return version_id
    
    def get_version(self, document_id: str, version: Optional[int] = None, 
                    as_of_date: Optional[datetime] = None) -> Optional[Dict[str, Any]]:
        """Retrieve a specific version or version as of a date."""
        cursor = self.conn.cursor()
        
        if version is not None:
            cursor.execute("""
                SELECT id, document_id, version, content, metadata, created_at, is_active
                FROM document_versions
                WHERE document_id = ? AND version = ?
            """, (document_id, version))
        elif as_of_date is not None:
            cursor.execute("""
                SELECT id, document_id, version, content, metadata, created_at, is_active
                FROM document_versions
                WHERE document_id = ? AND created_at <= ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (document_id, as_of_date.isoformat()))
        else:
            # Get latest active version
            cursor.execute("""
                SELECT id, document_id, version, content, metadata, created_at, is_active
                FROM document_versions
                WHERE document_id = ? AND is_active = 1
                ORDER BY version DESC
                LIMIT 1
            """, (document_id,))
        
        row = cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'document_id': row[1],
                'version': row[2],
                'content': row[3],
                'metadata': eval(row[4]) if row[4] else {},
                'created_at': row[5],
                'is_active': row[6]
            }
        return None
    
    def get_version_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get full version history for a document."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, document_id, version, content_hash, metadata, created_at, is_active
            FROM document_versions
            WHERE document_id = ?
            ORDER BY version ASC
        """, (document_id,))
        
        return [{
            'id': row[0],
            'document_id': row[1],
            'version': row[2],
            'content_hash': row[3],
            'metadata': eval(row[4]) if row[4] else {},
            'created_at': row[5],
            'is_active': row[6]
        } for row in cursor.fetchall()]
    
    def detect_conflicts(self, document_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Detect version conflicts and inconsistencies."""
        cursor = self.conn.cursor()
        conflicts = []
        
        # Check for multiple active versions
        query = """
            SELECT document_id, COUNT(*) as active_count
            FROM document_versions
            WHERE is_active = 1
        """
        if document_ids:
            placeholders = ','.join(['?' for _ in document_ids])
            query += f" AND document_id IN ({placeholders})"
        
        query += " GROUP BY document_id HAVING active_count > 1"
        
        if document_ids:
            cursor.execute(query, document_ids)
        else:
            cursor.execute(query)
        
        for row in cursor.fetchall():
            conflicts.append({
                'type': 'multiple_active_versions',
                'document_id': row[0],
                'active_count': row[1],
                'severity': 'high',
                'message': f"Document {row[0]} has {row[1]} active versions"
            })
        
        # Check for policy timeline conflicts
        cursor.execute("""
            SELECT policy_id, effective_date, expiration_date, superseded_by
            FROM policy_timeline
            WHERE expiration_date IS NOT NULL AND superseded_by IS NULL
        """)
        
        for row in cursor.fetchall():
            conflicts.append({
                'type': 'orphaned_policy',
                'policy_id': row[0],
                'effective_date': row[1],
                'expiration_date': row[2],
                'severity': 'medium',
                'message': f"Policy {row[0]} expired on {row[2]} but has no successor"
            })
        
        return conflicts
    
    def query_historical(self, question: str, as_of_date: datetime, 
                        index, client, cfg) -> Dict[str, Any]:
        """Answer a question as of a specific historical date."""
        from .agents import answer_question_agentic
        
        # This would need to be integrated with the retrieval system
        # to only consider documents active as_of_date
        result = answer_question_agentic(
            self.conn, index, client, question, cfg,
            temporal_context={'as_of_date': as_of_date}
        )
        
        result['temporal_metadata'] = {
            'query_date': datetime.now().isoformat(),
            'historical_date': as_of_date.isoformat(),
            'documents_considered': 'versions active as of historical date'
        }
        
        return result
