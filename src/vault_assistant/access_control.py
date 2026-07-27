"""
Granular Access Control (RBAC) Agent
Enterprise-grade security with pre-retrieval filtering and post-generation redaction.
"""
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sqlite3
import re


class AccessLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class User:
    """Represents a user with access levels."""
    id: str
    name: str
    access_levels: Set[AccessLevel]
    departments: Set[str]
    roles: Set[str]


@dataclass
class DocumentACL:
    """Access control list for a document."""
    document_id: str
    required_access_level: AccessLevel
    allowed_departments: Set[str]
    allowed_roles: Set[str]
    explicitly_denied_users: Set[str]


class AccessControlEngine:
    """Manages role-based access control for documents and queries."""
    
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._init_schema()
        self.user_cache: Dict[str, User] = {}
    
    def _init_schema(self):
        """Initialize RBAC tables."""
        cursor = self.conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # User access levels
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_access_levels (
                user_id TEXT NOT NULL,
                access_level TEXT NOT NULL,
                PRIMARY KEY (user_id, access_level),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # User departments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_departments (
                user_id TEXT NOT NULL,
                department TEXT NOT NULL,
                PRIMARY KEY (user_id, department),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # User roles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_roles (
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (user_id, role),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Document ACLs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_acls (
                document_id TEXT PRIMARY KEY,
                required_access_level TEXT DEFAULT 'internal',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (document_id) REFERENCES documents(id)
            )
        """)
        
        # Document allowed departments
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_allowed_departments (
                document_id TEXT NOT NULL,
                department TEXT NOT NULL,
                PRIMARY KEY (document_id, department),
                FOREIGN KEY (document_id) REFERENCES document_acls(document_id)
            )
        """)
        
        # Document allowed roles
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_allowed_roles (
                document_id TEXT NOT NULL,
                role TEXT NOT NULL,
                PRIMARY KEY (document_id, role),
                FOREIGN KEY (document_id) REFERENCES document_acls(document_id)
            )
        """)
        
        # Explicitly denied users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_denied_users (
                document_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                PRIMARY KEY (document_id, user_id),
                FOREIGN KEY (document_id) REFERENCES document_acls(document_id),
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        # Query audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL,
                documents_accessed JSON,
                documents_filtered INTEGER,
                response_redacted BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        
        self.conn.commit()
    
    def create_user(self, user_id: str, name: str,
                   access_levels: Optional[List[str]] = None,
                   departments: Optional[List[str]] = None,
                   roles: Optional[List[str]] = None) -> User:
        """Create or update a user."""
        cursor = self.conn.cursor()
        
        # Upsert user
        cursor.execute("""
            INSERT INTO users (id, name) VALUES (?, ?)
            ON CONFLICT(id) DO UPDATE SET name = excluded.name
        """, (user_id, name))
        
        # Clear existing permissions
        cursor.execute("DELETE FROM user_access_levels WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_departments WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_roles WHERE user_id = ?", (user_id,))
        
        # Add access levels
        if access_levels:
            for level in access_levels:
                cursor.execute(
                    "INSERT INTO user_access_levels (user_id, access_level) VALUES (?, ?)",
                    (user_id, level)
                )
        
        # Add departments
        if departments:
            for dept in departments:
                cursor.execute(
                    "INSERT INTO user_departments (user_id, department) VALUES (?, ?)",
                    (user_id, dept)
                )
        
        # Add roles
        if roles:
            for role in roles:
                cursor.execute(
                    "INSERT INTO user_roles (user_id, role) VALUES (?, ?)",
                    (user_id, role)
                )
        
        self.conn.commit()
        
        # Invalidate cache
        if user_id in self.user_cache:
            del self.user_cache[user_id]
        
        return self.get_user(user_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user with permissions."""
        if user_id in self.user_cache:
            return self.user_cache[user_id]
        
        cursor = self.conn.cursor()
        
        # Get user info
        cursor.execute("SELECT id, name FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if not row:
            return None
        
        # Get access levels
        cursor.execute(
            "SELECT access_level FROM user_access_levels WHERE user_id = ?",
            (user_id,)
        )
        access_levels = {AccessLevel(r[0]) for r in cursor.fetchall()}
        
        # Get departments
        cursor.execute(
            "SELECT department FROM user_departments WHERE user_id = ?",
            (user_id,)
        )
        departments = {r[0] for r in cursor.fetchall()}
        
        # Get roles
        cursor.execute(
            "SELECT role FROM user_roles WHERE user_id = ?",
            (user_id,)
        )
        roles = {r[0] for r in cursor.fetchall()}
        
        user = User(
            id=row[0],
            name=row[1],
            access_levels=access_levels,
            departments=departments,
            roles=roles
        )
        
        self.user_cache[user_id] = user
        return user
    
    def set_document_acl(self, document_id: str,
                        required_access_level: str = "internal",
                        allowed_departments: Optional[List[str]] = None,
                        allowed_roles: Optional[List[str]] = None,
                        denied_users: Optional[List[str]] = None):
        """Set access control list for a document."""
        cursor = self.conn.cursor()
        
        # Upsert ACL
        cursor.execute("""
            INSERT INTO document_acls (document_id, required_access_level)
            VALUES (?, ?)
            ON CONFLICT(document_id) DO UPDATE SET required_access_level = excluded.required_access_level
        """, (document_id, required_access_level))
        
        # Clear existing permissions
        cursor.execute(
            "DELETE FROM document_allowed_departments WHERE document_id = ?",
            (document_id,)
        )
        cursor.execute(
            "DELETE FROM document_allowed_roles WHERE document_id = ?",
            (document_id,)
        )
        cursor.execute(
            "DELETE FROM document_denied_users WHERE document_id = ?",
            (document_id,)
        )
        
        # Add allowed departments
        if allowed_departments:
            for dept in allowed_departments:
                cursor.execute(
                    "INSERT INTO document_allowed_departments (document_id, department) VALUES (?, ?)",
                    (document_id, dept)
                )
        
        # Add allowed roles
        if allowed_roles:
            for role in allowed_roles:
                cursor.execute(
                    "INSERT INTO document_allowed_roles (document_id, role) VALUES (?, ?)",
                    (document_id, role)
                )
        
        # Add denied users
        if denied_users:
            for user_id in denied_users:
                cursor.execute(
                    "INSERT INTO document_denied_users (document_id, user_id) VALUES (?, ?)",
                    (document_id, user_id)
                )
        
        self.conn.commit()
    
    def can_access_document(self, user: User, document_id: str) -> bool:
        """Check if user can access a document."""
        cursor = self.conn.cursor()
        
        # Get document ACL
        cursor.execute("""
            SELECT required_access_level FROM document_acls WHERE document_id = ?
        """, (document_id,))
        row = cursor.fetchone()
        
        if not row:
            # No ACL means default to internal
            required_level = AccessLevel.INTERNAL
        else:
            required_level = AccessLevel(row[0])
        
        # Check if user is explicitly denied
        cursor.execute("""
            SELECT COUNT(*) FROM document_denied_users
            WHERE document_id = ? AND user_id = ?
        """, (document_id, user.id))
        if cursor.fetchone()[0] > 0:
            return False
        
        # Check access level
        if required_level not in user.access_levels:
            # Allow if user has higher clearance
            level_order = {
                AccessLevel.PUBLIC: 0,
                AccessLevel.INTERNAL: 1,
                AccessLevel.CONFIDENTIAL: 2,
                AccessLevel.RESTRICTED: 3
            }
            user_max_level = max(user.access_levels, key=lambda x: level_order[x], default=AccessLevel.PUBLIC)
            if level_order[user_max_level] < level_order[required_level]:
                return False
        
        # Check department restriction (if any defined)
        cursor.execute("""
            SELECT department FROM document_allowed_departments WHERE document_id = ?
        """, (document_id,))
        allowed_depts = {r[0] for r in cursor.fetchall()}
        
        if allowed_depts and not user.departments.intersection(allowed_depts):
            return False
        
        # Check role restriction (if any defined)
        cursor.execute("""
            SELECT role FROM document_allowed_roles WHERE document_id = ?
        """, (document_id,))
        allowed_roles = {r[0] for r in cursor.fetchall()}
        
        if allowed_roles and not user.roles.intersection(allowed_roles):
            return False
        
        return True
    
    def filter_documents_by_access(self, user: User, 
                                   document_ids: List[str]) -> List[str]:
        """Filter document list to only accessible documents."""
        return [doc_id for doc_id in document_ids if self.can_access_document(user, doc_id)]
    
    def redact_response(self, response: str, user: User, 
                       accessible_docs: Set[str]) -> str:
        """Redact information from inaccessible sources in response."""
        # Simple pattern-based redaction
        # In production, this would use NER to identify sensitive info
        
        patterns = [
            (r'\[Source: ([^\]]+)\]', lambda m: f"[Source: REDACTED]" if m.group(1) not in accessible_docs else m.group(0)),
            (r'(Confidential|Restricted|Secret):\s*([^\n]+)', lambda m: f"REDACTED ({m.group(1)} information)"),
        ]
        
        redacted = response
        for pattern, replacer in patterns:
            redacted = re.sub(pattern, replacer, redacted, flags=re.IGNORECASE)
        
        return redacted
    
    def log_query(self, user_id: str, query: str, 
                 documents_accessed: List[str],
                 documents_filtered: int,
                 response_redacted: bool = False):
        """Log query for audit purposes."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO query_audit_log (
                user_id, query, documents_accessed, documents_filtered, response_redacted
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            user_id, query, 
            str(documents_accessed), 
            documents_filtered,
            response_redacted
        ))
        
        self.conn.commit()
    
    def get_audit_log(self, user_id: Optional[str] = None,
                     limit: int = 100) -> List[Dict[str, Any]]:
        """Get query audit log."""
        cursor = self.conn.cursor()
        
        query = """
            SELECT id, user_id, query, documents_accessed, documents_filtered, 
                   response_redacted, created_at
            FROM query_audit_log
            WHERE 1=1
        """
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        
        return [{
            'id': row[0],
            'user_id': row[1],
            'query': row[2],
            'documents_accessed': eval(row[3]) if row[3] else [],
            'documents_filtered': row[4],
            'response_redacted': row[5],
            'created_at': row[6]
        } for row in cursor.fetchall()]
