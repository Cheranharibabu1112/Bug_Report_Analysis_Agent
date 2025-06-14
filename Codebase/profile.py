import json
import sqlite3
from datetime import datetime
from typing import Dict, Optional, List

class UserProfileManager:
    """Manages user profile data and operations"""
    
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the user profile database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        email VARCHAR(100) UNIQUE NOT NULL,
                        first_name VARCHAR(50),
                        last_name VARCHAR(50),
                        profile_data TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                conn.commit()
        except Exception as e:
            print(f"Database initialization error: {e}")
    
    def create_profile(self, username: str, email: str, first_name: str = "", 
                      last_name: str = "", additional_data: Dict = None) -> bool:
        """Create a new user profile"""
        try:
            profile_data = json.dumps(additional_data or {})
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO user_profiles 
                    (username, email, first_name, last_name, profile_data)
                    VALUES (?, ?, ?, ?, ?)
                """, (username, email, first_name, last_name, profile_data))
                conn.commit()
                return True
                
        except sqlite3.IntegrityError:
            return False  # Username or email already exists
        except Exception as e:
            print(f"Profile creation error: {e}")
            return False
    
    def get_profile(self, username: str) -> Optional[Dict]:
        """Retrieve user profile by username"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, first_name, last_name, 
                           profile_data, created_at, updated_at
                    FROM user_profiles 
                    WHERE username = ?
                """, (username,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'first_name': row[3],
                        'last_name': row[4],
                        'profile_data': json.loads(row[5] or '{}'),
                        'created_at': row[6],
                        'updated_at': row[7]
                    }
                return None
                
        except Exception as e:
            print(f"Profile retrieval error: {e}")
            return None
    
    def update_profile(self, username: str, updates: Dict) -> bool:
        """Update user profile information"""
        try:
            # Build dynamic update query
            update_fields = []
            update_values = []
            
            allowed_fields = ['email', 'first_name', 'last_name', 'profile_data']
            
            for field, value in updates.items():
                if field in allowed_fields:
                    if field == 'profile_data':
                        value = json.dumps(value)
                    update_fields.append(f"{field} = ?")
                    update_values.append(value)
            
            if not update_fields:
                return False  # No valid fields to update
            
            # Add updated_at timestamp
            update_fields.append("updated_at = ?")
            update_values.append(datetime.now().isoformat())
            
            # Add username for WHERE clause
            update_values.append(username)
            
            query = f"""
                UPDATE user_profiles 
                SET {', '.join(update_fields)}
                WHERE username = ?
            """
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, update_values)
                conn.commit()
                
                return cursor.rowcount > 0  # True if row was updated
                
        except sqlite3.IntegrityError:
            return False  # Constraint violation (e.g., duplicate email)
        except Exception as e:
            print(f"Profile update error: {e}")
            return False
    
    def delete_profile(self, username: str) -> bool:
        """Delete user profile"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM user_profiles WHERE username = ?", (username,))
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            print(f"Profile deletion error: {e}")
            return False
    
    def list_profiles(self, limit: int = 50, offset: int = 0) -> List[Dict]:
        """List user profiles with pagination"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, first_name, last_name, created_at
                    FROM user_profiles 
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                """, (limit, offset))
                
                profiles = []
                for row in cursor.fetchall():
                    profiles.append({
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'first_name': row[3],
                        'last_name': row[4],
                        'created_at': row[5]
                    })
                
                return profiles
                
        except Exception as e:
            print(f"Profile listing error: {e}")
            return []
    
    def search_profiles(self, search_term: str) -> List[Dict]:
        """Search profiles by username, email, or name"""
        try:
            search_pattern = f"%{search_term}%"
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, username, email, first_name, last_name, created_at
                    FROM user_profiles 
                    WHERE username LIKE ? OR email LIKE ? 
                       OR first_name LIKE ? OR last_name LIKE ?
                    ORDER BY username
                """, (search_pattern, search_pattern, search_pattern, search_pattern))
                
                profiles = []
                for row in cursor.fetchall():
                    profiles.append({
                        'id': row[0],
                        'username': row[1],
                        'email': row[2],
                        'first_name': row[3],
                        'last_name': row[4],
                        'created_at': row[5]
                    })
                
                return profiles
                
        except Exception as e:
            print(f"Profile search error: {e}")
            return []

# API endpoints for profile management
def handle_profile_update_api(request_data: Dict) -> Dict:
    """Handle API request for profile updates"""
    try:
        username = request_data.get('username')
        updates = request_data.get('updates', {})
        
        if not username:
            return {'error': 'Username is required', 'status': 400}
        
        profile_manager = UserProfileManager()
        
        # Check if profile exists
        if not profile_manager.get_profile(username):
            return {'error': 'Profile not found', 'status': 404}
        
        # Attempt to update profile
        if profile_manager.update_profile(username, updates):
            return {'message': 'Profile updated successfully', 'status': 200}
        else:
            return {'error': 'Failed to update profile', 'status': 500}
            
    except Exception as e:
        return {'error': f'Internal server error: {str(e)}', 'status': 500} 