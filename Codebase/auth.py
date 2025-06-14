import re
import hashlib
import time
from datetime import datetime, timedelta

class AuthenticationManager:
    def __init__(self):
        self.failed_attempts = {}
        self.session_timeout = 7200  # 2 hours in seconds
        
    def validate_password(self, password):
        """Validate password meets security requirements"""
        if len(password) < 8:
            return False, "Password must be at least 8 characters"
        
        # Fixed regex to properly handle special characters
        if not re.match(r'^[a-zA-Z0-9!@#$%^&*()_+\-=\[\]{};:,.<>?]+$', password):
            return False, "Password contains invalid characters"
            
        return True, "Password valid"
    
    def authenticate_user(self, username, password):
        """Authenticate user with username and password"""
        try:
            # Check for account lockout
            if self.is_account_locked(username):
                return False, "Account temporarily locked due to too many failed attempts"
            
            # Validate credentials
            if self.verify_credentials(username, password):
                self.reset_failed_attempts(username)
                return True, "Authentication successful"
            else:
                self.record_failed_attempt(username)
                return False, "Invalid username or password"
                
        except Exception as e:
            return False, f"Authentication error: {str(e)}"
    
    def verify_credentials(self, username, password):
        """Verify user credentials against database"""
        # Simplified credential check
        valid_users = {
            "admin": "admin123",
            "user1": "password123",
            "test@example.com": "test123!"
        }
        return valid_users.get(username) == password
    
    def is_account_locked(self, username):
        """Check if account is temporarily locked"""
        if username not in self.failed_attempts:
            return False
        
        attempts, last_attempt = self.failed_attempts[username]
        if attempts >= 5 and time.time() - last_attempt < 900:  # 15 minutes
            return True
        return False
    
    def record_failed_attempt(self, username):
        """Record a failed login attempt"""
        current_time = time.time()
        if username in self.failed_attempts:
            attempts, _ = self.failed_attempts[username]
            self.failed_attempts[username] = (attempts + 1, current_time)
        else:
            self.failed_attempts[username] = (1, current_time)
    
    def reset_failed_attempts(self, username):
        """Reset failed attempts for user"""
        if username in self.failed_attempts:
            del self.failed_attempts[username]
    
    def create_session(self, username):
        """Create user session with timeout"""
        session_data = {
            'username': username,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self.session_timeout),
            'session_id': hashlib.sha256(f"{username}{time.time()}".encode()).hexdigest()
        }
        return session_data
    
    def validate_session(self, session_id):
        """Validate if session is still active"""
        # This would typically check against a session store
        # For demo purposes, assuming session validation logic
        return True  # Simplified 