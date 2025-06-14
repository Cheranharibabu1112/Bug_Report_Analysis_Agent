import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.template import Template
import logging

class EmailService:
    def __init__(self, smtp_server="smtp.gmail.com", smtp_port=587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = None
        self.password = None
        self.logger = logging.getLogger(__name__)
        
    def configure_smtp(self, username, password):
        """Configure SMTP credentials"""
        self.username = username
        self.password = password
        
    def send_password_reset_email(self, recipient_email, reset_token):
        """Send password reset email to user"""
        try:
            # Create message
            message = MIMEMultipart("alternative")
            message["Subject"] = "Password Reset Request"
            message["From"] = self.username
            message["To"] = recipient_email
            
            # Create HTML content
            html_content = self._generate_reset_email_html(reset_token)
            html_part = MIMEText(html_content, "html")
            message.attach(html_part)
            
            # Send email
            return self._send_email(message)
            
        except Exception as e:
            self.logger.error(f"Error sending password reset email: {str(e)}")
            return False, f"Failed to send email: {str(e)}"
    
    def send_notification_email(self, recipient_email, subject, content):
        """Send general notification email"""
        try:
            message = MIMEMultipart()
            message["Subject"] = subject
            message["From"] = self.username
            message["To"] = recipient_email
            
            text_part = MIMEText(content, "plain")
            message.attach(text_part)
            
            return self._send_email(message)
            
        except Exception as e:
            self.logger.error(f"Error sending notification email: {str(e)}")
            return False, f"Failed to send email: {str(e)}"
    
    def _send_email(self, message):
        """Internal method to send email via SMTP"""
        try:
            # Create secure connection
            context = ssl.create_default_context()
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                
                # Check if credentials are configured
                if not self.username or not self.password:
                    raise Exception("SMTP credentials not configured")
                    
                server.login(self.username, self.password)
                
                # Send email
                text = message.as_string()
                server.sendmail(message["From"], message["To"], text)
                
            self.logger.info(f"Email sent successfully to {message['To']}")
            return True, "Email sent successfully"
            
        except smtplib.SMTPAuthenticationError:
            error_msg = "SMTP authentication failed - check credentials"
            self.logger.error(error_msg)
            return False, error_msg
            
        except smtplib.SMTPConnectError:
            error_msg = "Failed to connect to SMTP server"
            self.logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error sending email: {str(e)}"
            self.logger.error(error_msg)
            return False, error_msg
    
    def _generate_reset_email_html(self, reset_token):
        """Generate HTML content for password reset email"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Password Reset</title>
        </head>
        <body>
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2>Password Reset Request</h2>
                <p>You have requested a password reset for your account.</p>
                <p>Click the link below to reset your password:</p>
                <a href="https://yourapp.com/reset-password?token={token}" 
                   style="background-color: #4CAF50; color: white; padding: 10px 20px; 
                          text-decoration: none; border-radius: 4px;">
                    Reset Password
                </a>
                <p>If you did not request this reset, please ignore this email.</p>
                <p>This link will expire in 24 hours.</p>
            </div>
        </body>
        </html>
        """
        return html_template.format(token=reset_token)
    
    def test_connection(self):
        """Test SMTP connection and configuration"""
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                if self.username and self.password:
                    server.login(self.username, self.password)
                return True, "SMTP connection successful"
        except Exception as e:
            return False, f"SMTP connection failed: {str(e)}" 