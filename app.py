"""
🐞 Enhanced Bug Report Analysis Agent
=====================================
A comprehensive RAG-based system for analyzing bug reports, finding similar issues,
and suggesting fixes with evaluation metrics for retrieval relevance and usefulness.
"""

import os
import pandas as pd
import numpy as np
import gradio as gr
import sqlite3
import json
import ast
import re
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

# Core RAG and ML imports
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from fuzzywuzzy import fuzz, process

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Evaluation metrics
from rouge_score import rouge_scorer
import difflib

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BugReportRAG:
    """Enhanced RAG system for bug report analysis"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.bug_index = None
        self.code_index = None
        self.bug_data = None
        self.code_data = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def load_and_index_data(self):
        """Load and index bug reports and code files"""
        logger.info("Loading and indexing data...")
        
        # Load bug reports
        self._load_bug_reports()
        
        # Load and process code files
        self._load_code_files()
        
        # Create FAISS indices
        self._create_faiss_indices()
        
        logger.info("Data loading and indexing completed")
    
    def _load_bug_reports(self):
        """Load and process bug reports from CSV"""
        try:
            df = pd.read_csv("bug_reports.csv")
            
            # Create comprehensive text representation for each bug
            bug_texts = []
            bug_metadata = []
            
            for _, row in df.iterrows():
                # Combine relevant fields for better semantic search
                text_parts = [
                    f"Title: {row.get('title', '')}",
                    f"Description: {row.get('description', '')}",
                    f"Component: {row.get('component', '')}",
                    f"Severity: {row.get('severity', '')}",
                    f"Status: {row.get('status', '')}",
                ]
                
                if pd.notna(row.get('fix_description')):
                    text_parts.append(f"Fix: {row['fix_description']}")
                
                bug_text = " | ".join(text_parts)
                bug_texts.append(bug_text)
                
                # Store metadata
                metadata = {
                    'id': row.get('id', ''),
                    'title': row.get('title', ''),
                    'description': row.get('description', ''),
                    'severity': row.get('severity', ''),
                    'status': row.get('status', ''),
                    'component': row.get('component', ''),
                    'fix_description': row.get('fix_description', ''),
                    'related_files': row.get('related_files', ''),
                    'created_date': row.get('created_date', ''),
                    'resolved_date': row.get('resolved_date', ''),
                }
                bug_metadata.append(metadata)
            
            self.bug_data = {
                'texts': bug_texts,
                'metadata': bug_metadata
            }
            
        except Exception as e:
            logger.error(f"Error loading bug reports: {e}")
            self.bug_data = {'texts': [], 'metadata': []}
    
    def _load_code_files(self):
        """Load and process code files"""
        code_texts = []
        code_metadata = []
        
        for root, dirs, files in os.walk("codebase"):
            for file in files:
                if file.endswith(('.py', '.js', '.html', '.css')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        # Split large files into chunks
                        if len(content) > 1000:
                            chunks = self.text_splitter.split_text(content)
                            for i, chunk in enumerate(chunks):
                                code_texts.append(f"File: {file} | {chunk}")
                                code_metadata.append({
                                    'file_path': file_path,
                                    'file_name': file,
                                    'chunk_index': i,
                                    'total_chunks': len(chunks)
                                })
                        else:
                            code_texts.append(f"File: {file} | {content}")
                            code_metadata.append({
                                'file_path': file_path,
                                'file_name': file,
                                'chunk_index': 0,
                                'total_chunks': 1
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        self.code_data = {
            'texts': code_texts,
            'metadata': code_metadata
        }
    
    def _create_faiss_indices(self):
        """Create FAISS indices for efficient similarity search"""
        # Create bug report index
        if self.bug_data['texts']:
            bug_embeddings = self.embedding_model.encode(self.bug_data['texts'])
            self.bug_index = faiss.IndexFlatIP(bug_embeddings.shape[1])
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(bug_embeddings)
            self.bug_index.add(bug_embeddings.astype('float32'))
        
        # Create code index
        if self.code_data['texts']:
            code_embeddings = self.embedding_model.encode(self.code_data['texts'])
            self.code_index = faiss.IndexFlatIP(code_embeddings.shape[1])
            faiss.normalize_L2(code_embeddings)
            self.code_index.add(code_embeddings.astype('float32'))
    
    def search_similar_bugs(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar bug reports"""
        if not self.bug_index or not self.bug_data['texts']:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.bug_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.bug_data['metadata']):
                result = self.bug_data['metadata'][idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def search_relevant_code(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant code sections"""
        if not self.code_index or not self.code_data['texts']:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.code_index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.code_data['metadata']):
                result = self.code_data['metadata'][idx].copy()
                result['similarity_score'] = float(score)
                result['code_text'] = self.code_data['texts'][idx]
                results.append(result)
        
        return results

class BugAnalysisEvaluator:
    """Evaluate the quality and relevance of bug analysis results"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate_retrieval_relevance(self, query: str, results: List[Dict]) -> Dict:
        """Evaluate how relevant retrieved results are to the query"""
        if not results:
            return {
                'average_similarity': 0.0,
                'relevance_score': 0.0,
                'result_count': 0
            }
        
        # Calculate average similarity score
        similarity_scores = [r.get('similarity_score', 0.0) for r in results]
        average_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
        
        # Calculate semantic relevance using text similarity
        query_lower = query.lower()
        relevance_scores = []
        
        for result in results:
            # Combine title and description for relevance calculation
            result_text = f"{result.get('title', '')} {result.get('description', '')}"
            relevance_score = fuzz.partial_ratio(query_lower, result_text.lower()) / 100.0
            relevance_scores.append(relevance_score)
        
        relevance_score = np.mean(relevance_scores) if relevance_scores else 0.0
        
        return {
            'average_similarity': float(average_similarity),
            'relevance_score': float(relevance_score),
            'result_count': len(results),
            'individual_scores': similarity_scores
        }
    
    def evaluate_suggestion_usefulness(self, query: str, suggestions: str) -> Dict:
        """Evaluate the usefulness of generated suggestions"""
        if not suggestions or not query:
            return {
                'completeness_score': 0.0,
                'specificity_score': 0.0,
                'actionability_score': 0.0,
                'overall_usefulness': 0.0
            }
        
        # Completeness: How well suggestions address the query
        rouge_scores = self.rouge_scorer.score(query.lower(), suggestions.lower())
        completeness_score = rouge_scores['rougeL'].fmeasure
        
        # Specificity: Presence of specific technical terms, file names, functions
        specificity_indicators = [
            r'\b\w+\.py\b',  # Python files
            r'\bdef \w+\b',   # Function definitions
            r'\bclass \w+\b', # Class definitions
            r'\b\w+\(\)',     # Function calls
            r'\bfix\b|\bupdate\b|\bchange\b|\bmodify\b',  # Action words
        ]
        
        specificity_count = sum(len(re.findall(pattern, suggestions.lower())) 
                               for pattern in specificity_indicators)
        specificity_score = min(specificity_count / 5.0, 1.0)  # Normalize to 0-1
        
        # Actionability: Presence of actionable steps
        actionable_phrases = [
            'check', 'verify', 'update', 'modify', 'fix', 'add', 'remove',
            'ensure', 'validate', 'test', 'debug', 'implement', 'configure'
        ]
        
        actionability_count = sum(1 for phrase in actionable_phrases 
                                 if phrase in suggestions.lower())
        actionability_score = min(actionability_count / 5.0, 1.0)
        
        # Overall usefulness (weighted average)
        overall_usefulness = (
            0.3 * completeness_score +
            0.4 * specificity_score +
            0.3 * actionability_score
        )
        
        return {
            'completeness_score': float(completeness_score),
            'specificity_score': float(specificity_score),
            'actionability_score': float(actionability_score),
            'overall_usefulness': float(overall_usefulness)
        }

class FixSuggestionEngine:
    """Generate intelligent fix suggestions based on analysis"""
    
    def __init__(self):
        self.common_fixes = {
            'authentication': [
                "Check password validation regex patterns",
                "Verify session management configuration",
                "Ensure proper error handling in login flow",
                "Review authentication middleware setup"
            ],
            'database': [
                "Check database connection pooling settings",
                "Review query optimization and indexing",
                "Verify transaction handling and rollbacks",
                "Check for connection timeout configurations"
            ],
            'email': [
                "Verify SMTP server configuration",
                "Check email template rendering",
                "Ensure email credentials are properly set",
                "Review email queue processing"
            ],
            'ui': [
                "Check JavaScript event listeners",
                "Verify CSS styling and responsive design",
                "Review form validation logic",
                "Ensure proper DOM element targeting"
            ]
        }
    
    def generate_suggestions(self, query: str, similar_bugs: List[Dict], 
                           relevant_code: List[Dict]) -> str:
        """Generate fix suggestions based on analysis"""
        suggestions = []
        
        # Add context-based suggestions
        suggestions.append("## 🔍 Analysis Summary")
        suggestions.append(f"Based on the query: '{query}'")
        suggestions.append("")
        
        # Add similar bug insights
        if similar_bugs:
            suggestions.append("## 🪲 Similar Issues Found")
            for i, bug in enumerate(similar_bugs[:3], 1):
                status = bug.get('status', 'Unknown')
                severity = bug.get('severity', 'Unknown')
                suggestions.append(f"{i}. **{bug.get('title', 'Untitled')}** (Status: {status}, Severity: {severity})")
                
                if bug.get('fix_description'):
                    suggestions.append(f"   - Previous fix: {bug['fix_description']}")
                suggestions.append("")
        
        # Add code analysis
        if relevant_code:
            suggestions.append("## 💻 Relevant Code Sections")
            for i, code in enumerate(relevant_code[:3], 1):
                file_name = code.get('file_name', 'Unknown file')
                suggestions.append(f"{i}. **{file_name}** (Similarity: {code.get('similarity_score', 0):.2f})")
                suggestions.append("")
        
        # Add specific fix suggestions based on component analysis
        component_suggestions = self._get_component_suggestions(query, similar_bugs)
        if component_suggestions:
            suggestions.append("## 🛠️ Suggested Actions")
            for suggestion in component_suggestions:
                suggestions.append(f"- {suggestion}")
            suggestions.append("")
        
        # Add general debugging steps
        suggestions.append("## 🔧 General Debugging Steps")
        suggestions.extend([
            "- Review error logs and stack traces",
            "- Test in different environments (dev/staging/prod)",
            "- Check recent code changes in related files",
            "- Verify configuration settings",
            "- Run relevant test suites",
            "- Consider rollback if issue is critical"
        ])
        
        return "\n".join(suggestions)
    
    def _get_component_suggestions(self, query: str, similar_bugs: List[Dict]) -> List[str]:
        """Get component-specific suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # Identify likely component based on keywords and similar bugs
        components = [bug.get('component', '').lower() for bug in similar_bugs]
        
        # Keyword-based component detection
        if any(keyword in query_lower for keyword in ['login', 'auth', 'password', 'session']):
            suggestions.extend(self.common_fixes.get('authentication', []))
        
        if any(keyword in query_lower for keyword in ['database', 'db', 'query', 'connection']):
            suggestions.extend(self.common_fixes.get('database', []))
        
        if any(keyword in query_lower for keyword in ['email', 'smtp', 'mail', 'notification']):
            suggestions.extend(self.common_fixes.get('email', []))
        
        if any(keyword in query_lower for keyword in ['button', 'form', 'ui', 'interface', 'display']):
            suggestions.extend(self.common_fixes.get('ui', []))
        
        # Component-based suggestions from similar bugs
        for component in components:
            if component and component in self.common_fixes:
                suggestions.extend(self.common_fixes[component])
        
        return list(set(suggestions))  # Remove duplicates

# Initialize the RAG system and other components
rag_system = BugReportRAG()
evaluator = BugAnalysisEvaluator()
suggestion_engine = FixSuggestionEngine()

# Load and index data on startup
rag_system.load_and_index_data()

def analyze_bug_report(query: str) -> Tuple[str, str, str, str]:
    """Main function to analyze bug reports"""
    try:
        if not query.strip():
            return "Please enter a bug description", "", "", ""
        
        logger.info(f"Analyzing query: {query}")
        
        # Search for similar bugs and relevant code
        similar_bugs = rag_system.search_similar_bugs(query, k=5)
        relevant_code = rag_system.search_relevant_code(query, k=5)
        
        # Generate suggestions
        suggestions = suggestion_engine.generate_suggestions(query, similar_bugs, relevant_code)
        
        # Evaluate results
        bug_evaluation = evaluator.evaluate_retrieval_relevance(query, similar_bugs)
        suggestion_evaluation = evaluator.evaluate_suggestion_usefulness(query, suggestions)
        
        # Format similar bugs output
        similar_bugs_output = format_similar_bugs(similar_bugs, bug_evaluation)
        
        # Format relevant code output
        relevant_code_output = format_relevant_code(relevant_code)
        
        # Format evaluation metrics
        evaluation_output = format_evaluation_metrics(bug_evaluation, suggestion_evaluation)
        
        return similar_bugs_output, relevant_code_output, suggestions, evaluation_output
        
    except Exception as e:
        logger.error(f"Error analyzing bug report: {e}")
        return f"Error: {str(e)}", "", "", ""

def format_similar_bugs(bugs: List[Dict], evaluation: Dict) -> str:
    """Format similar bugs for display"""
    if not bugs:
        return "No similar bugs found in the database."
    
    output = [f"## 🔍 Found {len(bugs)} Similar Bug Reports"]
    output.append(f"**Relevance Score: {evaluation['relevance_score']:.2f}/1.0**")
    output.append(f"**Average Similarity: {evaluation['average_similarity']:.2f}/1.0**")
    output.append("")
    
    for i, bug in enumerate(bugs, 1):
        output.append(f"### {i}. {bug.get('title', 'Untitled Bug')}")
        output.append(f"**ID:** {bug.get('id', 'N/A')} | **Severity:** {bug.get('severity', 'N/A')} | **Status:** {bug.get('status', 'N/A')}")
        output.append(f"**Similarity:** {bug.get('similarity_score', 0):.3f}")
        output.append(f"**Component:** {bug.get('component', 'N/A')}")
        output.append("")
        output.append(f"**Description:** {bug.get('description', 'No description available')}")
        
        if bug.get('fix_description'):
            output.append(f"**Previous Fix:** {bug['fix_description']}")
        
        if bug.get('related_files'):
            output.append(f"**Related Files:** {bug['related_files']}")
        
        output.append("---")
    
    return "\n".join(output)

def format_relevant_code(code_results: List[Dict]) -> str:
    """Format relevant code sections for display"""
    if not code_results:
        return "No relevant code sections found."
    
    output = [f"## 💻 Found {len(code_results)} Relevant Code Sections"]
    output.append("")
    
    for i, code in enumerate(code_results, 1):
        file_name = code.get('file_name', 'Unknown file')
        similarity = code.get('similarity_score', 0)
        
        output.append(f"### {i}. {file_name}")
        output.append(f"**Similarity:** {similarity:.3f} | **Path:** {code.get('file_path', 'N/A')}")
        
        if code.get('chunk_index', 0) > 0:
            total_chunks = code.get('total_chunks', 1)
            output.append(f"**Chunk:** {code['chunk_index'] + 1}/{total_chunks}")
        
        output.append("")
        
        # Extract and display code snippet
        code_text = code.get('code_text', '')
        if 'File:' in code_text:
            _, code_content = code_text.split('|', 1)
            code_content = code_content.strip()
        else:
            code_content = code_text
        
        # Limit code display length
        if len(code_content) > 500:
            code_content = code_content[:500] + "\n... (truncated)"
        
        output.append("```python")
        output.append(code_content)
        output.append("```")
        output.append("---")
    
    return "\n".join(output)

def format_evaluation_metrics(bug_eval: Dict, suggestion_eval: Dict) -> str:
    """Format evaluation metrics for display"""
    output = ["## 📊 Analysis Quality Metrics"]
    output.append("")
    
    # Bug retrieval metrics
    output.append("### 🔍 Retrieval Relevance")
    output.append(f"- **Average Similarity Score:** {bug_eval['average_similarity']:.3f}/1.0")
    output.append(f"- **Semantic Relevance:** {bug_eval['relevance_score']:.3f}/1.0")
    output.append(f"- **Results Retrieved:** {bug_eval['result_count']}")
    
    # Suggestion quality metrics
    output.append("")
    output.append("### 🛠️ Suggestion Quality")
    output.append(f"- **Completeness:** {suggestion_eval['completeness_score']:.3f}/1.0")
    output.append(f"- **Specificity:** {suggestion_eval['specificity_score']:.3f}/1.0")
    output.append(f"- **Actionability:** {suggestion_eval['actionability_score']:.3f}/1.0")
    output.append(f"- **Overall Usefulness:** {suggestion_eval['overall_usefulness']:.3f}/1.0")
    
    # Quality assessment
    overall_quality = (bug_eval['relevance_score'] + suggestion_eval['overall_usefulness']) / 2
    output.append("")
    output.append("### ⭐ Overall Analysis Quality")
    
    if overall_quality >= 0.8:
        quality_label = "🟢 Excellent"
    elif overall_quality >= 0.6:
        quality_label = "🟡 Good"
    elif overall_quality >= 0.4:
        quality_label = "🟠 Fair"
    else:
        quality_label = "🔴 Poor"
    
    output.append(f"**Quality Rating:** {quality_label} ({overall_quality:.3f}/1.0)")
    
    return "\n".join(output)

# Create Gradio interface
def create_interface():
    """Create the Gradio interface for the Bug Report Analysis Agent"""
    
    with gr.Blocks(
        title="🐞 Bug Report Analysis Agent",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .tab-nav {
            font-weight: bold;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 🐞 Bug Report Analysis Agent
        
        **Advanced RAG-powered system for intelligent bug analysis**
        
        This system analyzes bug reports using Retrieval-Augmented Generation (RAG) to:
        - 🔍 Find similar past issues in the bug database
        - 💻 Identify relevant code sections that might be related
        - 🛠️ Suggest potential causes and fixes
        - 📊 Evaluate retrieval relevance and suggestion usefulness
        
        ---
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    lines=6,
                    label="🔍 Bug Description",
                    placeholder="Describe the bug you're experiencing...\n\nExample: 'Login form redirects back to login page after entering correct credentials'",
                    info="Provide as much detail as possible for better analysis"
                )
                
                with gr.Row():
                    analyze_btn = gr.Button("🔍 Analyze Bug", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
        
        with gr.Row():
            with gr.Column(scale=1):
                similar_bugs_output = gr.Markdown(
                    label="🪲 Similar Bug Reports",
                    value="Enter a bug description and click 'Analyze Bug' to see similar issues..."
                )
            
            with gr.Column(scale=1):
                relevant_code_output = gr.Markdown(
                    label="💻 Relevant Code Sections", 
                    value="Code analysis will appear here..."
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                suggestions_output = gr.Markdown(
                    label="🛠️ Fix Suggestions",
                    value="Intelligent fix suggestions will be generated here..."
                )
            
            with gr.Column(scale=1):
                evaluation_output = gr.Markdown(
                    label="📊 Quality Metrics",
                    value="Analysis quality metrics will be shown here..."
                )
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_bug_report,
            inputs=[input_box],
            outputs=[similar_bugs_output, relevant_code_output, suggestions_output, evaluation_output],
            api_name="analyze_bug"
        )
        
        clear_btn.click(
            fn=lambda: ("", "Enter a bug description and click 'Analyze Bug' to see similar issues...", 
                       "Code analysis will appear here...", 
                       "Intelligent fix suggestions will be generated here...",
                       "Analysis quality metrics will be shown here..."),
            inputs=[],
            outputs=[input_box, similar_bugs_output, relevant_code_output, suggestions_output, evaluation_output]
        )
        
        # Footer
        gr.Markdown("""
        ---
        **🚀 Built with:** LangChain • Sentence Transformers • FAISS • Gradio
        
        **📈 Features:** Semantic Search • Similarity Scoring • Code Analysis • Fix Suggestions • Quality Evaluation
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
