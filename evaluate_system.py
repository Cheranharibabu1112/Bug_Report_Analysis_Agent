#!/usr/bin/env python3
"""
Bug Report Analysis Agent - Comprehensive Evaluation Script
============================================================
This script demonstrates and evaluates the RAG system's performance
on various types of bug reports and provides detailed analysis.
"""

import sys
import time
import json
from typing import Dict, List, Tuple
import pandas as pd

# Import the main system components
from app import (
    rag_system, evaluator, suggestion_engine,
    analyze_bug_report, format_similar_bugs, 
    format_relevant_code, format_evaluation_metrics
)

class SystemEvaluator:
    """Comprehensive evaluation of the Bug Report Analysis system"""
    
    def __init__(self):
        self.test_queries = [
            {
                "query": "Login form redirects back to login page after entering correct credentials",
                "category": "Authentication",
                "expected_components": ["login", "auth", "session"],
                "description": "Classic authentication redirect issue"
            },
            {
                "query": "Database connection times out during high traffic periods",
                "category": "Database",
                "expected_components": ["database", "connection", "timeout"],
                "description": "Performance issue under load"
            },
            {
                "query": "Email notifications for password reset are not being sent to users",
                "category": "Email",
                "expected_components": ["email", "smtp", "password"],
                "description": "Email service functionality problem"
            },
            {
                "query": "Submit button on contact form doesn't respond when clicked",
                "category": "UI/Frontend",
                "expected_components": ["button", "form", "javascript"],
                "description": "Frontend interaction issue"
            },
            {
                "query": "API returns 500 internal server error for user profile updates",
                "category": "API",
                "expected_components": ["api", "profile", "server"],
                "description": "Backend API error"
            },
            {
                "query": "Memory usage increases continuously when uploading large files",
                "category": "Performance",
                "expected_components": ["memory", "upload", "file"],
                "description": "Memory leak in file handling"
            },
            {
                "query": "Dashboard charts show incorrect data for monthly revenue reports",
                "category": "Data/Analytics",
                "expected_components": ["dashboard", "chart", "data"],
                "description": "Data visualization accuracy issue"
            },
            {
                "query": "User session expires too quickly causing frequent re-authentication",
                "category": "Session Management",
                "expected_components": ["session", "timeout", "authentication"],
                "description": "Session timeout configuration issue"
            }
        ]
    
    def run_comprehensive_evaluation(self) -> Dict:
        """Run comprehensive evaluation of the system"""
        print("🚀 Starting Comprehensive Bug Report Analysis Evaluation")
        print("=" * 70)
        
        start_time = time.time()
        results = {
            "test_results": [],
            "performance_metrics": {},
            "quality_analysis": {},
            "component_coverage": {},
            "recommendations": []
        }
        
        # Test each query
        for i, test_case in enumerate(self.test_queries, 1):
            print(f"\n📋 Test Case {i}/{len(self.test_queries)}: {test_case['category']}")
            print(f"Query: {test_case['query']}")
            print("-" * 50)
            
            # Run analysis
            test_result = self.evaluate_single_query(test_case)
            results["test_results"].append(test_result)
            
            # Print summary
            self.print_test_summary(test_result)
            
            time.sleep(0.5)  # Brief pause between tests
        
        # Calculate overall metrics
        results["performance_metrics"] = self.calculate_performance_metrics(results["test_results"])
        results["quality_analysis"] = self.analyze_quality_patterns(results["test_results"])
        results["component_coverage"] = self.analyze_component_coverage(results["test_results"])
        results["recommendations"] = self.generate_recommendations(results)
        
        total_time = time.time() - start_time
        results["evaluation_time"] = total_time
        
        # Print final report
        self.print_final_report(results)
        
        return results
    
    def evaluate_single_query(self, test_case: Dict) -> Dict:
        """Evaluate a single test query"""
        query = test_case["query"]
        start_time = time.time()
        
        # Run the analysis
        try:
            similar_bugs_output, relevant_code_output, suggestions, evaluation_output = analyze_bug_report(query)
            
            # Get raw data for analysis
            similar_bugs = rag_system.search_similar_bugs(query, k=5)
            relevant_code = rag_system.search_relevant_code(query, k=5)
            
            # Evaluate results
            bug_evaluation = evaluator.evaluate_retrieval_relevance(query, similar_bugs)
            suggestion_evaluation = evaluator.evaluate_suggestion_usefulness(query, suggestions)
            
            processing_time = time.time() - start_time
            
            return {
                "test_case": test_case,
                "processing_time": processing_time,
                "similar_bugs": similar_bugs,
                "relevant_code": relevant_code,
                "suggestions": suggestions,
                "bug_evaluation": bug_evaluation,
                "suggestion_evaluation": suggestion_evaluation,
                "outputs": {
                    "similar_bugs_output": similar_bugs_output,
                    "relevant_code_output": relevant_code_output,
                    "evaluation_output": evaluation_output
                },
                "success": True
            }
            
        except Exception as e:
            return {
                "test_case": test_case,
                "processing_time": time.time() - start_time,
                "error": str(e),
                "success": False
            }
    
    def print_test_summary(self, result: Dict):
        """Print summary for a single test"""
        if not result["success"]:
            print(f"❌ Error: {result['error']}")
            return
        
        bug_eval = result["bug_evaluation"]
        suggestion_eval = result["suggestion_evaluation"]
        
        print(f"⏱️  Processing Time: {result['processing_time']:.2f}s")
        print(f"🔍 Similar Bugs Found: {bug_eval['result_count']}")
        print(f"📊 Retrieval Relevance: {bug_eval['relevance_score']:.3f}/1.0")
        print(f"🛠️  Suggestion Quality: {suggestion_eval['overall_usefulness']:.3f}/1.0")
        
        # Quality indicator
        overall_quality = (bug_eval['relevance_score'] + suggestion_eval['overall_usefulness']) / 2
        if overall_quality >= 0.8:
            quality_icon = "🟢"
        elif overall_quality >= 0.6:
            quality_icon = "🟡"
        elif overall_quality >= 0.4:
            quality_icon = "🟠"
        else:
            quality_icon = "🔴"
        
        print(f"{quality_icon} Overall Quality: {overall_quality:.3f}/1.0")
    
    def calculate_performance_metrics(self, test_results: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        successful_tests = [r for r in test_results if r["success"]]
        
        if not successful_tests:
            return {"error": "No successful tests to analyze"}
        
        processing_times = [r["processing_time"] for r in successful_tests]
        retrieval_scores = [r["bug_evaluation"]["relevance_score"] for r in successful_tests]
        suggestion_scores = [r["suggestion_evaluation"]["overall_usefulness"] for r in successful_tests]
        bug_counts = [r["bug_evaluation"]["result_count"] for r in successful_tests]
        
        return {
            "total_tests": len(test_results),
            "successful_tests": len(successful_tests),
            "success_rate": len(successful_tests) / len(test_results),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "average_retrieval_score": sum(retrieval_scores) / len(retrieval_scores),
            "average_suggestion_score": sum(suggestion_scores) / len(suggestion_scores),
            "average_bugs_found": sum(bug_counts) / len(bug_counts),
            "retrieval_score_std": pd.Series(retrieval_scores).std(),
            "suggestion_score_std": pd.Series(suggestion_scores).std()
        }
    
    def analyze_quality_patterns(self, test_results: List[Dict]) -> Dict:
        """Analyze quality patterns across different categories"""
        successful_tests = [r for r in test_results if r["success"]]
        
        category_analysis = {}
        for result in successful_tests:
            category = result["test_case"]["category"]
            
            if category not in category_analysis:
                category_analysis[category] = {
                    "count": 0,
                    "retrieval_scores": [],
                    "suggestion_scores": [],
                    "processing_times": []
                }
            
            category_analysis[category]["count"] += 1
            category_analysis[category]["retrieval_scores"].append(
                result["bug_evaluation"]["relevance_score"]
            )
            category_analysis[category]["suggestion_scores"].append(
                result["suggestion_evaluation"]["overall_usefulness"]
            )
            category_analysis[category]["processing_times"].append(
                result["processing_time"]
            )
        
        # Calculate averages for each category
        for category, data in category_analysis.items():
            data["avg_retrieval"] = sum(data["retrieval_scores"]) / len(data["retrieval_scores"])
            data["avg_suggestion"] = sum(data["suggestion_scores"]) / len(data["suggestion_scores"])
            data["avg_processing_time"] = sum(data["processing_times"]) / len(data["processing_times"])
        
        return category_analysis
    
    def analyze_component_coverage(self, test_results: List[Dict]) -> Dict:
        """Analyze how well the system covers different components"""
        component_coverage = {}
        
        for result in test_results:
            if not result["success"]:
                continue
            
            test_case = result["test_case"]
            expected_components = test_case.get("expected_components", [])
            
            # Check if similar bugs contain expected components
            similar_bugs = result["similar_bugs"]
            found_components = set()
            
            for bug in similar_bugs:
                component = bug.get("component", "").lower()
                description = bug.get("description", "").lower()
                title = bug.get("title", "").lower()
                
                for expected in expected_components:
                    if expected.lower() in f"{component} {description} {title}":
                        found_components.add(expected)
            
            component_coverage[test_case["category"]] = {
                "expected": expected_components,
                "found": list(found_components),
                "coverage_ratio": len(found_components) / len(expected_components) if expected_components else 0
            }
        
        return component_coverage
    
    def generate_recommendations(self, results: Dict) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        performance = results["performance_metrics"]
        quality = results["quality_analysis"]
        
        # Performance recommendations
        if performance.get("average_processing_time", 0) > 3.0:
            recommendations.append("Consider optimizing query processing time (currently > 3s average)")
        
        if performance.get("success_rate", 1.0) < 0.95:
            recommendations.append("Improve error handling and system reliability")
        
        # Quality recommendations
        avg_retrieval = performance.get("average_retrieval_score", 0)
        avg_suggestion = performance.get("average_suggestion_score", 0)
        
        if avg_retrieval < 0.7:
            recommendations.append("Improve bug retrieval relevance (add more diverse training data)")
        
        if avg_suggestion < 0.7:
            recommendations.append("Enhance suggestion generation quality (refine fix templates)")
        
        # Category-specific recommendations
        for category, data in quality.items():
            if data["avg_retrieval"] < 0.6:
                recommendations.append(f"Improve {category} category retrieval performance")
            
            if data["avg_suggestion"] < 0.6:
                recommendations.append(f"Enhance {category} category suggestion quality")
        
        if not recommendations:
            recommendations.append("System performance is excellent across all metrics!")
        
        return recommendations
    
    def print_final_report(self, results: Dict):
        """Print comprehensive final evaluation report"""
        print("\n" + "=" * 70)
        print("📊 COMPREHENSIVE EVALUATION REPORT")
        print("=" * 70)
        
        # Performance Summary
        perf = results["performance_metrics"]
        print(f"\n🚀 PERFORMANCE SUMMARY")
        print(f"{'Total Tests:':<25} {perf['total_tests']}")
        print(f"{'Success Rate:':<25} {perf['success_rate']:.1%}")
        print(f"{'Avg Processing Time:':<25} {perf['average_processing_time']:.2f}s")
        print(f"{'Avg Retrieval Score:':<25} {perf['average_retrieval_score']:.3f}/1.0")
        print(f"{'Avg Suggestion Score:':<25} {perf['average_suggestion_score']:.3f}/1.0")
        print(f"{'Avg Bugs Found:':<25} {perf['average_bugs_found']:.1f}")
        
        # Quality Analysis by Category
        print(f"\n📈 QUALITY ANALYSIS BY CATEGORY")
        quality = results["quality_analysis"]
        for category, data in quality.items():
            print(f"\n{category}:")
            print(f"  Retrieval: {data['avg_retrieval']:.3f} | Suggestions: {data['avg_suggestion']:.3f}")
        
        # Component Coverage
        print(f"\n🎯 COMPONENT COVERAGE ANALYSIS")
        coverage = results["component_coverage"]
        for category, data in coverage.items():
            coverage_pct = data['coverage_ratio'] * 100
            print(f"{category}: {coverage_pct:.0f}% coverage ({len(data['found'])}/{len(data['expected'])} components)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(results["recommendations"], 1):
            print(f"{i}. {rec}")
        
        # Overall Rating
        overall_score = (perf['average_retrieval_score'] + perf['average_suggestion_score']) / 2
        if overall_score >= 0.8:
            rating = "🟢 EXCELLENT"
        elif overall_score >= 0.7:
            rating = "🟡 GOOD"
        elif overall_score >= 0.6:
            rating = "🟠 FAIR"
        else:
            rating = "🔴 NEEDS IMPROVEMENT"
        
        print(f"\n⭐ OVERALL SYSTEM RATING: {rating} ({overall_score:.3f}/1.0)")
        print(f"📅 Evaluation completed in {results['evaluation_time']:.1f} seconds")
        print("=" * 70)
    
    def save_results(self, results: Dict, filename: str = "evaluation_results.json"):
        """Save evaluation results to file"""
        try:
            # Convert numpy types to native Python types for JSON serialization
            def convert_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(item) for item in obj]
                else:
                    return obj
            
            serializable_results = convert_types(results)
            
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"📁 Results saved to {filename}")
            
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def run_interactive_demo():
    """Run an interactive demonstration of the system"""
    print("🎮 Interactive Bug Report Analysis Demo")
    print("Enter bug descriptions to see real-time analysis")
    print("Type 'quit' to exit\n")
    
    while True:
        try:
            query = input("🐞 Describe a bug: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Thanks for trying the Bug Report Analysis Agent!")
                break
            
            if not query:
                continue
            
            print("\n🔍 Analyzing...")
            start_time = time.time()
            
            similar_bugs_output, relevant_code_output, suggestions, evaluation_output = analyze_bug_report(query)
            
            processing_time = time.time() - start_time
            
            print(f"⏱️ Analysis completed in {processing_time:.2f} seconds\n")
            print("📋 RESULTS:")
            print("-" * 50)
            print(similar_bugs_output[:500] + "..." if len(similar_bugs_output) > 500 else similar_bugs_output)
            print("\n" + evaluation_output)
            print("\n" + "="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    evaluator_instance = SystemEvaluator()
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_interactive_demo()
    else:
        # Run comprehensive evaluation
        results = evaluator_instance.run_comprehensive_evaluation()
        evaluator_instance.save_results(results)
        
        print("\n🎯 To run interactive demo: python evaluate_system.py --demo") 