#!/usr/bin/env python3
"""Run ALPHA self-analysis on its own codebase."""

import asyncio
from pathlib import Path
import sys
import os

# Add project root to Python path
project_root = Path(__file__).parent  # Current directory is src
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import after path setup
from ALPHA.core.alpha_self_analysis import ALPHASelfAnalysis


async def main():
    """Run ALPHA self-analysis."""
    print("\nStarting ALPHA self-analysis...\n")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    
    try:
        analyzer = ALPHASelfAnalysis()
        report = await analyzer.analyze_codebase("ALPHA")
        
        print("\nAnalysis complete! Results:\n")
        print(f"Files analyzed: {report['files_analyzed']}")
        print(f"Patterns identified: {report['patterns_identified']}")
        
        print("\nPattern Statistics:")
        for pattern_type, count in report['pattern_stats'].items():
            print(f"- {pattern_type}: {count}")
        
        print("\nTranslation Effectiveness:")
        for metric, value in report['translation_effectiveness'].items():
            if isinstance(value, float):
                print(f"- {metric}: {value:.2%}")
            else:
                print(f"- {metric}: {value}")
        
        print("\nLearning Metrics:")
        for metric, value in report['learning_metrics'].items():
            print(f"- {metric}: {value:.2%}")
        
        if report.get('recommendations'):
            print("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"\n{i}. {rec['description']} (Priority: {rec['priority']})")
                if 'details' in rec:
                    print(f"   Details: {rec['details']}")
    
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main()) 