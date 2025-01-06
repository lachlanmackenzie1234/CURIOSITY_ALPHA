"""Analyze and organize entire codebase using ALPHA."""

import time
from pathlib import Path
from typing import Dict, List, Set

from .core.organizer import CodeOrganizer


class CodebaseAnalyzer:
    """Analyzes and organizes entire codebases."""

    def __init__(self, root_dir: str, max_files: int = 100):
        self.root_dir = Path(root_dir)
        self.organizer = CodeOrganizer()
        self.patterns: Set[str] = set()
        self.max_files = max_files

    def learn_codebase(self) -> None:
        """Learn patterns from entire codebase."""
        print(f"Learning patterns from {self.root_dir}...")

        # Get all Python files
        python_files = [
            f
            for f in self.root_dir.rglob("*.py")
            if "__pycache__" not in str(f)
            and "site-packages" not in str(f)  # Skip external packages
            and ".venv" not in str(f)  # Skip virtual environment
        ]

        # Limit to max_files
        if len(python_files) > self.max_files:
            print(f"Limiting analysis to {self.max_files} files")
            python_files = python_files[: self.max_files]

        # First pass: Learn all patterns
        for i, file_path in enumerate(python_files, 1):
            try:
                print(f"[{i}/{len(python_files)}] Learning from {file_path}")
                self.organizer.learn_from_file(str(file_path))
                time.sleep(0.1)  # Small delay to prevent overload
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def analyze_structure(self) -> Dict[str, List[str]]:
        """Analyze structural patterns in codebase."""
        print("\nAnalyzing codebase structure...")
        return self.organizer.organize_directory(str(self.root_dir))

    def generate_report(self, patterns: Dict[str, List[str]]) -> str:
        """Generate detailed analysis report."""
        report = ["# Codebase Analysis Report\n"]

        # Overall statistics
        total_files = sum(len(files) for files in patterns.values())
        report.append("## Overview\n")
        report.append(f"- Total patterns found: {len(patterns)}")
        report.append(f"- Total files analyzed: {total_files}\n")

        # Pattern categories
        report.append("## Pattern Categories\n")
        for category, files in sorted(patterns.items()):
            report.append(f"### {category}")
            report.append(f"Found in {len(files)} locations:")
            for file_loc in sorted(files):
                report.append(f"- {file_loc}")
            report.append("")

        # Add recommendations
        report.append("## Recommendations\n")
        if patterns:
            report.append("Based on the analysis, here are some suggestions:")
            for category in patterns:
                if len(patterns[category]) > 1:
                    report.append(
                        f"- Consider consolidating similar patterns in category '{category}'"
                    )
            report.append(
                "- Review files with unique patterns for potential refactoring"
            )
            report.append(
                "- Look for opportunities to standardize similar code structures"
            )
        else:
            report.append("No specific recommendations at this time.")

        return "\n".join(report)


def main():
    """Analyze current codebase."""
    # Get project root (parent of src directory)
    current_dir = Path(__file__).parent.parent.parent

    # Initialize analyzer
    analyzer = CodebaseAnalyzer(current_dir)

    # Learn and analyze
    analyzer.learn_codebase()
    patterns = analyzer.analyze_structure()

    # Generate and save report
    report = analyzer.generate_report(patterns)
    report_path = current_dir / "codebase_analysis.md"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nAnalysis complete! Report saved to {report_path}")


if __name__ == "__main__":
    main()
