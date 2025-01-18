"""Directory restructuring based on ALPHA pattern analysis."""

import logging
import os
from pathlib import Path
from typing import Dict, List, Set

from .core.organizer import CodeOrganizer


class DirectoryRestructurer:
    """Restructures codebase based on pattern analysis."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.organizer = CodeOrganizer()
        self.new_structure: Dict[str, Set[str]] = {
            "core": {"binary", "patterns", "translation"},
            "autonomous": {"learning", "evolution", "adaptation"},
            "processing": {"data", "analysis", "transformation"},
            "system": {"management", "monitoring", "integration"},
            "utils": {"testing", "tools", "helpers"},
        }

    def analyze_and_plan(self) -> Dict[str, List[str]]:
        """Analyze current structure and plan reorganization."""
        # Learn patterns from Python files
        for file_path in self.root_dir.rglob("*.py"):
            if not any(
                x in str(file_path) for x in [".venv", "__pycache__", "site-packages", ".git"]
            ):
                try:
                    self.organizer.learn_from_file(str(file_path))
                except Exception as e:
                    logging.error(f"Error learning from {file_path}: {e}")

        # Get organized patterns
        patterns = self.organizer.organize_directory(str(self.root_dir))

        # Plan new structure
        restructure_plan: Dict[str, List[str]] = {}

        # Map files to new directories based on patterns
        for category, files in patterns.items():
            if "pattern" in category or "binary" in category:
                target_dir = "core/patterns"
            elif "autonomous" in category or "evolution" in category:
                target_dir = "autonomous/evolution"
            elif "data" in category or "process" in category:
                target_dir = "processing/data"
            elif "system" in category or "manage" in category:
                target_dir = "system/management"
            elif "test" in category:
                target_dir = "utils/testing"
            else:
                target_dir = "utils/tools"

            if target_dir not in restructure_plan:
                restructure_plan[target_dir] = []
            restructure_plan[target_dir].extend(files)

        return restructure_plan

    def generate_restructure_script(self, plan: Dict[str, List[str]]) -> str:
        """Generate shell script for restructuring."""
        script_lines = ["#!/bin/bash", ""]

        # Create new directory structure
        for main_dir, subdirs in self.new_structure.items():
            for subdir in subdirs:
                script_lines.append(f"mkdir -p src/ALPHA/{main_dir}/{subdir}")

        # Move files to new locations
        for target_dir, files in plan.items():
            for file_path in files:
                rel_path = os.path.relpath(file_path, str(self.root_dir))
                new_path = f"src/ALPHA/{target_dir}/{os.path.basename(file_path)}"
                script_lines.append(f"mv {rel_path} {new_path}")

        return "\n".join(script_lines)

    def generate_report(self, plan: Dict[str, List[str]]) -> str:
        """Generate restructuring report."""
        report = ["# Directory Restructuring Plan", ""]

        # Overview
        report.append("## New Structure Overview")
        for main_dir, subdirs in self.new_structure.items():
            report.append(f"\n### {main_dir.title()}")
            for subdir in subdirs:
                report.append(f"- {subdir}")

        # File Movements
        report.append("\n## File Movements")
        total_files = sum(len(files) for files in plan.values())
        report.append(f"\nTotal files to move: {total_files}")

        for target_dir, files in plan.items():
            report.append(f"\n### To {target_dir} ({len(files)} files)")
            for file_path in files:
                rel_path = os.path.relpath(file_path, str(self.root_dir))
                report.append(f"- {rel_path}")

        # Benefits
        report.append("\n## Benefits")
        report.append("- Improved code organization based on functionality")
        report.append("- Better separation of concerns")
        report.append("- Clearer dependency management")
        report.append("- Enhanced maintainability")
        report.append("- Easier navigation and development")

        # Implementation Notes
        report.append("\n## Implementation Notes")
        report.append("1. Back up your codebase before running the restructuring script")
        report.append("2. Review the planned file movements carefully")
        report.append("3. Update import statements after restructuring")
        report.append("4. Run tests to verify functionality")
        report.append("5. Update documentation to reflect new structure")

        return "\n".join(report)


def main():
    """Generate restructuring plan and script."""
    try:
        project_root = Path(__file__).parent.parent.parent
        restructurer = DirectoryRestructurer(str(project_root))

        # Generate plan
        print("Analyzing codebase...")
        plan = restructurer.analyze_and_plan()

        # Generate report
        print("Generating report...")
        report = restructurer.generate_report(plan)
        report_path = project_root / "restructure_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        # Generate script
        print("Generating restructuring script...")
        script = restructurer.generate_restructure_script(plan)
        script_path = project_root / "restructure.sh"
        with open(script_path, "w") as f:
            f.write(script)

        print("\nRestructuring plan generated!")
        print(f"- Report: {report_path}")
        print(f"- Script: {script_path}")

    except Exception as e:
        logging.error(f"Error generating restructuring plan: {e}")


if __name__ == "__main__":
    main()
