#!/usr/bin/env python3
"""
Simple workflow runner script (core pipeline).

Run from project root:
  python src/run_workflow.py

Or provide a config path:
  python src/run_workflow.py path/to/config.json
"""

import sys

from workflow_runner import run_from_config_file


if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "config/config.json"
    print(f"Loading configuration from: {config_file}\n")

    try:
        results = run_from_config_file(config_file)

        # Print summary
        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        for step_name, step_result in results["step_results"].items():
            if "error" in step_result:
                print(f"\n{step_name}: [ERROR]")
                print(f"  {step_result['error']}")
            else:
                processed = step_result.get("total_files_processed") or step_result.get("total_files") or 0
                print(f"\n{step_name}: [OK]")
                print(f"  Files processed: {processed}")

        print("\n" + "=" * 70)

        # Exit with error if any step failed
        errors = sum(1 for r in results["step_results"].values() if "error" in r)
        sys.exit(1 if errors > 0 else 0)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


