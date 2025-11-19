#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
Find affected CI tests based on changed files.

Usage:
    python find_affected_tests.py <changed_file_path> [<changed_file_path> ...]

    # Or pipe from git diff:
    git diff --name-only HEAD~1 | python find_affected_tests.py --stdin

Examples:
    python find_affected_tests.py Deeploy/Targets/PULPOpen/TileConstraints/ConvTileConstraint.py
    python find_affected_tests.py Deeploy/DeeployTypes.py
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Set

# Load mappings
SCRIPT_DIR = Path(__file__).parent
DOCS_DIR = SCRIPT_DIR.parent / "docs"

def load_mappings():
    """Load the CI and operator mappings."""
    ci_mapping_path = DOCS_DIR / "ci_test_mapping.json"
    operator_mapping_path = DOCS_DIR / "test_operator_mapping.json"

    with open(ci_mapping_path) as f:
        ci_mapping = json.load(f)

    with open(operator_mapping_path) as f:
        operator_mapping = json.load(f)

    return ci_mapping, operator_mapping


def get_affected_tests(changed_files: List[str], ci_mapping: Dict, operator_mapping: Dict) -> Dict[str, Set[str]]:
    """
    Determine which tests are affected by the changed files.

    Returns a dict mapping platforms to sets of affected tests.
    """
    affected = {
        "Generic": set(),
        "CortexM": set(),
        "MemPool": set(),
        "Snitch": set(),
        "Snitch-Tiled": set(),
        "Siracusa": set(),
        "Siracusa-Tiled": set(),
        "Siracusa-Neureka-Tiled": set(),
        "SoftHier": set(),
        "Chimera": set(),
        "Unit-Tests": set(),  # pytest tests
    }

    for file_path in changed_files:
        path = Path(file_path)
        parts = path.parts

        # Core framework changes affect everything
        if "DeeployTypes.py" in str(path) or "AbstractDataTypes.py" in str(path):
            print(f"⚠️  Core framework change: {path}")
            print("   → Run ALL tests on ALL platforms")
            for platform in affected:
                affected[platform].add("ALL")
            affected["Unit-Tests"].add("testTypes.py")
            continue

        # Tiling extension changes affect all tiled tests
        if "TilingExtension" in str(path):
            print(f"📦 Tiling extension change: {path}")
            print("   → Run ALL tiled tests")
            affected["Siracusa-Tiled"].add("ALL")
            affected["Snitch-Tiled"].add("ALL")
            affected["Siracusa-Neureka-Tiled"].add("ALL")
            affected["Unit-Tests"].add("testTilerExtension.py")
            continue

        # Memory level extension
        if "MemoryLevelExtension" in str(path):
            print(f"📦 Memory level extension change: {path}")
            affected["Unit-Tests"].add("testMemoryLevelExtension.py")
            # Affects all platforms with memory levels
            for platform in ["Siracusa", "Siracusa-Tiled", "MemPool"]:
                affected[platform].add("simpleRegression")
            continue

        # Target-specific changes
        if "Targets" in parts:
            try:
                target_idx = parts.index("Targets")
                platform = parts[target_idx + 1]

                # Map platform names
                platform_map = {
                    "Generic": ["Generic"],
                    "CortexM": ["CortexM"],
                    "MemPool": ["MemPool"],
                    "Snitch": ["Snitch", "Snitch-Tiled"],
                    "PULPOpen": ["Siracusa", "Siracusa-Tiled"],
                    "Neureka": ["Siracusa-Neureka-Tiled"],
                    "Siracusa": ["Siracusa", "Siracusa-Tiled"],
                    "SoftHier": ["SoftHier"],
                    "Chimera": ["Chimera"],
                }

                affected_platforms = platform_map.get(platform, [])

                # Determine which operator/component
                if "TileConstraints" in parts:
                    # TileConstraint change
                    constraint_file = path.name
                    print(f"🔧 TileConstraint change: {path}")

                    # Map constraint to tests
                    constraint_tests = {
                        "ConvTileConstraint.py": ["testFloat2DConvolution", "test2DRequantizedConv", "miniMobileNet", "testFloatDemoTinyViT"],
                        "DWConvTileConstraint.py": ["testFloat2DDWConvolution", "testRequantizedDWConv", "miniMobileNet", "testFloatDemoTinyViT"],
                        "GEMMTileConstraint.py": ["testFloatGEMM", "testRQGEMM", "testRequantizedLinear"],
                        "MatMulTileConstraint.py": ["testFloatMatmul", "testMatMul", "Attention"],
                        "MaxPoolTileConstraint.py": ["testFloatMaxPool", "simpleRegression"],
                        "LayernormTileConstraint.py": ["testFloatLayerNorm"],
                        "ReduceMeanConstraint.py": ["testReduceMean", "testFloatDemoTinyViT"],
                    }

                    tests = constraint_tests.get(constraint_file, [])
                    for p in affected_platforms:
                        if "Tiled" in p:
                            affected[p].update(tests)

                elif "Templates" in parts:
                    # Template change
                    template_file = path.name
                    print(f"📝 Template change: {path}")

                    # Map template to operators and tests
                    if "Conv" in template_file:
                        tests = ["testFloat2DConvolution", "testFloat2DDWConvolution", "test2DRequantizedConv", "miniMobileNet"]
                    elif "GEMM" in template_file:
                        tests = ["testFloatGEMM", "testRQGEMM"]
                    elif "MatMul" in template_file:
                        tests = ["testFloatMatmul", "testMatMul"]
                    elif "Add" in template_file:
                        tests = ["Adder", "testFloatAdder"]
                    else:
                        tests = []

                    for p in affected_platforms:
                        affected[p].update(tests)

                elif "Parsers.py" in str(path) or "Bindings.py" in str(path) or "TypeCheckers.py" in str(path):
                    print(f"⚙️  Core platform component change: {path}")
                    print(f"   → Run basic tests on {affected_platforms}")
                    for p in affected_platforms:
                        affected[p].add("Adder")
                        affected[p].add("testMatMul")
                        affected[p].add("simpleRegression")

                else:
                    print(f"📁 Platform file change: {path}")
                    for p in affected_platforms:
                        affected[p].add("Adder")

            except (IndexError, ValueError):
                pass

        # TargetLibraries changes
        if "TargetLibraries" in parts:
            try:
                lib_idx = parts.index("TargetLibraries")
                platform = parts[lib_idx + 1]
                print(f"📚 Library change: {path}")

                platform_map = {
                    "Generic": ["Generic"],
                    "CMSIS": ["CortexM"],
                    "MemPool": ["MemPool"],
                    "PULPOpen": ["Siracusa", "Siracusa-Tiled"],
                    "Snitch": ["Snitch", "Snitch-Tiled"],
                }

                for p in platform_map.get(platform, []):
                    affected[p].add("Adder")
                    affected[p].add("simpleRegression")

            except (IndexError, ValueError):
                pass

        # Test changes
        if "DeeployTest/Tests" in str(path):
            # Direct test file change - run that test
            test_name = None
            for part in parts:
                if part in operator_mapping:
                    test_name = part
                    break

            if test_name:
                print(f"🧪 Test file change: {path}")
                # Add to all platforms that run this test
                for platform, tests in affected.items():
                    # Check if test is in platform's test list
                    affected[platform].add(test_name)

    return affected


def print_recommendations(affected: Dict[str, Set[str]], ci_mapping: Dict):
    """Print test run recommendations."""
    print("\n" + "="*60)
    print("TEST RECOMMENDATIONS")
    print("="*60 + "\n")

    # Unit tests first
    if affected["Unit-Tests"]:
        print("📋 Unit Tests (pytest):")
        for test in sorted(affected["Unit-Tests"]):
            print(f"   pytest {test}")
        print()

    # Platform tests
    platform_runners = {
        "Generic": "testRunner_generic.py",
        "CortexM": "testRunner_cortexm.py",
        "MemPool": "testRunner_mempool.py",
        "Snitch": "testRunner_snitch.py --cores=9",
        "Snitch-Tiled": "testRunner_tiled_snitch.py --cores=9",
        "Siracusa": "testRunner_siracusa.py --cores=8",
        "Siracusa-Tiled": "testRunner_tiled_siracusa.py --cores=8",
        "Siracusa-Neureka-Tiled": "testRunner_tiled_siracusa_w_neureka.py --cores=8",
        "SoftHier": "testRunner_softhier.py",
        "Chimera": "testRunner_chimera.py",
    }

    for platform, tests in affected.items():
        if platform == "Unit-Tests" or not tests:
            continue

        runner = platform_runners.get(platform, "")
        if not runner:
            continue

        print(f"🖥️  {platform}:")

        if "ALL" in tests:
            print(f"   # Run full CI workflow for {platform}")
            print(f"   # See .github/workflows/ci-platform-{platform.lower().replace('-', '-')}.yml")
        else:
            for test in sorted(tests):
                if "Tiled" in platform:
                    # Add L1 configuration for tiled tests
                    print(f"   python {runner} -t Tests/{test} --l1=64000")
                else:
                    print(f"   python {runner} -t Tests/{test}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Find CI tests affected by file changes")
    parser.add_argument("files", nargs="*", help="Changed file paths")
    parser.add_argument("--stdin", action="store_true", help="Read file paths from stdin")

    args = parser.parse_args()

    # Get changed files
    if args.stdin:
        changed_files = [line.strip() for line in sys.stdin if line.strip()]
    else:
        changed_files = args.files

    if not changed_files:
        print("Usage: python find_affected_tests.py <file1> [<file2> ...]")
        print("   or: git diff --name-only | python find_affected_tests.py --stdin")
        sys.exit(1)

    print("="*60)
    print("CHANGE IMPACT ANALYSIS")
    print("="*60 + "\n")

    print("Changed files:")
    for f in changed_files:
        print(f"  • {f}")
    print()

    # Load mappings
    try:
        ci_mapping, operator_mapping = load_mappings()
    except FileNotFoundError as e:
        print(f"Error: Could not load mappings: {e}")
        print("Make sure to run from the Deeploy root directory")
        sys.exit(1)

    # Find affected tests
    affected = get_affected_tests(changed_files, ci_mapping, operator_mapping)

    # Print recommendations
    print_recommendations(affected, ci_mapping)


if __name__ == "__main__":
    main()
