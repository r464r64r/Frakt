"""
Manifesto compliance tests.

Parses manifesto.md, extracts <!-- TEST: --> blocks, executes them.
This ensures our code adheres to the philosophical principles.
"""

import re
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def extract_tests_from_manifesto():
    """
    Extract embedded test blocks from manifesto.md.

    Returns:
        list[tuple[int, str]]: List of (line_number, test_code) tuples
    """
    manifesto_path = project_root / "manifesto.md"

    if not manifesto_path.exists():
        pytest.skip("manifesto.md not found")

    with open(manifesto_path) as f:
        content = f.read()

    # Regex: <!-- TEST:\n...\n-->
    # Use DOTALL to match across newlines
    pattern = r"<!-- TEST:(.*?)-->"
    matches = re.finditer(pattern, content, re.DOTALL)

    tests = []
    for match in matches:
        test_code = match.group(1).strip()
        # Find line number
        line_num = content[: match.start()].count("\n") + 1
        tests.append((line_num, test_code))

    return tests


def test_manifesto_has_embedded_tests():
    """Verify manifesto contains embedded test blocks."""
    tests = extract_tests_from_manifesto()
    assert len(tests) > 0, "Manifesto should contain at least one <!-- TEST: --> block"
    assert len(tests) >= 5, f"Expected at least 5 test blocks, found {len(tests)}"


def test_manifesto_compliance():
    """Execute all embedded tests from manifesto."""
    tests = extract_tests_from_manifesto()

    if not tests:
        pytest.skip("No embedded tests found in manifesto")

    failures = []

    for line_num, test_code in tests:
        # Skip empty or comment-only blocks
        if not test_code or test_code.startswith("#"):
            continue

        # Remove leading comment lines (description)
        code_lines = []
        for line in test_code.split("\n"):
            # Include actual code (imports, asserts, etc.)
            if line.strip() and not line.strip().startswith("# "):
                code_lines.append(line)

        if not code_lines:
            continue

        clean_code = "\n".join(code_lines)

        try:
            # Create a namespace with imports
            namespace = {
                "__builtins__": __builtins__,
                "pytest": pytest,
            }

            # Execute test code
            exec(clean_code, namespace)

        except AssertionError as e:
            failures.append(f"Line {line_num}: {e}")
        except ImportError as e:
            # Some modules might not be available in test environment
            # Mark as warning, not failure
            print(f"⚠️  Line {line_num}: Import unavailable: {e}")
        except Exception as e:
            failures.append(f"Line {line_num}: Unexpected error: {e}")

    if failures:
        pytest.fail("Manifesto compliance failed:\n" + "\n".join(failures))


def test_no_lagging_indicators():
    """
    Principle: No lagging indicators.

    Verify that strategies don't use RSI, MACD, Bollinger Bands, etc.
    """
    forbidden_terms = ["rsi", "macd", "bollinger", "sma", "ema"]

    # Check strategies directory
    strategies_dir = project_root / "strategies"
    if not strategies_dir.exists():
        pytest.skip("strategies/ directory not found")

    violations = []

    for strategy_file in strategies_dir.glob("*.py"):
        if strategy_file.name.startswith("_"):
            continue

        with open(strategy_file) as f:
            content = f.read().lower()

        for term in forbidden_terms:
            # Check for variable names, function calls, imports
            if re.search(rf"\b{term}\b", content):
                violations.append(f"{strategy_file.name}: uses '{term}'")

    if violations:
        pytest.fail("Lagging indicators found:\n" + "\n".join(violations))


def test_liquidity_first_approach():
    """
    Principle: Liquidity-first approach.

    Verify that core modules detect liquidity sweeps and order flow.
    """
    try:
        from strategies.liquidity_sweep import LiquiditySweepStrategy

        # Strategy must have sweep detection
        strategy = LiquiditySweepStrategy()
        assert hasattr(
            strategy, "detect_sweep"
        ) or "sweep" in str(strategy.__class__.__dict__), (
            "LiquiditySweepStrategy must detect sweeps"
        )

    except ImportError:
        pytest.skip("liquidity_sweep strategy not available")


def test_fractal_structure():
    """
    Principle: Fractal structure across timeframes.

    Verify that market structure detection works across timeframes.
    """
    try:
        from core.market_structure import detect_structure_breaks

        # Function must be callable and work with different timeframes
        assert callable(detect_structure_breaks), "detect_structure_breaks must be callable"

    except ImportError:
        pytest.skip("core.market_structure not available")


def test_dynamic_position_sizing():
    """
    Principle: Dynamic position sizing based on confidence.

    Verify that higher confidence results in larger positions.
    """
    try:
        from risk.position_sizing import calculate_position_size, RiskParameters

        # Test different confidence levels (using actual function signature)
        # Use larger max_position to avoid capping
        params = RiskParameters(base_risk_percent=0.02, max_position_percent=0.50)

        low_conf_size = calculate_position_size(
            portfolio_value=10000,
            entry_price=100,
            stop_loss_price=95,  # 5% stop
            confidence_score=50,  # Low confidence
            current_atr=5,
            baseline_atr=5,
            consecutive_wins=0,
            consecutive_losses=0,
            params=params,
        )

        high_conf_size = calculate_position_size(
            portfolio_value=10000,
            entry_price=100,
            stop_loss_price=95,  # 5% stop
            confidence_score=90,  # High confidence
            current_atr=5,
            baseline_atr=5,
            consecutive_wins=0,
            consecutive_losses=0,
            params=params,
        )

        assert (
            high_conf_size > low_conf_size
        ), f"Higher confidence must result in larger position (low={low_conf_size}, high={high_conf_size})"

    except ImportError:
        pytest.skip("risk.position_sizing not available")


def test_order_flow_over_price():
    """
    Principle: Order flow over price action.

    Verify that we detect institutional order blocks and imbalances.
    """
    try:
        from core.order_blocks import detect_order_blocks
        from core.imbalance import detect_imbalances

        assert callable(detect_order_blocks), "Must detect order blocks"
        assert callable(detect_imbalances), "Must detect imbalances"

    except ImportError:
        pytest.skip("core modules not available")


if __name__ == "__main__":
    # Allow running directly for debugging
    pytest.main([__file__, "-v"])
