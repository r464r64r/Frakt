# Frakt

SMC-based trading engine. Liquidity-first, fractal structure, no lagging indicators.

## Philosophy

See [manifesto.md](manifesto.md) for our core principles.

## Quick Start

```python
from frakt.core.market_structure import detect_bos, detect_choch
from frakt.core.order_blocks import detect_order_blocks
from frakt.strategies.liquidity_sweep import LiquiditySweepStrategy

# Load your data (OHLCV DataFrame)
ohlcv = ...

# Generate signals
strategy = LiquiditySweepStrategy()
signal = strategy.generate_signal(ohlcv)

if signal and signal.confidence > 0.8:
    print(f"Signal: {signal.side} at {signal.entry_price}")
    print(f"Confidence: {signal.confidence}")
```

## Installation

```bash
# Install from source
git clone https://github.com/r464r64r/Frakt.git
cd Frakt
pip install -e .
```

## Core Components

### 📊 Smart Money Concepts (core/)
- **Market Structure**: BOS (Break of Structure), CHoCH (Change of Character)
- **Order Blocks**: Institutional accumulation/distribution zones
- **Fair Value Gaps**: Imbalances in order flow
- **Liquidity Analysis**: Sweep detection, liquidity pools

### 🎯 Strategies (strategies/)
- **Liquidity Sweep**: Trade with institutions during stop hunts
- **FVG Fill**: Mean reversion to fair value gaps
- **Order Block Retest**: Trend continuation from institutional zones

### 🛡️ Risk Management (risk/)
- **Position Sizing**: Dynamic sizing based on confidence scores
- **Confidence Factors**: Multi-factor risk assessment

### 📈 Backtesting (backtesting/)
- Built on vectorbt for high-performance backtesting
- Multi-timeframe strategy evaluation

## Architecture

Frakt is designed as a **pure engine** - it provides:
- ✅ SMC detection algorithms
- ✅ Trading strategies (signal generation)
- ✅ Risk management tools
- ✅ Backtesting framework

It does **not** provide:
- ❌ Exchange API credentials
- ❌ Live order execution
- ❌ Position management
- ❌ Infrastructure/deployment

For a complete trading platform, see [FraktAl](https://github.com/r464r64r/FraktAl) (private).

## Testing

```bash
pytest tests/ -v
```

Coverage:
- Core modules: 95%+
- Strategies: 70%+
- Risk management: 98%

## Contributing

We welcome contributions! Please:

1. Read [manifesto.md](manifesto.md) to understand our philosophy
2. Ensure tests pass and coverage remains high
3. Follow our principles:
   - No lagging indicators
   - Liquidity-first approach
   - Fractal structure across timeframes

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

MIT - see [LICENSE](LICENSE)

---

**"There is no price. There is bid/ask, order flow, liquidity distribution."**
— Frakt Manifesto

🌀
