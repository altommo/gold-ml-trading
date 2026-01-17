"""
Configuration Manager

Flexible configuration system for strategy parameters.
Load/save configs, compare different setups, run parameter sweeps.
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import copy


@dataclass
class IndicatorConfig:
    """Configuration for indicators"""
    # Wolfpack ID
    wolfpack_fast: int = 3
    wolfpack_slow: int = 8

    # WaveTrend
    wt_channel_length: int = 9
    wt_average_length: int = 12
    wt_ma_length: int = 3
    wt_overbought: int = 53
    wt_oversold: int = -53

    # RSI
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30

    # Money Flow
    mfi_period: int = 60
    mfi_multiplier: float = 150.0


@dataclass
class RangeFilterConfig:
    """Configuration for Range Filter (Potato Signal)"""
    sampling_period: int = 27
    range_multiplier: float = 1.0
    source_type: str = 'close'  # 'close', 'wolfpack', 'wavetrend'
    use_heikin_ashi: bool = False


@dataclass
class ConfirmationConfig:
    """Configuration for signal confirmation logic"""
    require_wolfpack: bool = True
    require_wavetrend: bool = True
    require_money_flow: bool = False
    require_rsi: bool = False

    # WaveTrend confirmation mode
    wt_mode: str = 'trend'  # 'trend', 'cross', 'level'

    # Minimum signal strength to take trade (1-3)
    min_signal_strength: int = 1

    # Anti-whipsaw: minimum bars between trades
    min_bars_between_trades: int = 0


@dataclass
class RiskConfig:
    """Configuration for risk management"""
    # Position sizing
    position_size_pct: float = 0.1  # % of capital per trade
    max_positions: int = 1  # Max concurrent positions

    # Stop Loss
    use_stop_loss: bool = True
    stop_loss_type: str = 'percent'  # 'percent', 'atr', 'fixed'
    stop_loss_value: float = 2.0  # Percentage or ATR multiplier or fixed points

    # Take Profit
    use_take_profit: bool = True
    take_profit_type: str = 'percent'  # 'percent', 'atr', 'fixed', 'rr_ratio'
    take_profit_value: float = 3.0  # Value based on type

    # Trailing stop
    use_trailing_stop: bool = False
    trailing_stop_value: float = 1.5

    # Risk/Reward based TP
    risk_reward_ratio: float = 1.5  # If using rr_ratio type


@dataclass
class DataConfig:
    """Configuration for data source"""
    source: str = 'tradingview'  # 'tradingview', 'yahoo', 'csv'
    symbol: str = 'XAUUSD'
    exchange: str = 'OANDA'
    interval: str = '1h'
    n_bars: int = 5000
    csv_path: Optional[str] = None


@dataclass
class StrategyConfig:
    """Complete strategy configuration"""
    name: str = 'default'
    description: str = ''

    # Sub-configs
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    range_filter: RangeFilterConfig = field(default_factory=RangeFilterConfig)
    confirmation: ConfirmationConfig = field(default_factory=ConfirmationConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Trading
    initial_capital: float = 10000.0
    spread_pct: float = 0.03
    allow_shorting: bool = True

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'StrategyConfig':
        """Create from dictionary"""
        config = cls()

        if 'name' in data:
            config.name = data['name']
        if 'description' in data:
            config.description = data['description']
        if 'initial_capital' in data:
            config.initial_capital = data['initial_capital']
        if 'spread_pct' in data:
            config.spread_pct = data['spread_pct']
        if 'allow_shorting' in data:
            config.allow_shorting = data['allow_shorting']

        # Sub-configs
        if 'indicators' in data:
            for key, value in data['indicators'].items():
                if hasattr(config.indicators, key):
                    setattr(config.indicators, key, value)

        if 'range_filter' in data:
            for key, value in data['range_filter'].items():
                if hasattr(config.range_filter, key):
                    setattr(config.range_filter, key, value)

        if 'confirmation' in data:
            for key, value in data['confirmation'].items():
                if hasattr(config.confirmation, key):
                    setattr(config.confirmation, key, value)

        if 'risk' in data:
            for key, value in data['risk'].items():
                if hasattr(config.risk, key):
                    setattr(config.risk, key, value)

        if 'data' in data:
            for key, value in data['data'].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        return config


class ConfigManager:
    """Manage strategy configurations"""

    def __init__(self, config_dir: str = './configs'):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

    def save(self, config: StrategyConfig, filename: Optional[str] = None) -> Path:
        """Save configuration to file"""
        if filename is None:
            filename = f"{config.name}.json"

        filepath = self.config_dir / filename

        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        print(f"Config saved to: {filepath}")
        return filepath

    def load(self, filename: str) -> StrategyConfig:
        """Load configuration from file"""
        filepath = self.config_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return StrategyConfig.from_dict(data)

    def list_configs(self) -> List[str]:
        """List available configurations"""
        return [f.name for f in self.config_dir.glob('*.json')]

    def create_variant(
        self,
        base_config: StrategyConfig,
        name: str,
        **overrides
    ) -> StrategyConfig:
        """Create a variant of a config with specific overrides"""
        variant = copy.deepcopy(base_config)
        variant.name = name
        variant.created_at = datetime.now().isoformat()

        # Apply overrides
        for key, value in overrides.items():
            # Handle nested keys like 'risk.stop_loss_pct'
            if '.' in key:
                parts = key.split('.')
                obj = variant
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            elif hasattr(variant, key):
                setattr(variant, key, value)

        return variant


def create_preset_configs() -> Dict[str, StrategyConfig]:
    """Create preset configurations for common setups"""
    presets = {}

    # Conservative - fewer trades, higher confirmation
    conservative = StrategyConfig(
        name='conservative',
        description='Conservative setup - fewer trades, all confirmations required'
    )
    conservative.confirmation.require_wolfpack = True
    conservative.confirmation.require_wavetrend = True
    conservative.confirmation.min_signal_strength = 3
    conservative.confirmation.min_bars_between_trades = 5
    conservative.risk.stop_loss_value = 1.5
    conservative.risk.take_profit_value = 3.0
    presets['conservative'] = conservative

    # Aggressive - more trades, relaxed confirmation
    aggressive = StrategyConfig(
        name='aggressive',
        description='Aggressive setup - more trades, relaxed confirmations'
    )
    aggressive.confirmation.require_wolfpack = True
    aggressive.confirmation.require_wavetrend = False
    aggressive.confirmation.min_signal_strength = 1
    aggressive.risk.stop_loss_value = 2.5
    aggressive.risk.take_profit_value = 2.0
    presets['aggressive'] = aggressive

    # Wolfpack Source - use Wolfpack ID as Range Filter source
    wolfpack_source = StrategyConfig(
        name='wolfpack_source',
        description='Uses Wolfpack ID output as Range Filter source'
    )
    wolfpack_source.range_filter.source_type = 'wolfpack'
    wolfpack_source.confirmation.require_wolfpack = True
    wolfpack_source.confirmation.require_wavetrend = True
    presets['wolfpack_source'] = wolfpack_source

    # WaveTrend Source - use WaveTrend as Range Filter source
    wavetrend_source = StrategyConfig(
        name='wavetrend_source',
        description='Uses WaveTrend output as Range Filter source'
    )
    wavetrend_source.range_filter.source_type = 'wavetrend'
    wavetrend_source.confirmation.require_wolfpack = True
    wavetrend_source.confirmation.require_wavetrend = True
    presets['wavetrend_source'] = wavetrend_source

    # 4H Timeframe optimized
    four_hour = StrategyConfig(
        name='4h_optimized',
        description='Optimized for 4H timeframe'
    )
    four_hour.data.interval = '4h'
    four_hour.range_filter.sampling_period = 14  # Shorter for 4H
    four_hour.confirmation.min_bars_between_trades = 2
    four_hour.risk.stop_loss_value = 2.5
    four_hour.risk.take_profit_value = 4.0
    presets['4h_optimized'] = four_hour

    # Scalping - for lower timeframes
    scalping = StrategyConfig(
        name='scalping',
        description='Quick scalping setup for 15m-1H'
    )
    scalping.range_filter.sampling_period = 14
    scalping.range_filter.range_multiplier = 0.75
    scalping.risk.stop_loss_value = 1.0
    scalping.risk.take_profit_value = 1.5
    scalping.confirmation.min_bars_between_trades = 3
    presets['scalping'] = scalping

    return presets


def print_config(config: StrategyConfig):
    """Print configuration in readable format"""
    print(f"\n{'='*60}")
    print(f"STRATEGY: {config.name}")
    print(f"{'='*60}")
    if config.description:
        print(f"Description: {config.description}")

    print(f"\n[Data]")
    print(f"  Symbol: {config.data.symbol} ({config.data.exchange})")
    print(f"  Interval: {config.data.interval}")

    print(f"\n[Range Filter]")
    print(f"  Sampling Period: {config.range_filter.sampling_period}")
    print(f"  Range Multiplier: {config.range_filter.range_multiplier}")
    print(f"  Source: {config.range_filter.source_type}")

    print(f"\n[Indicators]")
    print(f"  Wolfpack: {config.indicators.wolfpack_fast}/{config.indicators.wolfpack_slow}")
    print(f"  WaveTrend: {config.indicators.wt_channel_length}/{config.indicators.wt_average_length}")

    print(f"\n[Confirmation]")
    print(f"  Require Wolfpack: {config.confirmation.require_wolfpack}")
    print(f"  Require WaveTrend: {config.confirmation.require_wavetrend}")
    print(f"  Min Strength: {config.confirmation.min_signal_strength}")
    print(f"  Min Bars Between: {config.confirmation.min_bars_between_trades}")

    print(f"\n[Risk]")
    print(f"  Position Size: {config.risk.position_size_pct*100:.1f}%")
    print(f"  Stop Loss: {config.risk.stop_loss_value}% ({config.risk.stop_loss_type})")
    print(f"  Take Profit: {config.risk.take_profit_value}% ({config.risk.take_profit_type})")

    print(f"\n[Capital]")
    print(f"  Initial: ${config.initial_capital:,.2f}")
    print(f"  Spread: {config.spread_pct}%")
    print(f"  Allow Short: {config.allow_shorting}")


if __name__ == "__main__":
    # Create and save preset configs
    manager = ConfigManager()
    presets = create_preset_configs()

    print("Available Preset Configurations:")
    print("-" * 40)

    for name, config in presets.items():
        print_config(config)
        manager.save(config)

    print(f"\nConfigs saved to: {manager.config_dir}")
