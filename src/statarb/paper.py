"""Paper ledger / broker-fill reconciliation ONLY; no real order API."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timezone
import json
from .risk import KillSwitch


@dataclass(frozen=True)
class PaperFill:
    timestamp_utc: str
    symbol: str
    side: str
    expected_price: float
    simulated_price: float
    quantity: float
    fee: float

    @property
    def slippage_bps(self) -> float:
        sign = 1 if self.side.lower() == 'buy' else -1
        return sign * (self.simulated_price/self.expected_price - 1) * 10_000


class PaperLedger:
    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.events = self.folder / 'paper_fills.jsonl'
        self.kill = KillSwitch(self.folder / 'kill_switch.json')

    def record(self, fill: PaperFill) -> None:
        if self.kill.is_disabled():
            raise PermissionError('Paper-trading kill switch active: human review required')
        if fill.expected_price <= 0 or fill.simulated_price <= 0 or fill.quantity <= 0:
            raise ValueError('Invalid paper fill')
        payload = asdict(fill)
        payload['slippage_bps'] = fill.slippage_bps
        with self.events.open('a') as f:
            f.write(json.dumps(payload, sort_keys=True) + '\n')

    def health(self, equity_multiple: float, stale_seconds: float, has_open_position: bool,
               realized_slippage_bps: float, assumed_slippage_bps: float) -> dict:
        if assumed_slippage_bps <= 0:
            raise ValueError('Expected slippage must be positive')
        disabled = self.kill.observe(equity_multiple, stale_seconds, has_open_position,
                                     realized_slippage_bps / assumed_slippage_bps)
        return {'disabled': disabled, 'status': 'HALTED' if disabled else 'PAPER_OK',
                'timestamp_utc': datetime.now(timezone.utc).isoformat()}
