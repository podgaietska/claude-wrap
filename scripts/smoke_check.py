"""Checks an installed claude-wrap from the inside; run by `smoke_test.sh`."""

from pathlib import Path

import wrap
from wrap.config import load_config
from wrap.dashboard.app import STATIC_DIR
from wrap.telemetry.pricing import load_pricing

assert "site-packages" in Path(wrap.__file__).parts, f"wrap imported from the source tree: {wrap.__file__}"

config = load_config()
pricing = load_pricing(config.telemetry)
for tier in config.tiers.values():
    assert pricing.lookup(tier.model), f"no packaged price for tier model {tier.model}"
assert (STATIC_DIR / "index.html").exists(), "dashboard page missing from the wheel"

print(f"checked wrap {wrap.__version__} at {Path(wrap.__file__).parent}")
