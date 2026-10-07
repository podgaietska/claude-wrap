import pytest

from wrap.config import REPO_ROOT, load_config
from wrap.telemetry.pricing import ModelPricing, PricingTable
from wrap.telemetry.usage import Usage

HAIKU = ModelPricing(input=1.0, output=5.0, cache_read=0.10, cache_write_5m=1.25, cache_write_1h=2.0)
SONNET = ModelPricing(input=2.0, output=10.0, cache_read=0.20, cache_write_5m=2.5, cache_write_1h=4.0)
SONNET_55 = ModelPricing(input=2.0, output=10.0, cache_read=0.20, cache_write_5m=2.5, cache_write_1h=4.0)


@pytest.fixture
def table() -> PricingTable:
    return PricingTable({"claude-haiku-4-5": HAIKU, "claude-sonnet-5": SONNET, "claude-sonnet-5-5": SONNET_55})


def test_exact_lookup(table):
    assert table.lookup("claude-sonnet-5") is SONNET


def test_dated_id_resolves_by_prefix(table):
    assert table.lookup("claude-haiku-4-5-20251001") is HAIKU


def test_longest_prefix_wins(table):
    assert table.lookup("claude-sonnet-5-5-20260901") is SONNET_55


def test_unknown_model_has_no_price(table):
    assert table.lookup("claude-unknown") is None
    assert table.cost("claude-unknown", Usage(input_tokens=100)) is None


def test_cost_prices_every_token_kind(table):
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_read_tokens=1_000_000)

    assert table.cost("claude-haiku-4-5", usage) == pytest.approx(1.0 + 5.0 + 0.10)


def test_cache_writes_use_the_5m_1h_split(table):
    usage = Usage(cache_creation_tokens=3_000_000, cache_creation_5m_tokens=1_000_000, cache_creation_1h_tokens=2_000_000)

    assert table.cost("claude-haiku-4-5", usage) == pytest.approx(1.25 + 2 * 2.0)


def test_cache_writes_without_a_split_are_priced_at_5m(table):
    usage = Usage(cache_creation_tokens=2_000_000)

    assert table.cost("claude-haiku-4-5", usage) == pytest.approx(2 * 1.25)


def test_shipped_pricing_file_prices_every_configured_tier():
    config = load_config()
    shipped = PricingTable.load(REPO_ROOT / config.telemetry.pricing_file)

    for tier in config.tiers.values():
        assert shipped.lookup(tier.model) is not None, tier.model
