from types import SimpleNamespace

from quant.execution import PositionInfo, Quote, RebalanceConfig, Rebalancer


class FakeBroker:
    def __init__(self, equity=10_000.0, positions=None, prices=None, fractionable=True):
        self._equity = equity
        self._pos = dict(positions or {})
        self.prices = prices or {}
        self.frac = fractionable
        self.submitted = []
        self.cancelled = []
        self.orders_by_cid = {}

    def equity(self):
        return self._equity

    def buying_power(self):
        return self._equity

    def positions(self):
        return {s: PositionInfo(s, q, q * self.prices[s], self.prices[s]) for s, q in self._pos.items()}

    def is_open(self):
        return True

    def latest_prices(self, symbols):
        return {s: self.prices[s] for s in symbols}

    def latest_quotes(self, symbols):
        return {s: Quote(self.prices[s] * 0.9999, self.prices[s] * 1.0001) for s in symbols}

    def fractionable(self, s):
        return self.frac

    def open_orders(self):
        return [SimpleNamespace(symbol="AAA", id="stale-1")]

    def cancel_order(self, oid):
        self.cancelled.append(oid)

    def order_exists(self, cid):
        return cid in self.orders_by_cid

    def submit(self, spec):
        self.submitted.append(spec)
        self.orders_by_cid[spec.client_order_id] = spec
        # simulate an instant fill
        signed = spec.qty if spec.side == "buy" else -spec.qty
        self._pos[spec.symbol] = self._pos.get(spec.symbol, 0.0) + signed
        return SimpleNamespace(id=f"oid-{len(self.submitted)}")


PRICES = {"AAA": 100.0, "BBB": 50.0, "CCC": 20.0, "USER": 10.0}


def test_plan_reaches_targets_and_is_idempotent():
    b = FakeBroker(prices=PRICES)
    r = Rebalancer(b, RebalanceConfig(order_style="market"), tag="t")
    targets = {"AAA": 0.5, "BBB": 0.3}
    res = r.execute(r.plan(targets, {"AAA", "BBB", "CCC"}, day="20260101"), dry_run=False)
    assert {x["status"] for x in res} == {"submitted"}
    assert abs(b._pos["AAA"] * 100 - 5000) < 1e-6 and abs(b._pos["BBB"] * 50 - 3000) < 1e-6
    # second run: positions already at target -> nothing to do
    plan2 = r.plan(targets, {"AAA", "BBB", "CCC"}, day="20260101")
    assert plan2.orders == []


def test_duplicate_client_order_id_not_resubmitted():
    b = FakeBroker(prices=PRICES)
    r = Rebalancer(b, RebalanceConfig(order_style="market"), tag="t")
    plan = r.plan({"AAA": 0.5}, {"AAA"}, day="20260101")
    r.execute(plan, dry_run=False)
    b._pos.clear()  # pretend fills not yet visible: same plan again must not double-trade
    res = r.execute(r.plan({"AAA": 0.5}, {"AAA"}, day="20260101"), dry_run=False)
    assert res[0]["status"] == "duplicate_skipped"
    assert len(b.submitted) == 1


def test_exits_dropped_names_sells_first_and_leaves_unmanaged_alone():
    b = FakeBroker(positions={"CCC": 100.0, "USER": 50.0}, prices=PRICES)
    r = Rebalancer(b, RebalanceConfig(order_style="market"), tag="t")
    plan = r.plan({"AAA": 0.5}, managed={"AAA", "CCC"}, day="20260102")
    syms = [(o.symbol, o.side) for o in plan.orders]
    assert syms[0] == ("CCC", "sell") and ("AAA", "buy") in syms
    assert all(o.symbol != "USER" for o in plan.orders)
    assert plan.orders[0].qty == 100.0


def test_small_deltas_skipped_and_whole_shares_for_moc():
    b = FakeBroker(positions={"AAA": 50.0}, prices=PRICES)
    r = Rebalancer(b, RebalanceConfig(order_style="moc", drift_band=0.01), tag="t")
    plan = r.plan({"AAA": 0.505, "BBB": 0.1234}, {"AAA", "BBB"}, day="20260103")
    assert [o.symbol for o in plan.orders] == ["BBB"]
    o = plan.orders[0]
    assert o.qty == int(o.qty) == 24 and o.tif == "cls"


def test_dry_run_sends_nothing_and_limit_prices_set():
    b = FakeBroker(prices=PRICES)
    r = Rebalancer(b, RebalanceConfig(order_style="limit", limit_slippage_bps=10), tag="t")
    res = r.execute(r.plan({"AAA": 0.5}, {"AAA"}, day="20260104"), dry_run=True)
    assert b.submitted == [] and b.cancelled == []
    assert res[0]["status"] == "dry_run" and res[0]["limit"] > 100.0
