"""Asset-class map used by group caps. Unknown symbols fall into 'other'.

'cash' (T-bill ETFs) is never capped.
"""
ASSET_CLASS: dict[str, str] = {
    # US equity
    **{s: "us_equity" for s in [
        "SPY", "QQQ", "IVV", "VOO", "IWM", "IJT", "IJS", "RSP", "DIA", "MDY", "VTI", "MTUM", "QUAL", "USMV",
        "VTV", "VUG", "IWD", "IWF", "XLE", "XLK", "XLV", "XLF", "XLI", "XLP", "XLU", "XLY", "XLB", "XLC", "XLRE",
        "ICLN",
    ]},
    # international equity
    **{s: "intl_equity" for s in [
        "EFA", "VEU", "EEM", "VWO", "EWJ", "EWY", "EWT", "FXI", "KWEB", "ACWX", "IEFA", "IEMG", "EWZ", "INDA",
    ]},
    # bonds
    **{s: "bonds" for s in ["IEF", "TLT", "AGG", "BND", "SHY", "LQD", "HYG", "TIP", "GOVT", "VGIT", "EMB"]},
    # commodities
    **{s: "commodities" for s in ["GLD", "IAU", "DBC", "SLV", "USO", "PDBC", "GSG", "DBA"]},
    # real estate
    **{s: "real_estate" for s in ["VNQ", "IYR", "SCHH"]},
    # cash-like
    **{s: "cash" for s in ["BIL", "SHV", "SGOV"]},
}

UNCAPPED_GROUPS = {"cash"}


def asset_class(symbol: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and symbol in overrides:
        return overrides[symbol]
    return ASSET_CLASS.get(symbol, "other")
