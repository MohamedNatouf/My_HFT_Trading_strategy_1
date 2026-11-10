"""Research universes.

SP100: approximate S&P 100 constituents as of 2026 mapped to their SPDR sector ETF.
CAUTION: using *today's* constituents over past data has survivorship bias (names that
fell out of the index are missing). Treat results as optimistic.
"""
SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

SP100: dict[str, str] = {
    # technology
    **{s: "XLK" for s in ["AAPL", "ACN", "ADBE", "AMD", "AVGO", "CRM", "CSCO", "IBM", "INTC", "INTU", "MSFT",
                          "NOW", "NVDA", "ORCL", "PLTR", "QCOM", "TXN"]},
    # communication
    **{s: "XLC" for s in ["CHTR", "CMCSA", "DIS", "GOOGL", "META", "NFLX", "T", "TMUS", "VZ"]},
    # consumer discretionary
    **{s: "XLY" for s in ["AMZN", "BKNG", "F", "GM", "HD", "LOW", "MCD", "NKE", "SBUX", "TGT", "TSLA", "UBER"]},
    # consumer staples
    **{s: "XLP" for s in ["CL", "COST", "KO", "MDLZ", "MO", "PEP", "PG", "PM", "WMT"]},
    # financials
    **{s: "XLF" for s in ["AIG", "AXP", "BAC", "BK", "BLK", "BRK.B", "C", "COF", "GS", "JPM", "MA", "MET", "MS",
                          "PYPL", "SCHW", "USB", "V", "WFC"]},
    # health care
    **{s: "XLV" for s in ["ABBV", "ABT", "AMGN", "BMY", "CVS", "DHR", "GILD", "ISRG", "JNJ", "LLY", "MDT", "MRK",
                          "PFE", "TMO", "UNH"]},
    # industrials
    **{s: "XLI" for s in ["BA", "CAT", "DE", "EMR", "FDX", "GD", "GE", "HON", "LMT", "MMM", "RTX", "UNP", "UPS"]},
    # energy
    **{s: "XLE" for s in ["COP", "CVX", "XOM"]},
    # utilities
    **{s: "XLU" for s in ["DUK", "NEE", "SO"]},
    # real estate
    **{s: "XLRE" for s in ["AMT", "SPG"]},
    # materials
    **{s: "XLB" for s in ["LIN"]},
}
