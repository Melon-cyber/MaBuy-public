class SymbolNormalizer:
    """
    Domain Utility for robust cross-market asset symbol normalization.
    Maps variations like 'btc', 'BTC/USDT', 'bitcoin' -> 'BTC',
    and 'nvidia', 'apple', 'sp500', 'nasdaq' -> 'NVDA', 'AAPL', 'SPY', 'QQQ'.
    """

    ALIAS_MAP = {
        # Crypto
        "BITCOIN": "BTC",
        "БИТКОИН": "BTC",
        "БИТОК": "BTC",
        "ETHEREUM": "ETH",
        "ЭФИР": "ETH",
        "ЭФИРИУМ": "ETH",
        "SOLANA": "SOL",
        "СОЛАНА": "SOL",
        "СОЛ": "SOL",
        "BINANCECOIN": "BNB",
        "БНБ": "BNB",
        "RIPPLE": "XRP",
        "РИПЛ": "XRP",
        "CARDANO": "ADA",
        "КАРДАНО": "ADA",
        "DOGECOIN": "DOGE",
        "ДОГИ": "DOGE",
        "ДОЖ": "DOGE",
        "AVALANCHE": "AVAX",
        "АВАКС": "AVAX",
        "TONCOIN": "TON",
        "ТОН": "TON",
        "ТОНКОИН": "TON",
        # US Equities
        "NVIDIA": "NVDA",
        "НВИДИА": "NVDA",
        "APPLE": "AAPL",
        "ЭППЛ": "AAPL",
        "TESLA": "TSLA",
        "ТЕСЛА": "TSLA",
        "MICROSOFT": "MSFT",
        "МАЙКРОСОФТ": "MSFT",
        "ALPHABET": "GOOGL",
        "GOOGLE": "GOOGL",
        "ГУГЛ": "GOOGL",
        # Macro ETFs
        "SP500": "SPY",
        "СП500": "SPY",
        "SP_500": "SPY",
        "SANDP500": "SPY",
        "NASDAQ": "QQQ",
        "НАСДАК": "QQQ",
        "NASDAQ100": "QQQ",
        "GOLD": "GLD",
        "ЗОЛОТО": "GLD",
        "SPDR_GOLD": "GLD",
    }

    @classmethod
    def normalize(cls, raw_symbol: str) -> str:
        if not raw_symbol:
            return "BTC"

        clean = raw_symbol.strip().upper()
        clean = clean.replace("/", "").replace("-", "").replace("_", "")

        # Strip USDT suffix if present for base coin evaluation
        if clean.endswith("USDT") and len(clean) > 4:
            clean = clean[:-4]

        # Map common full names
        return cls.ALIAS_MAP.get(clean, clean)

