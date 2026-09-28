"""Price calculations."""

TAX_RATE = 0.08


def apply_discount(price: float, percent: float) -> float:
    """Reduce price by a percentage between 0 and 100, rounded to cents."""
    if not 0 <= percent <= 100:
        raise ValueError("percent must be between 0 and 100")
    return round(price * (1 - percent / 100), 2)


def calculate_tax(amount: float, rate: float = TAX_RATE) -> float:
    """Sales tax for an amount, rounded to cents."""
    return round(amount * rate, 2)


def bulk_discount_rate(quantity: int) -> float:
    """Percent discount for buying in bulk: 10% from 100 units, 5% from 20 units."""
    if quantity >= 100:
        return 10.0
    if quantity >= 20:
        return 5.0
    return 0.0
