"""Orders built from inventory and pricing."""
from dataclasses import dataclass, field

from inventory import Inventory, OutOfStockError
from pricing import apply_discount, bulk_discount_rate, calculate_tax


@dataclass
class Order:
    """A customer order made of (sku, quantity, unit_price) lines."""

    lines: list[tuple[str, int, float]] = field(default_factory=list)

    def add_line(self, sku: str, quantity: int, unit_price: float) -> None:
        self.lines.append((sku, quantity, unit_price))

    def total(self) -> float:
        """Sum of lines after bulk discounts, plus sales tax."""
        subtotal = sum(
            apply_discount(qty * price, bulk_discount_rate(qty)) for _, qty, price in self.lines
        )
        return round(subtotal + calculate_tax(subtotal), 2)


def place_order(inventory: Inventory, order: Order) -> float:
    """Reserve stock for every line, rolling back if any line is out of stock."""
    reserved = []
    try:
        for sku, qty, _ in order.lines:
            inventory.remove_item(sku, qty)
            reserved.append((sku, qty))
    except OutOfStockError:
        for sku, qty in reserved:
            inventory.add_item(sku, qty)
        raise
    return order.total()
