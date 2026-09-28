"""In-memory stock keeping."""


class OutOfStockError(Exception):
    """Raised when removing more units than are available."""


class Inventory:
    """Tracks stock levels per SKU."""

    def __init__(self):
        self._stock: dict[str, int] = {}

    def add_item(self, sku: str, quantity: int) -> None:
        """Increase stock for a SKU; quantity must be positive."""
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        self._stock[sku] = self._stock.get(sku, 0) + quantity

    def remove_item(self, sku: str, quantity: int) -> None:
        """Decrease stock, raising OutOfStockError if not enough units exist."""
        available = self._stock.get(sku, 0)
        if quantity > available:
            raise OutOfStockError(f"{sku}: requested {quantity}, only {available} left")
        self._stock[sku] = available - quantity

    def get_stock(self, sku: str) -> int:
        return self._stock.get(sku, 0)

    def low_stock_items(self, threshold: int = 5) -> list[str]:
        """Return SKUs whose stock is below the threshold, sorted alphabetically."""
        return sorted(sku for sku, qty in self._stock.items() if qty < threshold)
