from .interface import Grid, Point


class CellMemory:
    """Память одного эпизода, обновляемая только окнами восприятия."""
    def __init__(self, width: int, height: int, services: dict[Point, str]):
        self.grid = tuple("".join(services.get((x, y), "?") for x in range(width))
                          for y in range(height))

    def update(self, position: Point, window: Grid) -> Grid:
        x, y = position
        rows = list(self.grid)
        left = max(0, x - 1)
        right = min(len(rows[0]), x + 2)
        for j in range(max(0, y - 1), min(len(rows), y + 2)):
            visible = window[j - y + 1][left - x + 1:right - x + 1]
            rows[j] = rows[j][:left] + visible + rows[j][right:]
        self.grid = tuple(rows)
        return self.grid
