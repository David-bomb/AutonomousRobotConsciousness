"""Защита выделенной выборки от записи общим генератором с другим рецептом."""
import sqlite3

from .corpus import DB
from .session import log


def install():
    with sqlite3.connect(DB, timeout=90) as db:
        wrong = db.execute(
            "SELECT count(*) FROM geoms WHERE split='shift_development' "
            "AND seed NOT BETWEEN 800000000000 + 1000*idx AND 800000000999 + 1000*idx"
        ).fetchone()[0]
        if wrong:
            raise ValueError("В выборке уже есть карты другого генератора")
        db.execute("""
            CREATE TRIGGER IF NOT EXISTS validate_shift_recipe
            BEFORE INSERT ON geoms
            WHEN NEW.split='shift_development' AND NEW.seed NOT BETWEEN
                800000000000 + 1000*NEW.idx AND 800000000999 + 1000*NEW.idx
            BEGIN
                SELECT RAISE(ROLLBACK, 'Required registered shift generator');
            END
        """)
    log("shift_recipe_guard", verified_existing=True)


if __name__ == "__main__":
    install()
