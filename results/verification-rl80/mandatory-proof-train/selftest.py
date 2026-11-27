"""Проверка границы двух регистров на одинаковом текущем Observation."""
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]))
import adapter
from maze.environment import Environment
from maze.interface import Move
from maze.levels import Level
from maze.pilot_automaton import Mode
from maze.resource_automaton import known_paths


def test():
    env = Environment(Level.parse("Синтетика\n20 20\n#########\n#A..F.RB#\n#########"))
    first = env.view
    middle = env.step(Move.EAST)
    # Текущий снимок один и тот же. Отличаться может любой из двух
    # прежних датчиков. Для крайних случаев допускается отсутствие показания.
    streams = [(3,5),(4,5),(3,6),(None,5),(3,None)]
    original = adapter.MandatoryProof
    called = []

    class Spy(original):
        def choose(self, view):
            assert all(fn.cache_info().currsize == 0 for fn in adapter.WRAPPERS)
            assert self.standby is self.ready is self.offer is self._unseen is None
            assert self._service_streak == 0 and not self._fallback.returning
            called.append((self, view, self._fallback.last_lengths.copy()))
            known_paths(view.memory,view.position)
            assert known_paths.cache_info().currsize > 0
            self.standby = self.ready = self.offer = self._unseen = 7
            self._fallback.last_lengths.clear()
            self._fallback.returning = True
            return Move.EAST

    with patch.object(adapter,"MandatoryProof",Spy):
        for f,r in streams:
            bot = adapter.Previous()
            bot.choose(replace(first,fuel_distance=f,repair_distance=r))
            bot.choose(middle)
            bot.choose(middle)
            assert called[-3][2] == {}
            assert called[-2][1] is middle and called[-2][2] == {Mode.FUEL:f,Mode.REPAIR:r}
            assert called[-1][2] == {Mode.FUEL:middle.fuel_distance,Mode.REPAIR:middle.repair_distance}
            assert vars(bot) == {"previous":(middle.fuel_distance,middle.repair_distance)}
    assert len({id(agent) for agent,_,_ in called}) == 3*len(streams)
    # Контроль с оригинальным choose и намеренно заполненным кэшем.
    for f,r in streams:
        bot = adapter.Previous()
        bot.choose(replace(first,fuel_distance=f,repair_distance=r))
        actual = bot.choose(middle)
        for fn in adapter.WRAPPERS:
            fn.cache_clear()
        ref = original()
        ref._fallback.last_lengths = {Mode.FUEL:f,Mode.REPAIR:r}
        assert actual is ref.choose(middle)
    print("OK: 15 вызовов с проверкой полей и пустых кэшей, 5 сверок штатного choose.")


if __name__ == "__main__":
    test()
