"""Публичные версии контроллера: исходный D-0008 и ресурсный E-0002."""
from .pilot_automaton import Mode, PilotAutomaton as Automaton, separation
from .resource_automaton import Automaton as ResourceAutomaton

__all__ = ["Automaton", "ResourceAutomaton", "Mode", "separation"]
