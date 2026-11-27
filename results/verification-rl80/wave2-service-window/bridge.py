"""Публичная точка load(bytes) из замороженных компонентов второй волны."""
from importlib import import_module

FROZEN="results.verification-rl80.wave2-service-window.frozen"
IMPORTS={"results.E-0016.prev-two.adapter":FROZEN+".prev-two.adapter",
         "results.E-0016.sequence.belief.joint_model":FROZEN+".sequence.belief.joint_model"}
adapter=import_module(FROZEN+".service_window_policy")
adapter.import_module=lambda name: import_module(IMPORTS[name])
load=adapter.load
