from .environment import Environment, Status
from .interface import Controller


def advance(env: Environment, controller: Controller):
    env.step(controller.choose(env.view))


def run(env: Environment, controller: Controller) -> Status:
    while env.status == Status.RUNNING:
        advance(env, controller)
    return env.status
