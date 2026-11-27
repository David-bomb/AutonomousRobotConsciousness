"""Адаптер волны с последовательностями сервисов; true map и учебные метки отсутствуют."""
from importlib import import_module
from io import BytesIO
import torch


def load(checkpoint:bytes):
    policy=import_module("results.E-0016.sequence.model")
    moves=import_module("maze.interface").Move
    state=torch.load(BytesIO(checkpoint),weights_only=True,map_location="cpu")
    kind="rich" if "base.wall_prior" in state else "critical" if "base.risk.0.weight" in state else "plain"
    torch.set_num_threads(1)
    net=policy.SequenceNet(kind).eval()
    net.load_state_dict(state)
    net.requires_grad_(False)
    prior=net.wall_prior.numpy().tobytes()
    actions=tuple(moves)
    class Controller:
        @torch.inference_mode()
        def choose(self,view):
            x,mask=policy.encode(view,prior)
            return actions[int(net(torch.from_numpy(x[None]),torch.from_numpy(mask[None])).argmax())]
    return Controller()
