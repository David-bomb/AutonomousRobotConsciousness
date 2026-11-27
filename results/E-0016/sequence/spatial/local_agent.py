"""Восемь направлений из одного текущего наблюдения с локальным прогнозом."""
from io import BytesIO
import numpy as np
import torch
from ...policy import geometry
from ..belief.joint_model import Joint, MOVES
from ..belief.paths import confidence
from .local_model import LocalNet, local_image
from .prefix import Prefix


class LocalAgent:
    def __init__(self, pack):
        self.base = Joint(pack["foundation"])
        self.base_weights = pack["foundation"]["linear"].numpy()
        self.local = LocalNet().eval().requires_grad_(False)
        self.local.load_state_dict(pack["local"])
        self.prefix = Prefix(pack["prefix"].numpy().view(np.uint32))

    @torch.inference_mode()
    def directions(self, view):
        base, legal = self.base.directions(view)
        result = np.zeros((8, 10), np.float64)
        result[:, 0] = base @ self.base_weights
        if legal.sum() < 2 or base[:, 1:5].any():
            return result, legal
        observed = np.array((view.fuel_distance, view.repair_distance))
        stocks = np.array((view.fuel, view.health))
        slack = stocks - observed
        urgent = int(slack.argmin())
        if slack[urgent] >= 10:
            return result, legal
        strength = np.clip((10 - slack[urgent])/5., 0., 1.)
        patch, meta = local_image(view)
        previous = MOVES.index(view.last_move) if view.last_move else 8
        signal, quality, mask = self.local(torch.from_numpy(patch[None]), torch.from_numpy(meta[None]),
            torch.from_numpy(stocks[None]), torch.tensor([previous]))
        assert np.array_equal(mask[0].numpy(), legal)
        result[:, 1] = base[:, 7:] @ self.base_weights[7:]
        result[:, 2] = signal[0, :, urgent].softmax(-1)[:, 0].numpy()
        result[:, 3:5] = quality[0].sigmoid().numpy()
        result[:, 5:7] = self.prefix.score(view)[:, :2]
        trial = confidence(view, self.base.tables)
        result[:, 7] = trial[:, 1]
        result[:, 8] = trial[:, 0] > 0
        goal = (view.position[0] + view.dx, view.position[1] + view.dy)
        _, known = geometry(view.memory, view.position, goal)
        offset = 1 + urgent * 2
        lengths = known[:, :15].reshape(8, 5, 3)[:, offset:offset+2, 0].min(1)
        result[:, 9] = lengths == observed[urgent]
        result[:, 2:] *= strength
        return result, legal


def load(checkpoint):
    torch.set_num_threads(1)
    pack = torch.load(BytesIO(checkpoint), weights_only=True, map_location="cpu")
    net = LocalAgent(pack)
    weights = pack["linear"].numpy()

    class Controller:
        def choose(self, view):
            scores, legal = net.directions(view)
            return MOVES[int(np.where(legal, scores @ weights, -1e12).argmax())]

    return Controller()
