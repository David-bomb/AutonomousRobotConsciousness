"""Точки входа двух закреплённых нейронных политик, по одной на чистый процесс."""
from hashlib import sha256
from importlib import import_module


def load(data):
    package="results.verification-rl80.wave2-service-window"
    key=sha256(data).hexdigest()
    if key=="ebf419fe5ed73a9204b8cc976bedbc1ca53aeffe615605fb443f20fb4f852f56":
        return import_module(package+".frozen.sequence.belief.joint_model").load(data)
    if key!="eb11b8ec3f1c6c2f24996e23fe9384f91b3bba33cffa8202193109911cce6f4c":
        raise ValueError("Другие веса")
    policy=import_module(package+".bridge").load(data)
    assert policy.previous==(None,None) and policy.cfg==[9,9,-101] and not policy.ablate
    cache={}
    class Fixed:
        def choose(self,view):
            m=policy.choose(view,cache)
            assert policy.previous==(view.fuel_distance,view.repair_distance)
            return m
    return Fixed()
