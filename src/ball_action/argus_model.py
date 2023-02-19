import argus

from src.models.action_timm import ActionTimm


class BallActionModel(argus.Model):
    nn_module = {
        "action_timm": ActionTimm,
    }
