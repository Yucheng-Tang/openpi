"""
A server for hosting a Pi0 and Pi0-Fast model for inference.

On action server: pip install uvicorn fastapi json-numpy
On client: pip install requests json-numpy

On client:

import requests
import json_numpy
from json_numpy import loads
json_numpy.patch()

Reset and provide the task before starting the rollout:

requests.post("http://serverip:port/reset", json={"text": ...})

Sample an action:

action = loads(
    requests.post(
        "http://serverip:port/query",
        json={"observation": ...},
    ).json()
)
"""

import logging
import os
import random
import json_numpy
import torch

from openpi.training import config
from openpi.policies import policy_config

json_numpy.patch()

import traceback
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import uvicorn


def json_response(obj):
    return JSONResponse(json_numpy.dumps(obj))


class PhysicalIntelligenceServer:
    def __init__(self, cfg, checkpoint_dir):

        self.model = policy_config.create_trained_policy(cfg, checkpoint_dir, local_model_only=True)

        self.text = None
        self.action = None
        self.counter = 0

    def run(self, port=8000, host="10.10.10.130"):
        self.app = FastAPI()
        self.app.post("/query")(self.sample_actions)
        self.app.post("/reset")(self.reset)
        uvicorn.run(self.app, host=host, port=port)

    def reset(self, payload: Dict[Any, Any]):
        self.text = payload["text"]
        self.model.reset()

        return "reset"

    def sample_actions(self, payload: Dict[Any, Any]):
        # payload needs to contain primary_image, secondary_image
        try:
            ts_payload = self.ensure_tensor_payload(payload)
            ts_input = {
                "observation/image_0": ts_payload["primary_image"],
                "observation/image_1": ts_payload["secondary_image"],
                "observation/state": ts_payload["joint_pos"],
                "prompt": self.text
            }
            if self.counter % 20 == 0:
                self.action = self.model.infer(ts_input)
                self.action = self.action["actions"]
                self.counter = 0


            action = self.action[self.counter, :]
            action[-1] = 1 if action[-1] > 0.5 else -1

            self.counter += 1

            return json_response(action)
        except:
            print(traceback.format_exc())
            return "error"

    def ensure_tensor_payload(self, payload: dict) -> dict:
        tensor_payload = {}
        for key, value in payload.items():
            if isinstance(value, torch.Tensor):
                tensor_payload[key] = value
            elif isinstance(value, (int, float, list, np.ndarray)):
                tensor_payload[key] = torch.tensor(value)
            else:
                # Keep str unchanged
                tensor_payload[key] = value
        return tensor_payload


# @hydra.main(config_path="/home/irl-admin/Omer/flower_baselines/vla_mp/conf/eval", config_name="kitchen_server")
def main():
    # base dataset
    model_config = config.get_config("pi0_real_franka_kitchen_low_mem_finetune")

    checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
                      "pi0_real_franka_kitchen_basemodel_24_04/45000")

    # # kitchen seq dataset
    # model_config = config.get_config("pi0_real_franka_kitchen_low_mem_finetune")
    #
    # checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
    #                   "pi0_real_franka_kitchen_basemodel_27_04_4090/20000")

    # # pi0 fast
    # model_config = config.get_config("pi0_fast_real_franka_kitchen_low_mem_finetune")
    #
    # # checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
    # #                   "pi0_fast_real_franka_kitchen_basemodel_24_04/50000")
    # checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
    #                   "pi0_fast_real_franka_kitchen_basemodel_28_04_35/25000")


    server = PhysicalIntelligenceServer(model_config, checkpoint_dir)
    server.run(host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()