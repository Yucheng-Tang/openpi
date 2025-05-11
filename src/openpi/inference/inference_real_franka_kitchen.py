from openpi.training import config, data_loader
from openpi.policies import policy_config
from openpi.shared import download
import openpi.transforms as transforms
import numpy as np

import matplotlib.pyplot as plt
import jax.numpy as jnp


model_config = config.get_config("pi0_fast_real_franka_kitchen_low_mem_finetune")

# checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")
# checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
#                   "pi0_real_franka_mix_basemodel_26_04/40000")

checkpoint_dir = ("/media/developer/Samsung SSD/irl_ws/checkpoints/"
                      "pi0_fast_real_franka_kitchen_basemodel_28_04_35/25000")

# Create a trained policy.
policy = policy_config.create_trained_policy(model_config, checkpoint_dir, local_model_only=True)

# Run inference on a dummy example.
# example = {
#     "observation/exterior_image_1_left": ...,
#     "observation/wrist_image_left": ...,
#     ...
#     "prompt": "pick up the fork"
# }

data_config = model_config.data.create(model_config.assets_dirs, model_config.model)

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}

if data_config.repo_id is None:
    raise ValueError("Data config must have a repo_id")
dataset = data_loader.create_dataset(data_config, model_config.model)
# dataset = data_loader.TransformedDataset(
#     dataset,
#     [
#         *data_config.repack_transforms.inputs,
#         *data_config.data_transforms.inputs,
#         # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
#         RemoveStrings(),
#     ],
# )

# data_config = config.LeRobotFrankaDataConfig(
#     repo_id="tyc1333/real_franka_kitchen",  # or whatever dataset repo
#     base_config=config.DataConfig(
#         local_files_only=False,
#         prompt_from_task=True,
#     )
# )

# dataset = data_loader.create_dataset(data_config, model_config.model)[1]

# example = next(iter(dataset))

for example in dataset:
    example_input = {
        "observation/image_0": example["image_0"],
        "observation/image_1": example["image_1"],
        "observation/state": example["state"],
        "prompt": example["prompt"]
    }
    example_output = example["actions"]

    # 4. Inference
    out = policy.infer(example_input)

    pred_actions = out["actions"]
    gt_actions = example_output

    # Convert to numpy if needed
    if isinstance(pred_actions, jnp.ndarray):
        pred_actions = jnp.array(pred_actions)
    if isinstance(gt_actions, jnp.ndarray):
        gt_actions = jnp.array(gt_actions)

    pred_actions = pred_actions.squeeze()
    gt_actions = gt_actions.squeeze()

    num_dims = pred_actions.shape[-1]  # Should be 8

    fig, axs = plt.subplots(2, 4, figsize=(20, 8))  # 2 rows, 4 columns
    axs = axs.flatten()

    for i in range(num_dims):
        axs[i].plot(pred_actions[..., i], label="Predicted", marker='o')
        axs[i].plot(gt_actions[..., i], label="Ground Truth", marker='x')
        axs[i].set_title(f"Action Dimension {i}")
        axs[i].set_xlabel("Time step (or action index)")
        axs[i].set_ylabel("Action value")
        axs[i].grid(True)
        axs[i].legend()

    plt.tight_layout()
    plt.show()

    # print(out["actions"])
    # print(example["actions"])

# action_chunk = policy.infer(example)["actions"]