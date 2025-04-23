import shutil

import numpy as np
import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "tyc1333/real_franka_kitchen"
RAW_DATASET_NAME = ("/home/developer/irl_ws/vla_2025/datasets/kit_irl_real_kitchen_non_play_apr25/"
                    "kit_irl_real_kitchen_non_play_apr25/1.0.0")


def main(data_dir: str, *,
         push_to_hub: bool = True,
         split: str = "20%",
         num_episodes=20,
         action_key="action",
         image_key="observation"):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Load the dataset
    # dataset = tfds.load("libero_10_no_noops", data_dir=data_dir, split="train")
    raw_dataset = tfds.builder_from_directory(data_dir).as_dataset(split=f'train[:{split}]')
    raw_dataset = raw_dataset.prefetch(1)
    episodes = [e for e in raw_dataset.take(num_episodes)]

    goal_dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image_0": {
                "dtype": "image",
                "shape": (250, 250, 3),
                "names": ["height", "width", "channel"],
            },
            "image_1": {
                "dtype": "image",
                "shape": (250, 250, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # # Iterate through the dataset
    # for episode in dataset:
    #     # Extract the image and state
    #     image = episode["image"].numpy()
    #     state = episode["state"].numpy()
    #
    #     # Display the image and state
    #     plt.imshow(image)
    #     plt.title(f"State: {state}")
    #     plt.axis('off')
    #     plt.show()

    for episode in raw_dataset:
        for step in episode["steps"].as_numpy_iterator():
            # print("timestamp:", step["timestamp"])
            goal_dataset.add_frame(
                {
                    "image_0": step["observation"]["image_top"],
                    "image_1": step["observation"]["image_side"],
                    "state": step["observation"]["joint_state"],
                    "actions": step["action"],
                }
            )
        goal_dataset.save_episode(task=step["language_instruction"].decode())

    # Consolidate the dataset, skip computing stats since we will do that later
    goal_dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        goal_dataset.push_to_hub(
            tags=["real_franka_kitchen", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    # tyro.cli(main)
    main(RAW_DATASET_NAME)
