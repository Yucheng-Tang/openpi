import shutil

import numpy as np
import tensorflow_datasets as tfds
# import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

REPO_NAME = "tyc1333/real_franka_mix"
RAW_DATASET_NAME = "/home/developer/irl_ws/vla_2025/datasets/real_franka_mixer/1.0.0"

RAW_DATASET_NAME_MIX = "/media/developer/Samsung SSD/irl_ws/datasets/real_franka_experiments"
FOLDER_TO_LOAD = ["real_franka_fold_towel", "real_franka_pour", "real_franka_sweeps",
                  "real_franka_pick_place", "real_franka_mixer"]


def load_multiple_datasets(base_dir: str, folder_list: list[str], split="100%", num_episodes=None):
    all_episodes = []

    for folder_name in folder_list:
        data_dir = f"{base_dir}/{folder_name}/1.0.0"
        builder = tfds.builder_from_directory(data_dir)
        raw_dataset = builder.as_dataset(split=f"train[:{split}]")
        raw_dataset = raw_dataset.prefetch(1)

        if num_episodes is not None:
            episodes = [e for e in raw_dataset.take(num_episodes)]
        else:
            episodes = list(raw_dataset)

        all_episodes.extend(episodes)

    return all_episodes

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

    # Load single dataset
    # raw_dataset = tfds.builder_from_directory(data_dir).as_dataset(split=f'train[:{split}]')
    # raw_dataset = raw_dataset.prefetch(1)
    # episodes = [e for e in raw_dataset.take(num_episodes)]

    # Load multiple datasets
    episodes = load_multiple_datasets(data_dir, FOLDER_TO_LOAD, split=split, num_episodes=num_episodes)

    goal_dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=10,
        features={
            "image_0": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "image_1": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
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

    for episode in episodes:
        for step in episode["steps"].as_numpy_iterator():
            # print("timestamp:", step["timestamp"])
            goal_dataset.add_frame(
                {
                    "image_0": step["observation"]["orb_0"],
                    "image_1": step["observation"]["orb_1"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                }
            )
        goal_dataset.save_episode(task=step["language_instruction"].decode())

    # Consolidate the dataset, skip computing stats since we will do that later
    goal_dataset.consolidate(run_compute_stats=False)

    if push_to_hub:
        goal_dataset.push_to_hub(
            tags=["real_franka_mix", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
            tag="v1.0.0"
        )


if __name__ == "__main__":
    # tyro.cli(main)
    main(RAW_DATASET_NAME_MIX)
