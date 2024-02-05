import json
import os
import pickle
import random
import sys
from pathlib import Path

import ipdb
import numpy as np

from vh.data_gene.agents import MCTS_agent
from vh.data_gene.algos.arena_mp2 import ArenaMP
from vh.data_gene.arguments import get_args
from vh.data_gene.envs.unity_environment import UnityEnvironment
from vh.data_gene.utils import utils_goals

if __name__ == "__main__":
    args = get_args()

    num_tries = 1
    args.max_episode_length = 50

    env_task_set = pickle.load(open(args.dataset_path, "rb"))
    print(f"episode tast set len : {len(env_task_set)}")
    args.record_dir = f"./expert_actions/expert_{args.mode}"
    if not os.path.exists(args.record_dir):
        os.makedirs(args.record_dir)

    executable_args = {
        "file_name": args.executable_file,
        "x_display": "1",
        "no_graphics": True,
    }

    id_run = 0
    random.seed(id_run)
    episode_ids = list(range(len(env_task_set)))

    episode_ids = sorted(episode_ids)

    S = [[] for _ in range(len(episode_ids))]
    L = [[] for _ in range(len(episode_ids))]

    test_results = {}

    def env_fn(env_id):
        return UnityEnvironment(
            num_agents=1,
            max_episode_length=args.max_episode_length,
            port_id=env_id,
            env_task_set=env_task_set,
            observation_types=[args.obs_type],
            use_editor=args.use_editor,
            executable_args=executable_args,
            base_port=args.base_port,
        )

    args_common = dict(
        recursive=False,
        max_episode_length=5,
        num_simulation=1000,
        max_rollout_steps=50,
        c_init=0.1,
        c_base=1000000,
        num_samples=1,
        num_processes=1,
        logging=True,
        logging_graphs=True,
    )

    args_agent1 = {"agent_id": 1, "char_index": 0}
    args_agent1.update(args_common)
    agents = [lambda x, y: MCTS_agent(**args_agent1)]
    arena = ArenaMP(args.max_episode_length, id_run, env_fn, agents)

    for iter_id in range(num_tries):

        cnt = 0
        steps_list, failed_tasks = [], []
        current_tried = iter_id

        if not os.path.isfile(args.record_dir + "/results_{}.pik".format(0)):
            test_results = {}
        else:
            test_results = pickle.load(
                open(args.record_dir + "/results_{}.pik".format(0), "rb")
            )

        for episode_id in episode_ids:

            curr_log_file_name = args.record_dir + "/expert_traj_{}_{}_{}.pik".format(
                env_task_set[episode_id]["task_id"],
                env_task_set[episode_id]["task_name"],
                iter_id,
            )
            if env_task_set[episode_id]["task_name"] not in [
                "setup_table",
                "put_dishwasher",
                "put_microwave",
                "put_bathroom_cabinet",
                "put_fridge",
                "put_kitchencabinet",
                "prepare_drinks",
                "prepare_snack",
                "prepare_wash",
                "prepare_food",
                "setup_table_prepare_food",
                "setup_table_put_microwave",
                "setup_table_put_fridge",
                "setup_table_put_dishwasher",
                "prepare_food_put_dishwasher",
                "put_fridge_put_bathroom_cabinet",
                "put_fridge_put_dishwasher",
                "put_dishwasher_prepare_snack",
                "prepare_wash_put_fridge",
                "put_fridge_put_dishwasher",
                "prepare_food_prepare_wash",
                "setup_table_prepare_wash",
                "put_microwave_put_dishwasher",
            ]:
                continue

            print(env_task_set[episode_id]["task_name"])
            print("episode:", episode_id)

            for it_agent, agent in enumerate(arena.agents):
                agent.seed = it_agent + current_tried * 2

            try:
                arena.reset(episode_id)
                success, steps, saved_info = arena.run()
                print("-------------------------------------")
                print("success" if success else "failure")
                print("steps:", steps)
                print("-------------------------------------")

                if not success:
                    failed_tasks.append(episode_id)
                else:
                    steps_list.append(steps)
                is_finished = 1 if success else 0

                Path(args.record_dir).mkdir(parents=True, exist_ok=True)

                if success:
                    log_file_name = (
                        args.record_dir
                        + "/expert_traj_{}_{}_{}.pik".format(
                            saved_info["task_id"],
                            saved_info["task_name"],
                            current_tried,
                        )
                    )
                    if len(saved_info["obs"]) > 0:
                        pickle.dump(saved_info, open(log_file_name, "wb"))
                    else:
                        with open(log_file_name, "w+") as f:
                            f.write(json.dumps(saved_info, indent=4))
            except:
                # ipdb.set_trace()
                arena.reset_env()

            S[episode_id].append(is_finished)
            L[episode_id].append(steps)
            test_results[episode_id] = {"S": S[episode_id], "L": L[episode_id]}

        pickle.dump(
            test_results, open(args.record_dir + "/results_{}.pik".format(0), "wb")
        )
        print(
            "average steps (finishing the tasks):",
            np.array(steps_list).mean() if len(steps_list) > 0 else None,
        )
        print("failed_tasks:", failed_tasks)
        pickle.dump(
            test_results, open(args.record_dir + "/results_{}.pik".format(0), "wb")
        )
