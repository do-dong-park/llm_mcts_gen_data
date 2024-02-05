import argparse
import copy
import json
import os
import pdb
import pickle
import random
import sys

import ipdb
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(f"{curr_dir}")
sys.path.append(f"{curr_dir}/../")

from init_goal_setter.init_goal_base import SetInitialGoal
from init_goal_setter.tasks import Task
from utils import utils_goals

from vh.vh_sim.simulation.unity_simulator import comm_unity

""" Goal Domain을 만드는 Script

"""

parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-per-apartment", type=int, default=1, help="Maximum #episodes/apartment"
)
parser.add_argument("--seed", type=int, default=10, help="Seed for the apartments")

parser.add_argument("--task", type=str, default="setup_table", help="Task name")
parser.add_argument(
    "--apt_str",
    type=str,
    default="0,1,2,4,5",
    help="The apartments where we will generate the data",
)
parser.add_argument("--port", type=str, default="8093", help="Task name")
parser.add_argument("--display", type=int, default=1, help="Task name")
parser.add_argument(
    "--mode", type=str, default="full", choices=["simple", "full"], help="Task name"
)
parser.add_argument(
    "--usage",
    type=str,
    default="train",
    choices=["test", "train"],
    help="Train or Test",
)
parser.add_argument(
    "--use-editor", action="store_true", default=False, help="Use unity editor"
)
parser.add_argument(
    "--unseen-item", action="store_true", default=False, help="Use unity editor"
)
parser.add_argument(
    "--unseen-apartment", action="store_true", default=False, help="Use unity editor"
)
parser.add_argument(
    "--exec_file",
    type=str,
    default="./vh/vh_sim/macos_exec.2.2.4.app",
    help="Use unity editor",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.seed == 0:
        rand = random.Random()
    else:
        rand = random.Random(args.seed)

    if args.unseen_item:
        with open(f"{curr_dir}/data/init_pool_unseen.json") as file:
            init_pool = json.load(file)
    else:
        with open(f"{curr_dir}/data/init_pool.json") as file:
            init_pool = json.load(file)
    # comm = comm_unity.UnityCommunication()
    if args.use_editor:
        comm = comm_unity.UnityCommunication()
    else:
        print(comm_unity)
        comm = comm_unity.UnityCommunication(
            port=args.port,
            file_name=args.exec_file,
            no_graphics=True,
            logging=False,
            x_display=str(args.display),
        )
    comm.reset()

    ## -------------------------------------------------------------
    ## step3 load object size
    with open(f"{curr_dir}/data/class_name_size.json", "r") as file:
        class_name_size = json.load(file)

    ## -------------------------------------------------------------
    ## gen graph
    ## -------------------------------------------------------------

    all_tasks = [
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
        "put_fridge_put_bathroom_cabinet",
        "put_fridge_put_dishwasher",
        "prepare_food_prepare_wash",
        "setup_table_prepare_wash",
        "put_microwave_put_dishwasher",
    ]

    success_init_graph = []

    # apartment_ids = [int(apt_id) for apt_id in args.apt_str.split(',')]
    if args.unseen_apartment:
        apartment_ids = [1]
    else:
        apartment_ids = [3]
    if args.task == "all":
        if args.unseen_item:
            tasks = [
                "setup_table",
                "put_dishwasher",
                "put_microwave",
                "put_fridge",
                "prepare_food",
            ]
        else:
            tasks = [
                "setup_table",
                "put_dishwasher",
                "put_microwave",
                "put_bathroom_cabinet",
                "put_fridge",
                "prepare_snack",
                "prepare_wash",
                "prepare_food",
                # "put_kitchencabinet"
            ]
        if args.mode == "full":
            tasks += ["setup_table_prepare_food"]
    elif args.task == "unseen_comp":
        if args.unseen_item:
            tasks = ["put_fridge_put_dishwasher", "put_microwave_put_dishwasher"]
        else:
            tasks = [
                "put_fridge_put_bathroom_cabinet",
                "put_microwave_put_dishwasher",
                "put_fridge_put_dishwasher",
                "prepare_food_prepare_wash",
                "setup_table_prepare_wash",
            ]
    else:
        tasks = [args.task]
    num_per_apartment = args.num_per_apartment

    for task in tasks:
        # 정의한 테스크에 대하여 object의 위치 설정 -> env graph 생성
        for apartment in apartment_ids:
            print("apartment", apartment)

            if task not in all_tasks:
                continue

            with open(
                f"{curr_dir}/data/object_info%s.json" % (apartment + 1), "r"
            ) as file:
                obj_position = json.load(file)

            # filtering out certain locations
            for obj, pos_list in obj_position.items():
                if obj in ["book", "remotecontrol"]:
                    positions = [
                        pos
                        for pos in pos_list
                        if pos[0] == "INSIDE"
                        and pos[1] in ["kitchencabinet", "cabinet"]
                        or pos[0] == "ON"
                        and pos[1]
                        in (
                            ["cabinet", "bench", "nightstand"]
                            + ([] if apartment == 2 else ["kitchentable"])
                        )
                    ]
                elif obj == "remotecontrol":
                    # TODO: we never get here
                    positions = [
                        pos
                        for pos in pos_list
                        if pos[0] == "ON" and pos[1] in ["tvstand"]
                    ]
                else:
                    positions = [
                        pos
                        for pos in pos_list
                        if pos[0] == "INSIDE"
                        and pos[1]
                        in [
                            "fridge",
                            "kitchencabinet",
                            "cabinet",
                            "microwave",
                            "dishwasher",
                            "stove",
                            "bathroomcabinet",
                        ]
                        or pos[0] == "ON"
                        and pos[1]
                        in (
                            [
                                "cabinet",
                                "coffeetable",
                                "bench",
                                "kitchencounter",
                                "bathroomcounter",
                            ]
                            + ([] if apartment == 2 else ["kitchentable"])
                        )
                    ]
                obj_position[obj] = positions
            # print(obj_position['cutleryfork'])

            num_test = 100000
            count_success = 0
            for i in range(num_test):
                comm.reset(apartment)
                s, original_graph = comm.environment_graph()
                graph = copy.deepcopy(original_graph)
                kitchencabinet_id = []
                for node in original_graph["nodes"]:
                    if node["class_name"] == "kitchencabinet":
                        if len(kitchencabinet_id) == 0:
                            kitchencabinet_id.append(node["id"])
                        else:
                            graph["nodes"].remove(node)
                            for edge in graph["edges"]:
                                if (
                                    edge["from_id"] == node["id"]
                                    or edge["to_id"] == node["id"]
                                ):
                                    graph["edges"].remove(edge)
                task_name = task

                print(
                    "------------------------------------------------------------------------------"
                )
                print(
                    "testing %d/%d: %s. apartment %d"
                    % (i, num_test, task_name, apartment)
                )
                print(
                    "------------------------------------------------------------------------------"
                )

                ## -------------------------------------------------------------
                ## setup goal based on currect environment
                ## -------------------------------------------------------------
                set_init_goal = SetInitialGoal(
                    obj_position,
                    class_name_size,
                    init_pool,
                    task_name,
                    same_room=False,
                    rand=rand,
                    mode=args.mode,
                )
                init_graph, env_goal, success_setup = getattr(Task, task_name)(
                    set_init_goal, graph
                )

                if success_setup:
                    # If all objects were well added
                    success, message = comm.expand_scene(
                        init_graph, transfer_transform=False
                    )
                    print(
                        "----------------------------------------------------------------------"
                    )
                    print(task_name, success, message, set_init_goal.num_other_obj)
                    # print(env_goal)

                    if not success:
                        print(message)
                        goal_objs = []
                        goal_names = []
                        for k, goals in env_goal.items():
                            goal_objs += [
                                int(list(goal.keys())[0].split("_")[-1])
                                for goal in goals
                                if list(goal.keys())[0].split("_")[-1]
                                not in ["book", "remotecontrol"]
                            ]
                            goal_names += [
                                list(goal.keys())[0].split("_")[1] for goal in goals
                            ]
                        obj_names = [obj.split(".")[0] for obj in message["unplaced"]]
                        obj_ids = [
                            int(obj.split(".")[1]) for obj in message["unplaced"]
                        ]
                        id2node = {node["id"]: node for node in init_graph["nodes"]}

                        for obj_id in obj_ids:
                            print("Objects unplaced")
                            print(
                                [
                                    id2node[edge["to_id"]]["class_name"]
                                    for edge in init_graph["edges"]
                                    if edge["from_id"] == obj_id
                                ]
                            )
                            # ipdb.set_trace()
                        if task_name != "read_book" and task_name != "watch_tv":
                            intersection = set(obj_names) & set(goal_names)
                        else:
                            intersection = set(obj_ids) & set(goal_objs)

                        ## goal objects cannot be placed
                        if len(intersection) != 0:
                            success2 = False
                        else:
                            init_graph = set_init_goal.remove_obj(init_graph, obj_ids)
                            comm.reset(apartment)
                            success2, message2 = comm.expand_scene(
                                init_graph, transfer_transform=False
                            )
                            success = True

                    else:
                        success2 = True

                    if success2 and success:

                        success = set_init_goal.check_goal_achievable(
                            init_graph, comm, env_goal, apartment
                        )

                        if success:
                            init_graph0 = copy.deepcopy(init_graph)
                            comm.reset(apartment)
                            comm.expand_scene(init_graph, transfer_transform=False)
                            s, init_graph = comm.environment_graph()
                            print("final s:", s)
                            if s:
                                subgoals = []
                                if task_name in [
                                    "setup_table_prepare_food",
                                    "setup_table_put_microwave",
                                    "setup_table_put_fridge",
                                    "setup_table_put_dishwasher",
                                    "prepare_food_put_dishwasher",
                                    "put_fridge_put_bathroom_cabinet",
                                    "put_fridge_put_dishwasher",
                                    "put_dishwasher_prepare_snack",
                                    "prepare_wash_put_fridge",
                                    "put_fridge_put_bathroom_cabinet",
                                    "put_fridge_put_dishwasher",
                                    "prepare_food_prepare_wash",
                                    "setup_table_prepare_wash",
                                    "put_microwave_put_dishwasher",
                                ]:
                                    names = task_name.split("_")
                                    names_ = [
                                        names[0] + "_" + names[1],
                                        names[2] + "_" + names[3],
                                    ]
                                    for name in names_:
                                        if name in env_goal:
                                            subgoals += env_goal[name]
                                else:
                                    subgoals = env_goal[task_name]
                                for subgoal in subgoals:
                                    for k, v in subgoal.items():
                                        elements = k.split("_")
                                        # print(elements)
                                        # pdb.set_trace()
                                        if len(elements) == 4:
                                            obj_class_name = elements[1]
                                            ids = [
                                                node["id"]
                                                for node in init_graph["nodes"]
                                                if node["class_name"] == obj_class_name
                                            ]
                                            print(obj_class_name, v, ids)

                                count_success += s
                                check_result = set_init_goal.check_graph(
                                    init_graph, apartment + 1, original_graph
                                )
                                assert check_result == True

                                success_init_graph.append(
                                    {
                                        "id": count_success,
                                        "apartment": (apartment + 1),
                                        "task_name": task_name,
                                        "init_graph": init_graph,
                                        "original_graph": original_graph,
                                        "goal": env_goal,
                                    }
                                )
                # else:
                #     continue
                print(
                    "apartment: %d: success %d over %d (total: %d)"
                    % (apartment, count_success, i + 1, num_test)
                )
                if count_success >= num_per_apartment:
                    break

    data = success_init_graph
    env_task_set = []

    # for task in ['setup_table', 'put_fridge', 'put_dishwasher', 'prepare_food', 'read_book']:

    for task_id, problem_setup in enumerate(data):
        env_id = problem_setup["apartment"] - 1
        task_name = problem_setup["task_name"]
        init_graph = problem_setup["init_graph"]
        goal = []
        if task_name in [
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
            names = task_name.split("_")
            if len(names) == 4:
                names_ = [names[0] + "_" + names[1], names[2] + "_" + names[3]]
            else:
                names_ = [
                    names[0] + "_" + names[1],
                    names[2] + "_" + names[3] + "_" + names[4],
                ]
            for name in names_:
                if name in problem_setup["goal"]:
                    goal += problem_setup["goal"][name]
            goals = utils_goals.convert_goal_spec(
                names_,
                goal,
                init_graph,
            )
        else:
            goal = problem_setup["goal"][task_name]
            goals = utils_goals.convert_goal_spec(
                task_name,
                goal,
                init_graph,
            )
            # exclude=['cutleryknife'])
        print("env_id:", env_id)
        print("task_name:", task_name)
        print("goals:", goals)

        task_goal = {}
        for i in range(2):
            task_goal[i] = goals

        env_task_set.append(
            {
                "task_id": task_id,
                "task_name": task_name,
                "env_id": env_id,
                "init_graph": init_graph,
                "task_goal": task_goal,
                "level": 0,
                "init_rooms": rand.sample(
                    ["kitchen", "bedroom", "livingroom", "bathroom"], 2
                ),
            }
        )
    if args.usage == "train":
        pickle.dump(
            env_task_set,
            open(
                f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}.pik",
                "wb",
            ),
        )
    else:
        if args.unseen_item:
            if args.task == "unseen_comp":
                pickle.dump(
                    env_task_set,
                    open(
                        f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}_unseen_composition_unseen_item.pik",
                        "wb",
                    ),
                )
            else:
                pickle.dump(
                    env_task_set,
                    open(
                        f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}_unseen_item.pik",
                        "wb",
                    ),
                )
        elif args.unseen_apartment:
            pickle.dump(
                env_task_set,
                open(
                    f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}_unseen_apartment.pik",
                    "wb",
                ),
            )
        elif args.task == "unseen_comp":
            pickle.dump(
                env_task_set,
                open(
                    f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}_unseen_composition.pik",
                    "wb",
                ),
            )
        else:
            pickle.dump(
                env_task_set,
                open(
                    f"./vh/dataset/env_task_set_{args.num_per_apartment}_{args.mode}_seen.pik",
                    "wb",
                ),
            )
    # pickle.dump(env_task_set, open(f'{curr_dir}/dataset/env_task_set_{args.num_per_apartment}_{args.mode}.pik', 'wb'))
