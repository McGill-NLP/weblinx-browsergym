import ast
import json
import logging
import os
import time
import zipfile
from copy import deepcopy
from logging import warning
from pathlib import Path
from typing import Any, Literal, Optional

import gymnasium as gym
import numpy as np
import weblinx as wl
import weblinx.eval.metrics
import weblinx.processing as wlp
from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.core.chat import Chat
from browsergym.core.env import BrowserEnv
from browsergym.core.task import AbstractBrowserTask
from huggingface_hub import snapshot_download, hf_hub_download
from PIL import Image
from weblinx.processing.intent import Intent

from .version import __version__

version = __version__

# create a logger
logger = logging.getLogger(__name__)


class WindowlessChat(Chat):
    def __init__(self, *args, **kwargs):
        self.messages = []

    def add_message(
        self,
        role: Literal["user", "user_image", "assistant", "info", "infeasible"],
        msg: str,
    ):
        """Add a message to the chatbox and update the page accordingly."""
        utc_time = time.time()
        if role not in ("user", "user_image", "assistant", "info", "infeasible"):
            raise ValueError(f"Invalid role: {role}")
        if role in ("user", "user_image", "assistant", "infeasible"):
            self.messages.append({"role": role, "timestamp": utc_time, "message": msg})

    def wait_for_user_message(self):
        return None

    def close(self):
        del self.messages


class ScrollSimilarityMetric(weblinx.eval.metrics.Metric):
    def __init__(self, x_threshold=100, y_threshold=100):
        super().__init__(name="ScrollSimilarity", args={})
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold

        if x_threshold < 0 or y_threshold < 0:
            raise ValueError("Thresholds should be non-negative.")

    def is_applicable(self, pred, ref):
        return pred["intent"] == ref["intent"] and pred["intent"] == Intent.SCROLL

    def score(self, pred, ref, **kwargs) -> float:
        if pred["intent"] != ref["intent"]:
            return 0

        pred_x, pred_y = pred["args"]["x"], pred["args"]["y"]
        ref_x, ref_y = ref["args"]["x"], ref["args"]["y"]

        x_dist = abs(pred_x - ref_x)
        y_dist = abs(pred_y - ref_y)
        # scale so it's a value between 0 and 1 based on threshold
        x_score = min(max(0, 1 - (x_dist / self.x_threshold)), 1)
        y_score = min(max(0, 1 - (y_dist / self.y_threshold)), 1)

        # we take the product of the two scores so that the score is 1 if both x and y are within the threshold
        # and 0 if either x or y is outside the threshold, and a blend if they are close but not exact matches
        return x_score * y_score


class ExactMatchTabSwitchMetric(weblinx.eval.metrics.Metric):
    def __init__(self):
        super().__init__(name="ExactMatchTabSwitch", args={})

    def is_applicable(self, pred, ref):
        return pred["intent"] == ref["intent"] and pred["intent"] == Intent.TAB_SWITCH

    def score(self, pred, ref, **kwargs) -> float:
        if pred["intent"] != ref["intent"]:
            return 0

        pred_origin = pred["args"].get("origin", -1)
        pred_target = pred["args"].get("target", -1)

        ref_target = ref["args"]["target"]

        target_score = 1 if pred_target == ref_target else 0

        return target_score


def apply_zoom_to_extra_properties(extra_properties, zoom, inplace=True):
    if not inplace:
        from copy import deepcopy

        extra_properties = deepcopy(extra_properties)

    for key, value in extra_properties.items():
        if value["bbox"] is None:
            continue
        extra_properties[key]["bbox"] = [int(coord * zoom) for coord in value["bbox"]]
    return extra_properties


def parse_function_string(func_str):
    # Parse the function string to an AST (Abstract Syntax Tree)
    tree = ast.parse(func_str)

    # The parsed body should contain an expression, and it must be a Call node
    if isinstance(tree.body[0].value, ast.Call):
        func_call = tree.body[0].value

        # Extract function name
        func_name = func_call.func.id

        # Extract positional arguments (args)
        args = [ast.literal_eval(arg) for arg in func_call.args]

        # Extract keyword arguments (kwargs)
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in func_call.keywords}

        return func_name, args, kwargs
    else:
        return "<unk>", [], {}


def map_args_to_kwargs(func_name, args, kwargs, arg_order):
    """
    The action string in BrowserGym is a string that looks like a function call in Python, e.g.
    click('251d0ba4-b4d7-4d52')
    send_msg_to_user("Hi! How can I assist you today?")

    We want to convert this string into the keyword arguments format, i.e.
    click(uid='251d0ba4-b4d7-4d52')
    send_msg_to_user(text="Hi! How can I assist you today?")
    """

    if func_name not in arg_order:
        return kwargs

    new_kwargs = {}
    for i, arg in enumerate(args):
        new_kwargs[arg_order[func_name][i]] = arg

    # merge the new kwargs with the existing kwargs
    new_kwargs.update(kwargs)

    return new_kwargs


def convert_weblinx_to_browsergym_action(
    intent: str, args: dict, warn_on_unknown=False
):
    """
    This converts an action using the BrowserGym structure to the WebLINX format,
    which is different. The mapping of weblinx to browsergym is as follows:

    say(speaker="assistant", utterance=[str]) -> send_msg_to_user(text=[str])
    click(uid=[element id]) -> click(bid=[element id])
    hover(uid=[element id]) -> hover(bid=[element id])
    textinput(uid=[element id], value=[str]) -> fill(bid=[element id], value=[str])
    change(uid=[element], value=[str]) -> ❌
    load(url=[link]) -> goto(url=[link])
    submit(uid=[element]) -> click(bid=[element id])
    scroll(x=[int x],y=[int y]) -> scroll(delta_x=[int x], delta_y=[int y])
    copy(uid=[element],text=[str]) -> ❌
    paste(uid=[element],text=[str]) -> ❌
    tabcreate() -> new_tab()
    tabremove(target=[tabId]) -> tab_close()
    tabswitch(origin=[origin tabId],target=[target tabId]) -> tab_focus(index=[target tabid])
    """
    if intent in ["copy", "paste", "change"]:
        if warn_on_unknown:
            warning(
                f"Intent {intent} is not supported by BrowserGym. Skipping this action."
            )
        return "fail", {}

    if intent == "click":
        new_intent = "click"
        new_args = {"bid": args["uid"]}

    elif intent == "hover":
        new_intent = "hover"
        new_args = {"bid": args["uid"]}

    elif intent == "textinput":
        new_intent = "fill"
        new_args = {"bid": args["uid"], "value": args["value"]}

    elif intent == "submit":
        new_intent = "click"
        new_args = {"bid": args["uid"]}

    elif intent == "load":
        new_intent = "goto"
        new_args = {"url": args["url"]}

    elif intent == "scroll":
        new_intent = "scroll"
        new_args = {"delta_x": args["x"], "delta_y": args["y"]}

    elif intent == "tabcreate":
        new_intent = "new_tab"
        new_args = {}

    elif intent == "tabremove":
        new_intent = "tab_close"
        new_args = {}

    elif intent == "tabswitch":
        new_intent = "tab_focus"
        new_args = {"index": args["target"]}

    elif intent == "say":
        # we don't need to convert this, since later we will convert it to the chat message
        new_intent, new_args = intent, args

    else:
        new_intent = "noop"
        new_args = {}

    return new_intent, new_args


def convert_browsergym_to_weblinx_action(
    intent: str, args: dict, warn_on_unknown=False
):
    """
    This converts an action using the BrowserGym structure to the WebLINX format,
    which is different. The mapping of weblinx to browsergym is as follows:

    send_msg_to_user(text=[str]) -> say(speaker="assistant", utterance=[str])
    click(bid=[element id]) -> click(uid=[element id])
    hover(bid=[element id]) -> hover(uid=[element id])
    fill(bid=[element id], value=[str]) -> textinput(uid=[element id], value=[str])
    goto(url=[link]) -> load(url=[link])
    scroll(delta_x=[int x], delta_y=[int y]) -> scroll(x=[int x],y=[int y])
    new_tab() -> tabcreate()
    tab_close() -> tabremove(target=[tabId])
    tab_focus(index=[target tabid]) -> tabswitch(origin=[origin tabId],target=[target tabId])
    """
    if intent in ["copy", "paste", "change"]:
        if warn_on_unknown:
            warning(
                f"Intent {intent} is not supported by WebLINX. Skipping this action."
            )
        return "fail", {}

    if intent == "click":
        new_intent = "click"
        new_args = {"uid": args["bid"]}

    elif intent == "send_msg_to_user":
        new_intent = "say"
        new_args = {"speaker": "assistant", "utterance": args["text"]}

    elif intent == "hover":
        new_intent = "hover"
        new_args = {"uid": args["bid"]}

    elif intent == "fill":
        new_intent = "textinput"
        new_args = {"uid": args["bid"], "value": args["value"]}

    elif intent == "goto":
        new_intent = "load"
        new_args = {"url": args["url"]}

    elif intent == "scroll":
        new_intent = "scroll"
        new_args = {"x": args["delta_x"], "y": args["delta_y"]}

    elif intent == "new_tab":
        new_intent = "tabcreate"
        new_args = {}

    elif intent == "tab_close":
        new_intent = "tabremove"
        new_args = {}

    elif intent == "tab_focus":
        new_intent = "tabswitch"
        new_args = {"target": args["index"]}

    else:
        new_intent = Intent.UNKNOWN
        new_args = {}

    return new_intent, new_args


def download_metadata(
    cache_dir="./bg_wl_data",
    repo_id="McGill-NLP/weblinx-browsergym",
    skip_if_exists=True,
) -> str:
    cache_dir = Path(cache_dir).expanduser()
    metadata_path = cache_dir / "metadata.json"

    if skip_if_exists and metadata_path.exists():
        logger.debug(
            f"Metadata file already exists at {metadata_path}. Skipping download."
        )
        return str(metadata_path)

    hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="metadata.json",
        local_dir=str(cache_dir),
    )

    return str(metadata_path)


def list_tasks(split="test_iid", cache_dir="./bg_wl_data", metadata_path=None):
    if metadata_path is None:
        metadata_path = download_metadata(cache_dir=cache_dir)
    else:
        metadata_path = Path(metadata_path)

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    tasks = []

    for demo_id in metadata[split]:
        for step in metadata[split][demo_id]:
            step_dict = metadata[split][demo_id][step]
            if step_dict["is_task"] and step_dict["has_full_snapshot"]:
                tasks.append(f"weblinx.{demo_id}.{step}")

    logger.debug(f"Found {len(tasks)} tasks in the WebLINX environment for split {split}.")
    return tasks


def extract_demo_id(task_name: str):
    if task_name.startswith("browsergym/"):
        task_name = task_name.split("/", 1)[1]

    if task_name.startswith("weblinx."):
        task_name = task_name.split(".", 1)[1]

    if "." in task_name:
        demo_id, step = task_name.split(".", 1)

    return demo_id


def get_unique_demo_ids(tasks):
    demo_ids = set()
    for task in tasks:
        demo_ids.add(extract_demo_id(task))
    return list(demo_ids)


def create_goal_message(
    demo_id: str, meta_split_dict: dict, chat_messages, option="action_history"
):
    if option not in [
        "first_instructor",
        "last_instructor",
        "flattened_both",
        "action_history",
    ]:
        raise ValueError(
            "Invalid option. Please use one of 'first_instructor', 'last_instructor', 'action_history', or 'flattened_both'."
        )

    speaker_map = {"instructor": "User", "navigator": "Assistant"}

    demo_dict = meta_split_dict[demo_id]

    if option == "first_instructor":
        for step, step_dict in demo_dict.items():
            if (
                step_dict["intent"] == "say"
                and step_dict["args"]["speaker"] == "instructor"
            ):
                return step_dict["args"]["utterance"]

    if option == "last_instructor":
        for step, step_dict in reversed(list(demo_dict.items())):
            if (
                step_dict["intent"] == "say"
                and step_dict["args"]["speaker"] == "instructor"
            ):
                return step_dict["args"]["utterance"]

    if option == "flattened_both":
        messages = []
        for step, step_dict in demo_dict.items():
            if step_dict["intent"] == "say":
                speaker = step_dict["args"]["speaker"]
                speaker = speaker_map.get(speaker, speaker)

                messages.append(f'{speaker}: {step_dict["args"]["utterance"]}')

        return "\n".join(messages)

    if option == "action_history":
        messages = []
        for chat in chat_messages:
            # show {role}: {message}
            speaker = chat["role"]
            msg = chat["message"]

            messages.append(f"{speaker}: {msg}")

        return "\n".join(messages)
    return ""


def get_open_tabs(demo_id: str, step_num: int, meta_split_dict: dict):
    step_num = str(step_num)
    demo_dict = meta_split_dict[demo_id]
    tab_id_to_url = {}

    for curr_step, step_dict in demo_dict.items():
        if curr_step == step_num:
            break
        if "tab" not in step_dict:
            continue

        if step_dict["intent"] == "tabremove":
            tab_id_to_url.pop(step_dict["args"]["target"], None)
        else:
            tab = step_dict["tab"]
            if tab["url"] is not None and tab["url"] != "":
                tab_id_to_url[tab["id"]] = tab["url"]

    return {
        "ids": list(tab_id_to_url.keys()),
        "urls": list(set(tab_id_to_url.values())),
        "titles": list(["" for _ in tab_id_to_url.keys()]),
    }


def get_active_index(current_tab_id: str, open_tabs: dict):
    if current_tab_id not in open_tabs["ids"]:
        return 0
    return open_tabs["ids"].index(current_tab_id)


def format_action(intent: str, args: dict):
    args_str = []
    for k, v in args.items():
        if isinstance(v, str):
            v = f'"{v}"'
        else:
            v = str(v)
        args_str.append(f"{k}={v}")
    return f"{intent}({', '.join(args_str)})"


def convert_weblinx_action_history_to_chat_messages(
    demo_id: str, step_num: int, meta_split_dict: dict, convert_to_browsergym=True
):
    step_num = str(step_num)
    demo_dict = meta_split_dict[demo_id]
    chat_messages = []

    for curr_step, step_dict in demo_dict.items():
        if curr_step == step_num:
            break

        if step_dict["intent"] == "say":
            if step_dict["args"]["speaker"] == "instructor":
                role = "user"
            else:
                role = "assistant"

            msg = {
                "role": role,
                "message": step_dict["args"]["utterance"],
                "timestamp": step_dict["timestamp"],
            }
        else:
            # display the action verbatim
            intent, args = step_dict["intent"], step_dict["args"]
            if convert_to_browsergym:
                intent, args = convert_weblinx_to_browsergym_action(
                    intent, args, warn_on_unknown=True
                )

            action = format_action(intent, args)

            msg = {
                "role": "assistant",
                "message": action,
                "timestamp": step_dict["timestamp"],
            }

        chat_messages.append(msg)

    return chat_messages


def clip_to_1(x):
    return max(0, min(1, x))


def extract_function_call_from_action_string(raw_str: str, only_first=True) -> str:
    """
    This function takes a string with all type of text, and returns the first function call in the string,
    if only_first is True. If only_first is False, it returns all function calls in the string, separated by
    a newline character.
    """
    from browsergym.core.action.parsers import highlevel_action_parser

    calls = highlevel_action_parser.search_string(raw_str)
    calls = sum(calls.as_list(), [])
    if only_first:
        calls = calls[:1]

    python_code = ""
    for function_name, function_args in calls:
        python_code += (
            function_name
            + "("
            + ", ".join([repr(arg) for arg in function_args])
            + ")\n"
        )

    return python_code


def get_reward(pred_action: dict, ref_action: dict, turn: wl.Turn = None):
    """
    This function evaluates the predicted action against the reference action,
    leveraging the bounding boxes of the elements on the page for the evaluation.

    Parameters
    ----------
    pred_action : dict
        The predicted action, with the following keys:
        - intent: str
        - args: dict
        - element: dict

    ref_action : dict
        The reference action, with the following keys:
        - intent: str
        - args: dict
        - element: dict

    turn : wl.Turn
        The turn object containing the information about the current state of the page.

    Returns
    -------
    float
        The reward for the action.
    """
    accepted_bg_intents = {
        "click",  # == click
        "hover",  # == hover
        "fill",  # == textinput
        "goto",  # == load
        "scroll",  # == scroll
        "new_tab",  # == tabcreate
        "tab_close",  # == tabremove
        "tab_focus",  # == tabswitch
        "send_msg_to_user",  # == say
    }

    accepted_wl_intents = {
        "click",  # iou
        "hover",  # iou
        "textinput",  # iou, chrF
        "load",  # urlF
        "submit",  # iou, not present in browsergym
        "scroll",  # distance (new)
        "tabcreate",  # exact match
        "tabremove",  # exact match
        "tabswitch",  # exact match
        "say",  # chrF
    }

    if pred_action["intent"] == Intent.UNKNOWN:
        return 0

    if (
        pred_action["intent"] not in accepted_bg_intents
        and pred_action["intent"] not in accepted_wl_intents
    ):
        warning(
            f"Predicted intent {pred_action['intent']} is not supported by WebLINX. Assigning 0."
        )
        return 0

    # if pred_action["intent"] in accepted_bg_intents:
    #     pred_action["intent"], pred_action["args"] = (
    #         convert_browsergym_to_weblinx_action(
    #             pred_action["intent"], pred_action["args"], warn_on_unknown=False
    #         )
    #     )
    if pred_action["intent"] not in accepted_wl_intents:
        warning(
            f"Predicted intent {pred_action['intent']} is not supported by WebLINX. Assigning 0."
        )
        return 0

    # we remap submit to click since it does not exist in browsergym
    if pred_action["intent"] in ["change", "submit"]:
        pred_action["intent"] = "click"
    if ref_action["intent"] in ["change", "submit"]:
        ref_action["intent"] = "click"

    # now, we are sure that the pred_action is in the WebLINX format, and we can use hte weblinx library to evaluate it
    if pred_action["intent"] in ["click", "hover"]:
        # use IoU
        metric = weblinx.eval.metrics.IOUMetric()

    elif pred_action["intent"] == "load":
        metric = weblinx.eval.metrics.URLFMetric()

    elif pred_action["intent"] == "scroll":
        metric = ScrollSimilarityMetric()

    elif pred_action["intent"] == "say":
        metric = weblinx.eval.metrics.ChrFMetric()

    elif pred_action["intent"] in ["tabcreate", "tabremove"]:
        metric = weblinx.eval.metrics.IntentMatchMetric()

    elif pred_action["intent"] == "tabswitch":
        metric = ExactMatchTabSwitchMetric()

    elif pred_action["intent"] == "textinput":
        metric = weblinx.eval.metrics.ChrFMetric()
        metric2 = weblinx.eval.metrics.IOUMetric()

        score1 = metric.score(pred=pred_action, ref=ref_action)
        score2 = metric2.score(pred=pred_action, ref=ref_action)

        # this is a weighted average of the two scores
        return clip_to_1(score1 * score2)

    else:
        # if the intent is not supported, assign 0
        return 0

    return clip_to_1(metric.score(pred=pred_action, ref=ref_action))


def download_and_unzip_demos(
    demo_ids, cache_dir="./bg_wl_data", repo_id="McGill-NLP/weblinx-browsergym"
):
    base_demo_dir = Path(cache_dir) / "demonstrations"
    base_zip_dir = Path(cache_dir) / "demonstrations_zip"

    download_patterns = []
    not_to_download = []

    for demo_id in demo_ids:
        if not base_demo_dir.joinpath(demo_id).exists():
            pattern = f"demonstrations_zip/{demo_id}.zip"
            download_patterns.append(pattern)
        else:
            not_to_download.append(demo_id)

    num_to_download = len(download_patterns)
    num_not_to_download = len(not_to_download)
    logger.debug(
        f"Downloading {num_to_download} zips to {base_zip_dir}, skipping {num_not_to_download} demos."
    )

    for pattern in download_patterns:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=cache_dir,
            filename=pattern,
        )

    logger.debug(f"Downloaded {num_to_download} zips to {base_zip_dir}")

    for demo_id in demo_ids:
        # if the demo directory does not exist...
        demo_dir = base_demo_dir.joinpath(demo_id)
        if not demo_dir.exists():
            # ... then we unzip the zip file to the demo directory
            zip_path = base_zip_dir.joinpath(f"{demo_id}.zip")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(demo_dir)

            logger.debug(f"Unzipped {demo_id}.zip to {demo_dir}")

    return str(base_demo_dir)

def download_and_unzip_demo(demo_id, cache_dir="./bg_wl_data", repo_id="McGill-NLP/weblinx-browsergym"):
    base_demo_dir = Path(cache_dir) / "demonstrations"
    base_zip_dir = Path(cache_dir) / "demonstrations_zip"

    download_and_unzip_demos([demo_id], cache_dir=cache_dir, repo_id=repo_id)

    return str(base_demo_dir / demo_id)

class WeblinxTasks:
    def __init__(
        self,
        demo_id,
        step_num,
        split,
        cache_dir="./bg_wl_data",
        goal_option="action_history",
        metadata_path=None,
        action_mapping=None,
    ):
        cache_dir = str(Path(cache_dir).expanduser())
        base_demo_dir = Path(cache_dir) / "demonstrations"
        base_zip_dir = Path(cache_dir) / "demonstrations_zip"

        logging.getLogger("PIL").setLevel(logging.INFO)

        if metadata_path is None:
            metadata_path = download_metadata(cache_dir=cache_dir)
        else:
            metadata_path = Path(metadata_path)

        # load the dataset
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        meta_split_dict = metadata[split]
        if demo_id not in meta_split_dict:
            raise ValueError(
                f"Demo '{demo_id}' not found in the test split '{split}' of WebLINX environment."
            )
        if str(step_num) not in meta_split_dict[demo_id]:
            raise ValueError(
                f"Step number '{step_num}' not found in the demo '{demo_id}' of the test split '{split}' of WebLINX environment."
            )

        task_dict = meta_split_dict[demo_id][str(step_num)]

        if task_dict["is_task"] is False:
            raise ValueError(
                f"Step number '{step_num}' of demo '{demo_id}' is not a proper task in the test split '{split}' of WebLINX environment."
            )

        # if the base_dir / project_id does not exist, download the dataset
        if not base_demo_dir.joinpath(demo_id).exists():
            download_and_unzip_demo(demo_id, cache_dir=cache_dir)

        # assign the attributes
        self.demo = wl.Demonstration(demo_id, base_dir=base_demo_dir)
        self.replay = wl.Replay.from_demonstration(self.demo)
        self.turn = self.replay[step_num]

        self.chat_messages = convert_weblinx_action_history_to_chat_messages(
            demo_id=demo_id,
            step_num=step_num,
            meta_split_dict=meta_split_dict,
        )

        self.action_mapping = action_mapping

        self.goal = create_goal_message(
            demo_id=demo_id,
            meta_split_dict=meta_split_dict,
            chat_messages=self.chat_messages,
            option=goal_option,
        )
        self.open_tabs = get_open_tabs(
            demo_id=demo_id, step_num=step_num, meta_split_dict=meta_split_dict
        )
        self.base_dir = str(base_demo_dir)
        self.meta_split_dict = meta_split_dict
        self.cache_dir = cache_dir
        self.demo_id = demo_id
        self.step_num = step_num
        self.current_url = task_dict["tab"]["url"]
        self.current_tab_id = task_dict["tab"]["id"]

        self.task_dict = task_dict

        self.screen_path = base_demo_dir / self.task_dict["screenshot_path"]
        self.html_path = base_demo_dir / self.task_dict["html_path"]
        self.bbox_path = base_demo_dir / self.task_dict["bbox_path"]
        self.axtree_path = base_demo_dir / self.task_dict["axtree_path"]
        self.dom_path = base_demo_dir / self.task_dict["dom_object_path"]
        self.extra_properties_path = base_demo_dir / self.task_dict["extra_props_path"]
        self.last_action = ""
        self.last_action_error = ""

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        self.start_time = time.time()

        active_index = get_active_index(
            current_tab_id=self.current_tab_id, open_tabs=self.open_tabs
        )

        with open(self.dom_path, "r") as f:
            dom = json.load(f)

        with open(self.axtree_path, "r") as f:
            axtree = json.load(f)

        with open(self.extra_properties_path, "r") as f:
            extra_properties = json.load(f)

        # also apply zoom level
        zoom = self.task_dict["zoom"]
        extra_properties = apply_zoom_to_extra_properties(extra_properties, zoom)

        # https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/env.py#L526-L542
        obs = {
            "chat_messages": self.chat_messages,
            "goal": self.goal,
            "goal_object": [{"type": "text", "text": self.goal}],
            "open_pages_urls": self.open_tabs["urls"],
            "open_pages_titles": self.open_tabs["titles"],
            "active_page_index": np.asarray([active_index]),
            "url": self.current_url,
            "screenshot": np.array(Image.open(self.screen_path).convert(mode="RGB")),
            "dom_object": dom,
            "axtree_object": axtree,
            "extra_element_properties": extra_properties,
            "focused_element_bid": None,  # we don't have this information
            "last_action": self.last_action,
            "last_action_error": self.last_action_error,
            "elapsed_time": np.asarray([time.time() - self.start_time]),
        }

        # Extra information, only needed for debugging
        info = {"task_info": {"weblinx_task_dict": self.task_dict}}

        self.obs = deepcopy(obs)
        self.info = deepcopy(info)

        return obs, info

    def step(self, action, *args, **kwargs):

        lst_args = args
        info = deepcopy(self.info)
        info["action_exec_start"] = time.time()
        info["action_exec_timeout"] = 0

        # we actually ignore the action, but keep the interface
        # to make it compatible with the rest of browsergym

        terminated = True  # always done after one step
        truncated = False  # always done after one step
        obs = self.obs  # keep the obs from the last step

        # https://github.com/McGill-NLP/weblinx/blob/b27473fc7ab523278ed7d0ea57d9525a447e28b8/weblinx/eval/__init__.py#L15

        # first, define the ground truth action
        ref_action = {
            "intent": self.task_dict["intent"],
            "args": self.task_dict["args"],
            "element": self.task_dict.get("element"),
        }

        bg_actions_args_orders = {
            "send_msg_to_user": ["text"],
            "click": ["bid"],
            "hover": ["bid"],
            "fill": ["bid", "value"],
            "goto": ["url"],
            "scroll": ["delta_x", "delta_y"],
            "new_tab": [],
            "tab_close": [],
            "tab_focus": ["index"],
        }

        try:
            if self.action_mapping:
                self.action_mapping(action)
            action = extract_function_call_from_action_string(action, only_first=True)
            # First, use parse_function_string to get the function name, args, and kwargs
            func_name, args, kwargs = parse_function_string(action)
            # Next, map the args to kwargs
            merged_kwargs = map_args_to_kwargs(
                func_name, args, kwargs, bg_actions_args_orders
            )
            intent, args = func_name, merged_kwargs
            logger.debug("=" * 80)
            logger.debug(f"raw intent, args: {intent}; {args}")
            logger.debug("-" * 80)
            intent, args = convert_browsergym_to_weblinx_action(
                intent=intent, args=args, warn_on_unknown=False
            )
            args = wlp.outputs.sanitize_args(args)
            infered_element = wlp.outputs.infer_element_for_action(
                intent=intent, args=args, turn=self.turn
            )
            pred_action = {"intent": intent, "args": args, "element": infered_element}

            # now, evaluate action against the ground truth
            # use pred_action and ref_action to compute the reward
            reward = get_reward(pred_action=pred_action, ref_action=ref_action)

            logger.debug(
                f"pred_action: {pred_action['intent']}, args={pred_action['args']}"
            )
            logger.debug(
                f"ref_action:  {ref_action['intent']}, args={ref_action['args']}"
            )
            logger.debug("-" * 80)
            logger.debug(f"reward: {reward}")
            logger.debug("=" * 80)

        except Exception as e:
            logger.error(f"Error in executing the action: {e}")
            reward = 0
            obs["last_action_error"] = str(f"Error in executing the action: {e}")
        info["action_exec_stop"] = time.time()

        return obs, reward, terminated, truncated, info


def make_pseudo_env(
    name,
    split,
    metadata_path="./metadata.json",
    cache_dir="./bg_wl_data",
    goal_option="action_history",
    action_mapping=None,
):
    name = name.split("/")[-1]
    dataset, demo_id, step_num = name.split(".")
    try:
        step_num = int(step_num)
    except ValueError:
        raise ValueError(f"Step should be an integer, but got {step_num}.")

    assert (
        dataset == "weblinx"
    ), "Only weblinx dataset is supported. Please use the format weblinx.<demo_id>.<step>"

    # load data from the dataset

    return WeblinxTasks(
        demo_id=demo_id,
        split=split,
        step_num=step_num,
        metadata_path=metadata_path,
        cache_dir=cache_dir,
        goal_option=goal_option,
        action_mapping=action_mapping,
    )


class WeblinxEnv(BrowserEnv):
    def __init__(
        self,
        # task-related arguments
        task_entrypoint: type[AbstractBrowserTask],
        task_kwargs: dict = {},
        viewport: Optional[dict] = None,  # will override the task's viewport
        slow_mo: Optional[int] = None,  # will override the task's slow_mo
        timeout: Optional[int] = None,  # will override the task's timeout
        tags_to_mark: Literal["all", "standard_html"] = "standard_html",
        # interactive / debugging arguments
        headless: bool = True,
        wait_for_user_message: bool = False,
        terminate_on_infeasible: bool = True,
        resizeable_window: bool = False,
        record_video_dir: Optional[str] = None,
        pw_chromium_kwargs: dict = {},
        pw_context_kwargs: dict = {},
        # agent-related arguments
        action_mapping: Optional[callable] = HighLevelActionSet().to_python_code,
    ):
        super().__init__(
            task_entrypoint=task_entrypoint,
            task_kwargs=task_kwargs,
            viewport=viewport,
            slow_mo=slow_mo,
            timeout=timeout,
            tags_to_mark=tags_to_mark,
            headless=headless,
            wait_for_user_message=wait_for_user_message,
            terminate_on_infeasible=terminate_on_infeasible,
            resizeable_window=resizeable_window,
            record_video_dir=record_video_dir,
            pw_chromium_kwargs=pw_chromium_kwargs,
            pw_context_kwargs=pw_context_kwargs,
            action_mapping=action_mapping,
        )

        goal_option = task_kwargs.get("goal_option", "action_history")

        default_cache_dir = task_kwargs.get("cache_dir", "./bg_wl_data")
        default_cache_dir = os.path.expanduser(default_cache_dir)

        cache_dir = os.getenv("BROWSERGYM_WEBLINX_CACHE_DIR", default_cache_dir)
        cache_dir = os.path.expanduser(cache_dir)

        metadata_path = task_kwargs.get(
            "metadata_path", os.path.join(cache_dir, "metadata.json")
        )
        self.weblinx_pseudo_env = make_pseudo_env(
            task_kwargs["task_name"],
            split=task_kwargs["split"],
            metadata_path=metadata_path,
            cache_dir=cache_dir,
            goal_option=goal_option,
            action_mapping=action_mapping,
        )

        # We need to have a chat in case browsergym.experiments.loop tries to add
        # a message to the chat
        self.chat = WindowlessChat(headless=False)

    def close(self):
        # delete the pseudo env
        del self.weblinx_pseudo_env
        self.weblinx_pseudo_env = None

    def reset(self, *args, **kwargs):
        return self.weblinx_pseudo_env.reset(*args, **kwargs)

    def step(self, action, *args, **kwargs):
        return self.weblinx_pseudo_env.step(action, *args, **kwargs)


def create_entry_point(
    task_name, split, cache_dir="./bg_wl_data", goal_option="action_history"
):
    def entry_point(*env_args, **env_kwargs):
        env_kwargs["task_kwargs"] = env_kwargs.get("task_kwargs", {})
        env_kwargs["task_kwargs"]["task_name"] = task_name
        env_kwargs["task_kwargs"]["split"] = split
        env_kwargs["task_kwargs"]["cache_dir"] = cache_dir
        env_kwargs["task_kwargs"]["goal_option"] = goal_option

        # We don't really need a task_entrypoint, so we set it to None
        return WeblinxEnv(task_entrypoint=None, *env_args, **env_kwargs)

    return entry_point


ALL_WEBLINX_TASK_IDS = []


def register_weblinx_tasks(
    split="test_iid",
    metadata_path=None,
    cache_dir="./bg_wl_data",
    goal_option="action_history",
):
    if os.getenv("BROWSERGYM_WEBLINX_CACHE_DIR") is not None:
        cache_dir = os.getenv("BROWSERGYM_WEBLINX_CACHE_DIR")

    if cache_dir.startswith("~"):
        cache_dir = os.path.expanduser(cache_dir)

    logger.info("Registering Weblinx tasks...")
    # reset ALL_WEBLINX_TASK_IDS
    ALL_WEBLINX_TASK_IDS.clear()

    tasks = list_tasks(split=split, metadata_path=metadata_path, cache_dir=cache_dir)
    task_class = None  # we don't really need it in our case, since we don't call the task class directly

    for task_shortname in tasks:
        # entry_point = lambda *env_args, **env_kwargs: WeblinxEnv(task_class, *env_args, **env_kwargs)
        gym.register(
            id=f"browsergym/{task_shortname}",
            entry_point=create_entry_point(
                task_shortname, split, cache_dir=cache_dir, goal_option=goal_option
            ),
            nondeterministic=False,
        )
        ALL_WEBLINX_TASK_IDS.append(task_shortname)

    logger.info("Weblinx tasks registered.")

def is_true(text: str) -> bool:
    true_equiv = {"true", "t", "yes", "y", "1", 1, "enable", "enabled"}

    return text.lower().strip() in true_equiv

def is_false(text: str) -> bool:
    false_equiv = {"false", "f", "no", "n", "0", 0, "disable", "disabled"}

    return text.lower().strip() in false_equiv

PREVENT_REGISTRATION = os.getenv("BROWSERGYM_WEBLINX_PREVENT_REGISTRATION", "false").lower()
REGISTER_TRAIN = os.getenv("BROWSERGYM_WEBLINX_REGISTER_TRAIN", "true").lower()
REGISTER_VALID = os.getenv("BROWSERGYM_WEBLINX_REGISTER_VALID", "true").lower()
REGISTER_TEST = os.getenv("BROWSERGYM_WEBLINX_REGISTER_TEST", "true").lower()
REGISTER_TEST_OOD = os.getenv("BROWSERGYM_WEBLINX_REGISTER_TEST_OOD", "true").lower()

if is_true(REGISTER_TRAIN) or is_true(PREVENT_REGISTRATION):
    logger.debug("Registering training tasks...")
    register_weblinx_tasks(split="train")
elif is_false(REGISTER_TRAIN):
    logger.debug("Skipping registration of training tasks.")
else:
    logger.warning(
        f"Invalid value for BROWSERGYM_WEBLINX_REGISTER_TRAIN: {REGISTER_TRAIN}. Skipping registration of training tasks."
    )

if is_true(REGISTER_VALID) or is_true(PREVENT_REGISTRATION):
    logger.debug("Registering validation tasks...")
    register_weblinx_tasks(split="valid")
elif is_false(REGISTER_VALID):
    logger.debug("Skipping registration of validation tasks.")
else:
    logger.warning(
        f"Invalid value for BROWSERGYM_WEBLINX_REGISTER_VALID: {REGISTER_VALID}. Skipping registration of validation tasks."
    )

if is_true(REGISTER_TEST) or is_true(PREVENT_REGISTRATION):
    logger.debug("Registering test tasks...")
    register_weblinx_tasks(split="test_iid")
elif is_false(REGISTER_TEST):
    logger.debug("Skipping registration of test tasks.")
else:
    logger.warning(
        f"Invalid value for BROWSERGYM_WEBLINX_REGISTER_TEST: {REGISTER_TEST}. Skipping registration of test tasks."
    )

if is_true(REGISTER_TEST_OOD) or is_true(PREVENT_REGISTRATION):
    for split in ['test_geo', 'test_web', 'test_cat', 'test_vis']:
        logger.debug(f"Registering {split} tasks...")
        register_weblinx_tasks(split=split)
elif is_false(REGISTER_TEST_OOD):
    logger.debug("Skipping registration of OOD test tasks.")
else:
    logger.warning(
        f"Invalid value for BROWSERGYM_WEBLINX_REGISTER_TEST_OOD: {REGISTER_TEST_OOD}. Skipping registration of OOD test tasks."
    )
