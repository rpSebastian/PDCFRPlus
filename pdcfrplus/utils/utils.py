import datetime
import importlib
import inspect
import pickle
import random
import time
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, Type

import numpy as np
import torch


def load_module(name):
    if ":" in name:
        mod_name, attr_name = name.split(":")
    else:
        li = name.split(".")
        mod_name, attr_name = ".".join(li[:-1]), li[-1]
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, attr_name)
    return fn


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def init_cuda():
    a = np.array([100, 500])
    a_cuda = torch.from_numpy(a).cuda()


class Timer:
    def __init__(self):
        self.func_single_time = defaultdict(float)
        self.func_total_time = defaultdict(float)

    def timer(self, func):
        def wrap_func(*args, **kwargs):
            t1 = time.time()
            self.active = True
            result = func(*args, **kwargs)
            self.activate = False
            t2 = time.time()
            if not self.activate:
                self.func_single_time[func.__name__] = t2 - t1
                self.func_total_time[func.__name__] += t2 - t1
            return result

        return wrap_func

    def reset(self):
        self.func_single_time = defaultdict(float)
        self.func_total_time = defaultdict(float)

    def print_total(self):
        for key, value in self.func_total_time.items():
            print(f"{key}: {value:.4f}", end=" ")
        print()


timer = Timer()


def save_pickle(data, path):
    new_path = path.parent / (path.name + ".pkl")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    with open(new_path, "wb") as f:
        pickle.dump(data, f)


def load_pickle(path):
    new_path = path.parent / (path.name + ".pkl")
    with open(new_path, "rb") as f:
        data = pickle.load(f)
    return data


def init_object(init_class: Type[object], possible_args: Dict[str, Any], **kwargs):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(init_class.__init__).args
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    new_object = init_class(**params)
    return new_object


def run_method(
    method: Callable,
    possible_args: Dict[str, Any],
    **kwargs,
):
    possible_args_copy = deepcopy(possible_args)
    for k, v in kwargs.items():
        possible_args_copy[k] = v
    args = inspect.getfullargspec(method).args
    params = {k: v for k, v in possible_args_copy.items() if k in args}
    result = method(**params)
    return result


g_log_time = defaultdict(list)


def log_time(text):
    def decorator(func):
        def wrapper(*args, **kws):
            start = datetime.datetime.now()
            result = func(*args, **kws)
            end = datetime.datetime.now()
            time = (end - start).total_seconds()
            g_log_time[text].append(time)
            return result

        return wrapper

    return decorator


def log_time_func(text, end=False):
    now = datetime.datetime.now()
    if (
        text in g_log_time
        and len(g_log_time[text]) > 0
        and isinstance(g_log_time[text][-1], datetime.datetime)
    ):
        start = g_log_time[text][-1]
        t = (now - start).total_seconds()
        g_log_time[text][-1] = t
    if not end:
        g_log_time[text].append(datetime.datetime.now())


def print_time():
    for item in g_log_time.items():
        if len(item) <= 1 or len(item[1]) == 0 or len(item[0]) == 0:
            continue
        mean = np.mean(item[1])
        max = np.max(item[1])
        print(
            "{} | mean:{:.3f}ms, max:{:.3f}ms, times:{}".format(
                item[0], mean * 1000, max * 1000, len(item[1])
            )
        )
        g_log_time[item[0]] = []


def get_host_ip():
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip


def get_server_id():
    ip = get_host_ip()
    server_id = int(ip.split(".")[-1])
    return server_id
