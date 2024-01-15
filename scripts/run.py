from pathlib import Path

from PokerRL.cfr_runner import CFRRunner
from pdcfrplus.utils.exp import ServerFileStorageObserver, ex
from pdcfrplus.utils.logger import Logger
from pdcfrplus.utils.utils import init_object, load_module, run_method

from pdcfrplus.game.game_config import read_game_config


# flake8: noqa: C901
@ex.config
def config():
    seed = 0
    # init params
    algo_name = "CFR"
    game_name = "KuhnPoker"
    iterations = 2000
    gamma = None
    alpha = None
    beta = None

    # logger
    writer_strings = ["stdout"]
    save_log = False
    folder = Path(__file__).parents[1] / "results" / game_name / algo_name
    if save_log:
        writer_strings += ["csv", "sacred"]
        ex.observers.append(ServerFileStorageObserver(folder))


@ex.automain
def main(_config, _run, folder, game_name, algo_name):
    configs = dict(_config)
    for arg in ["gamma", "alpha", "beta"]:
        if configs[arg] is None:
            del configs[arg]
    if configs["save_log"]:
        configs["folder"] = folder / str(_run._id)

    logger = init_object(Logger, configs)

    if game_name in [
        "Subgame3",
        "Subgame4",
    ]:
        runner = init_object(CFRRunner, configs, logger=logger)
        runner.run()
    else:
        game_config = run_method(read_game_config, configs)
        solver_class = load_module("pdcfrplus.cfr:{}".format(algo_name))
        solver = init_object(
            solver_class, configs, game_config=game_config, logger=logger
        )
        solver.learn()
    logger.close()
