import os
from pathlib import Path

from absl import app, flags

FLAGS = flags.FLAGS
flags.DEFINE_enum(
    "algo",
    "CFR",
    [
        "CFR",
        "CFRPlus",
        "LinearCFR",
        "DCFR",
        "PCFRPlus",
        "PDCFRPlus",
        "DCFRPlus",
        "PDCFR",
    ],
    "name of the algorithm",
)
flags.DEFINE_float("alpha", None, "alpha")
flags.DEFINE_float("beta", None, "beta")
flags.DEFINE_float("gamma", None, "gamma")


def main(argv):
    algo_name = FLAGS.algo
    param = {}
    if FLAGS.alpha:
        param["alpha"] = FLAGS.alpha
    if FLAGS.beta:
        param["beta"] = FLAGS.beta
    if FLAGS.gamma:
        param["gamma"] = FLAGS.gamma
    game_names = [
        "KuhnPoker",
        "LeducPokerIso",
        "LiarsDice4",
        "LiarsDice5",
        "GoofSpiel4",
        "GoofSpiel5",
        "GoofSpielImp4",
        "GoofSpielImp5",
        "Battleship_22_3",
        "Battleship_32_3",
        "Subgame3",
        "Subgame4",
    ]
    for game_name in game_names:
        run_file = Path(__file__).absolute().parent / "run.py"
        script = "python {} with game_name={} algo_name={} iterations=20000 save_log=True ".format(
            run_file, game_name, algo_name
        )
        for param_name, param_value in param.items():
            script += "{}={} ".format(param_name, param_value)
        script = "nohup {} 2>&1 >/dev/null &".format(script)
        # os.system(script)

        os.system(script)
        # print(script)


if __name__ == "__main__":
    app.run(main)
