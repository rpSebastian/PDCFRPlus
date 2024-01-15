from pathlib import Path

import pyspiel


class GameConfig:
    def __init__(
        self,
        iterations=1000,
        game_name=None,
        transform=False,
    ):
        self.game_name = game_name
        self.params = {}
        self.transform = transform
        self.iterations = iterations
        self.name = self.__class__.__name__

    def load_game(self):
        params = {}
        for p, v in self.params.items():
            if p == "filename":
                v = str(Path(__file__).absolute().parents[2] / v)
            params[p] = v
        game = pyspiel.load_game(self.game_name, params)
        if self.transform:
            game = pyspiel.convert_to_turn_based(game)
        self.num_nodes = 0
        self.calc_nodes(game.new_initial_state())
        return game

    def get_draw_file(self):
        file = (
            Path(__file__).absolute().parents[2]
            / "images"
            / "VIS_{}.pdf".format(self.name)
        )
        file.parent.mkdir(parents=True, exist_ok=True)
        return file

    def __repr__(self):
        return "{}({})".format(self.name, self.iterations)

    def calc_nodes(self, h):
        self.num_nodes += 1
        if h.is_terminal():
            return
        for a in h.legal_actions():
            self.calc_nodes(h.child(a))

    def visulize(self):
        from open_spiel.python.visualizations import treeviz

        game = self.load_game()
        gametree = treeviz.GameTree(
            game,
            node_decorator=_zero_sum_node_decorator,
            group_infosets=True,
            group_terminal=False,
            group_pubsets=False,
            target_pubset="*",
        )
        gametree.draw(str(self.get_draw_file()), prog="dot")


def _zero_sum_node_decorator(state):
    """Custom node decorator that only shows the return of the first player."""
    from open_spiel.python.visualizations import treeviz

    attrs = treeviz.default_node_decorator(state)  # get default attributes
    if state.is_terminal():
        attrs["label"] = str(int(state.returns()[0]))
    return attrs


class KuhnPoker(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="kuhn_poker",
        )



class BattleShip(GameConfig):
    def __init__(
        self,
        iterations=1000,
        board_width=2,
        board_height=2,
        num_shots=2,
    ):
        super().__init__(
            iterations=iterations,
            game_name="battleship",
        )
        self.params = {
            "board_width": board_width,
            "board_height": board_height,
            "ship_sizes": "[2]",
            "ship_values": "[2]",
            "num_shots": num_shots,
            "allow_repeated_shots": False,
        }
        self.name = "Battleship_{}{}_{}".format(board_width, board_height, num_shots)


class LeducPoker(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="leduc_poker",
        )


class LeducPokerIso(GameConfig):
    def __init__(self, iterations=1000):
        super().__init__(
            iterations=iterations,
            game_name="leduc_poker",
        )
        self.params = {"suit_isomorphism": True}


class GoofSpiel(GameConfig):
    def __init__(
        self,
        iterations=1000,
        num_cards=3,
        imp_info=False,
        points_order="descending",
        returns_type="win_loss",
    ):
        super().__init__(
            iterations=iterations,
            game_name="goofspiel",
            transform=True,
        )
        self.params = {
            "num_cards": num_cards,
            "imp_info": imp_info,
            "points_order": points_order,
            "returns_type": returns_type,
        }
        self.name = "GoofSpiel"
        if imp_info:
            self.name += "Imp"
        self.name += str(num_cards)


class LiarsDice(GameConfig):
    def __init__(
        self,
        iterations=1000,
        dice_sides=3,
        num_dice=1,
    ):
        super().__init__(
            iterations=iterations,
            game_name="liars_dice",
        )
        self.params = {"numdice": num_dice, "dice_sides": dice_sides}
        self.name = "LiarsDice{}".format(dice_sides)


def read_game_config(game_name, iterations=1000) -> GameConfig:
    game_configs = [
        KuhnPoker(iterations),
        LeducPoker(iterations),
        LeducPokerIso(iterations),
        *[GoofSpiel(iterations, num_cards=cards) for cards in [4, 5]],
        *[
            GoofSpiel(iterations, num_cards=cards, imp_info=True)
            for cards in [4, 5]
        ],
        *[LiarsDice(iterations, dice_sides=sides, num_dice=1) for sides in [4, 5]],
        BattleShip(iterations, board_width=3, board_height=2, num_shots=3),
        BattleShip(iterations, board_width=2, board_height=2, num_shots=3),
    ]
    game_dict = {game_config.name: game_config for game_config in game_configs}
    game_config = game_dict[game_name]
    return game_config


if __name__ == "__main__":
    game_config = read_game_config("Battleship_22_3")
