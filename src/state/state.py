from dataclasses import dataclass


@dataclass
class State:
    """
    状態を管理するクラス
    """

    iteration: int = 0
