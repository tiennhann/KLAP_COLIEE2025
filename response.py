from typing import Literal


class Response:
    answer: Literal["Y", "N"]
    def __init__(self, answer):
        self.answer = answer