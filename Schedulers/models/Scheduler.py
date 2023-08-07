from abc import ABC, abstractclassmethod
from typing import List


class Scheduler(ABC):
    def __init__(self):
        super().__init__()

    @abstractclassmethod
    def scheduler_algorithm(
        self,
        a: List[List[float]],
        b: List[List[float]],
        c: List[List[float]],
    ):
        """
        Method to create the scheduler list
        Returns:
            list of dict with the scheduler information
            [
                {
                    "Cluster": 0,
                    "SM": 0,
                    "CTA": {
                        "id": 0,
                        "x": 0,
                        "y": 0
                    }
                }
            ]
        """
        pass
