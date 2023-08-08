from abc import ABC, abstractclassmethod
from typing import List
import random




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
    import random

    def random_delay_generator_simulator(self, l):
        input_list = []
        output_list = []
        for i in range(l):
            input_list.append(i)

        for i in range(l):
            if((len(input_list)-1) != 0):
                element = random.randint(0, len(input_list)-1)
                output_list.append(input_list[element])
                input_list.remove(input_list[element])
            else:
                output_list.append(input_list[0])
    
        return output_list