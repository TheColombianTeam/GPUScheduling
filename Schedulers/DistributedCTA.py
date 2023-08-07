from .models import Scheduler


class DistributedCTA(Scheduler.Scheduler):
    def __init__(self):
        super().__init__()

    def scheduler_algorithm(self, a, b, c):
        pass
