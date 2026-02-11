import pickle
import random

from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    relaxed_tasks as tasks_conditions_effects,
)
from calvin.calvin_models.calvin_agent.evaluation.multistep_sequences import (
    valid_task,
)


class StateBuffer:
    def __init__(self, tasks, max_size, valid_idx=None, buffer=None, balanced=False):
        self.balanced = balanced
        self.max_size = max_size
        self.buffer = []
        self.valid_idx = {}
        self.tasks = tasks
        for task in tasks:
            self.valid_idx[task] = []

    def add(self, state, task=None, exection=None):
        logic_state = state[0]
        if task is not None:
            # Adding for a specific task (no check)
            self.valid_idx[task].append(len(self.buffer))
        elif self.balanced:
            order = {task: len(self.valid_idx[task]) for task in self.tasks}
            sorted_tasks = sorted(order, key=order.get)
            for task in sorted_tasks:
                if (task != exection) and valid_task(
                    logic_state, tasks_conditions_effects[task]
                ):
                    self.valid_idx[task].append(len(self.buffer))
                    break
        else:
            for task in self.tasks:
                if (task != exection) and valid_task(
                    logic_state, tasks_conditions_effects[task]
                ):
                    self.valid_idx[task].append(len(self.buffer))
        self.buffer.append(state)

        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            for task in self.tasks:
                if self.valid_idx[task] and self.valid_idx[task][0] == 0:
                    self.valid_idx[task].pop(0)
                self.valid_idx[task] = [i - 1 for i in self.valid_idx[task]]

    def __len__(self):
        return len(self.buffer)

    def num_valid(self, task):
        return len(self.valid_idx[task])

    def sample(self, task, batch_size):
        indices = random.choices(self.valid_idx[task], k=batch_size)
        return [self.buffer[i] for i in indices]

    def save(self, filepath):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "buffer": self.buffer,
                    "valid_idx": self.valid_idx,
                    "tasks": list(self.tasks),
                    "max_size": self.max_size,
                },
                f,
            )

    def load(self, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.buffer = data["buffer"]
            self.valid_idx = data["valid_idx"]
            self.tasks = data["tasks"]
            self.max_size = data["max_size"]

    def get(self, task, idx):
        buffer_idx = self.valid_idx[task][idx]
        return self.buffer[buffer_idx]
