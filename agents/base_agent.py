from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state):
        pass

    def update(self, *args, **kwargs):
        pass