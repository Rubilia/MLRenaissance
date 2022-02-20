from abc import ABC, abstractmethod

from Algorithms.utils.Spaces import DiscreteActionSpace


class BaseDiscreteEnv(ABC):
    @abstractmethod
    def getDiscountFactor(self) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def getCurrentState(self):
        pass

    @abstractmethod
    def getStateSpace(self):
        pass

    @abstractmethod
    def getActionSpace(self):
        pass

    @abstractmethod
    def step(self, a: DiscreteActionSpace):
        pass

    @abstractmethod
    def isTerminated(self) -> bool:
        pass
