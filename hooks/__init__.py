from abc import ABC, abstractmethod

class Hook(ABC):
    """
    Base hook class
    """
    def __init__(self):
        pass

    @abstractmethod
    def before_run(self, runner) -> None:
        pass

    @abstractmethod
    def after_run(self, runner) -> None:
        pass

    @abstractmethod
    def before_train(self, runner) -> None:
        pass

    @abstractmethod
    def after_train(self, runner) -> None:
        pass

    @abstractmethod
    def before_eval(self, runner) -> None:
        pass

    @abstractmethod
    def after_eval(self, runner) -> None:
        pass
