class Hook:
    """
    Base hook class
    """
    def __init__(self):
        pass

    def before_run(self, runner) -> None:
        pass

    def after_run(self, runner) -> None:
        pass

    def before_train(self, runner) -> None:
        pass

    def after_train(self, runner) -> None:
        pass

    def before_train_step(self, runner) -> None:
        pass

    def after_train_step(self, runner) -> None:
        pass

    def before_eval(self, runner) -> None:
        pass

    def after_eval(self, runner) -> None:
        pass
