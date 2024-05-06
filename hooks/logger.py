from hooks import Hook


class LoggerHook(Hook):
    """
    Base hook class
    """
    def before_run(self, runner) -> None:
        runner.accelerator.print("Starting Training")

    def before_train(self, runner) -> None:
        runner.accelerator.print(f"Starting epoch: {runner.hook_state.epoch}")

    def after_train_step(self, runner) -> None:
        runner.accelerator.print(f"Train loss: {runner.hook_state.train_loss_step}")
