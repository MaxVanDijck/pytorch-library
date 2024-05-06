from hooks import Hook
import wandb

class WandbHook(Hook):
    """
    Base hook class
    """
    def before_run(self, runner) -> None:
        self.wandb = wandb.init(
            project="test-project",
            config=dict(runner.config.hydra)
        )

    def after_train_step(self, runner) -> None:
        self.wandb.log({"train_loss_step": runner.hook_state.train_loss_step})

