import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from abc import abstractmethod
from log_tools.logger import (
    LoggerCollection,
    BaseMetricsLogger,
    ScalarLogger,
    MetricsTableLogger,
    DictLogger,
    ConfusionMatrixLogger,
    PerClassLogger,
)
from log_tools.metrics_manager import (
    MetricsManager,
    MetricsManagerCollection,
    BaseMetricsManager,
)
import config


class BaseLightningModule(LightningModule):
    def __init__(
        self,
        model,
        model_name,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
    ):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.loss_fn = loss_fn
        self.scheduler_max_it = scheduler_max_it
        self.weight_decay = weight_decay
        self.lr = lr
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.train_metrics_manager: BaseMetricsManager
        self.val_metrics_manager: BaseMetricsManager
        self.test_metrics_manager: BaseMetricsManager

        self.train_loss_logger: BaseMetricsLogger
        self.val_loss_logger: BaseMetricsLogger
        self.test_loss_logger: BaseMetricsLogger

        self.train_metrics_logger: BaseMetricsLogger
        self.val_metrics_logger: BaseMetricsLogger
        self.test_metrics_logger: BaseMetricsLogger

    @abstractmethod
    def configure_metrics_managers(self) -> None:
        """Configure metrics managers."""

    @abstractmethod
    def configure_loggers(self) -> None:
        """Configure loggers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _final_step(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Final step of the forward pass.
        This should be overriden by the subclass if needed

        Args:
        ----
            y_hat: Predicted outputs.

        Returns:
        -------
            Predicted outputs.

        """
        return y_hat

    def _common_step(self, batch: torch.Tensor, batch_idx: int) -> tuple:
        """Common step for training, validation, and testing.

        Args:
        ----
            batch: Input batch.
            batch_idx: Index of the current batch.

        Returns:
        -------
            Tuple containing the ground truth labels, predicted outputs, and loss.

        """
        x, y = batch
        outputs = self.forward(x)
        y_hat = outputs.logits
        loss = self.loss_fn(y_hat, y)
        return y, y_hat, loss

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)
        y_true = self.transform_y_true(y)
        print(f"\ny: {y.shape}\n y_hat: {y_hat.shape}")
        print(f"\ny: {y}\n y_hat: {y_hat}")
        print(f"\n - loss: {loss}")
        self.train_metrics_manager.update_metrics(y_true=y_true, y_pred=y_hat)

        self.train_loss_logger.log(loss)
        return {"loss": loss, "train/labels": y, "train/predictions": y_hat}

    def on_train_epoch_end(self) -> None:
        """Callback function called at the end of each training epoch.
        Computes and logs the training metrics.
        """
        self.train_metrics_logger.log()
        self.train_metrics_manager.reset_metrics()

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)
        y_true = self.transform_y_true(y)
        self.val_metrics_manager.update_metrics(y_true=y_true, y_pred=y_hat)

        self.val_loss_logger.log(loss)
        return {"loss": loss, "val/labels": y, "val/predictions": y_hat}

    def on_validation_epoch_end(self) -> None:
        """Callback function called at the end of each validation epoch.
        Computes and logs the validation metrics.
        """
        self.val_metrics_logger.log()
        self.val_metrics_manager.reset_metrics()

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> None:
        y, y_hat, loss = self._common_step(batch, batch_idx)
        y_hat = self._final_step(y_hat)

        y_true = self.transform_y_true(y)
        self.test_metrics_manager.update_metrics(y_true=y_true, y_pred=y_hat)

        self.test_loss_logger.log(loss)
        return {"loss": loss, "test/labels": y, "test/predictions": y_hat}

    def on_test_epoch_end(self) -> None:
        self.test_metrics_logger.log()
        self.test_metrics_manager.reset_metrics()

    def configure_optimizers(self) -> tuple:
        optimizer = Adam(self.model.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.scheduler_max_it)
        return [optimizer], [scheduler]

    def transform_y_true(self, y) -> torch.Tensor:
        return torch.stack(
            [
                torch.Tensor([1, 0]) if label == 0 else torch.Tensor([0, 1])
                for label in y
            ],
        ).to(config.DEVICE)


class ClassificationLightningModule(BaseLightningModule):
    def __init__(
        self,
        model,
        model_name,
        class_names,
        loss_fn,
        metrics,
        lr,
        scheduler_max_it,
        weight_decay=0,
        per_class_metrics=None,
    ) -> None:
        super().__init__(
            model=model,
            model_name=model_name,
            loss_fn=loss_fn,
            metrics=metrics,
            lr=lr,
            scheduler_max_it=scheduler_max_it,
            weight_decay=weight_decay,
        )
        self.class_names = class_names
        self.per_class_metrics = per_class_metrics
        self.test_all_preds = []
        self.test_all_targets = []

        self.configure_metrics_managers()
        self.configure_loggers()

    def configure_metrics_managers(self) -> None:
        self.train_metrics_manager = MetricsManager(
            module=self,
            metrics=self.train_metrics,
        )
        self.val_metrics_manager = MetricsManager(
            module=self,
            metrics=self.val_metrics,
        )

        if self.per_class_metrics is not None:
            self.test_metrics_manager = MetricsManagerCollection(
                module=self,
                managers=[
                    MetricsManager(module=self, metrics=self.test_metrics),
                    MetricsManager(module=self, metrics=self.per_class_metrics),
                ],
            )
        else:
            self.test_metrics_manager = MetricsManager(
                module=self,
                metrics=self.test_metrics,
            )

    def configure_loggers(self) -> None:
        self.train_loss_logger = ScalarLogger(
            prefix="train/",
            module=self,
            scalar_name="loss",
        )
        self.val_loss_logger = ScalarLogger(
            prefix="val/",
            module=self,
            scalar_name="loss",
        )
        self.test_loss_logger = ScalarLogger(
            prefix="test/",
            module=self,
            scalar_name="loss",
        )

        self.train_metrics_logger = DictLogger(
            prefix="train/",
            module=self,
            metrics=self.train_metrics,
        )
        self.val_metrics_logger = DictLogger(
            prefix="val/",
            module=self,
            metrics=self.val_metrics,
        )

        if self.per_class_metrics is not None:
            self.test_metrics_logger = LoggerCollection(
                prefix="test/",
                module=self,
                loggers=[
                    DictLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                    ),
                    MetricsTableLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                        table_name="Metrics",
                    ),
                    PerClassLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.per_class_metrics,
                    ),
                    ConfusionMatrixLogger(
                        prefix="test/",
                        module=self,
                        y_true_ref=self.test_all_targets,
                        y_pred_ref=self.test_all_preds,
                    ),
                ],
            )
        else:
            self.test_metrics_logger = LoggerCollection(
                prefix="test/",
                module=self,
                loggers=[
                    DictLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                    ),
                    MetricsTableLogger(
                        prefix="test/",
                        module=self,
                        metrics=self.test_metrics,
                        table_name="Metrics",
                    ),
                    ConfusionMatrixLogger(
                        prefix="test/",
                        module=self,
                        y_true_ref=self.test_all_targets,
                        y_pred_ref=self.test_all_preds,
                    ),
                ],
            )

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        test_results = super().test_step(batch, batch_idx)

        self.test_all_preds.append(test_results["test/predictions"])
        self.test_all_targets.append(test_results["test/labels"])

        return test_results

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        self.test_all_preds = []
        self.test_all_targets = []


def testing() -> None:
    class SoftmaxTester(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return self.predict(x)

        def predict(self, y_hat):
            # Apply softmax and argmax while maintaining the shape of y_hat
            softmaxed_y_hat = torch.softmax(y_hat, dim=-1)
            return torch.argmax(softmaxed_y_hat, dim=-1, keepdim=True)

    # Example usage
    model = SoftmaxTester()
    input_tensor = torch.randn(3, 2)  # Example input tensor
    print("Input: ", input_tensor)
    output = model(input_tensor)
    print("Shape of output:", output.shape)
    print("Values", output)

    print(torch.argmax(torch.Tensor([0.5, 0.9])))


if __name__ == "__main__":
    testing()
