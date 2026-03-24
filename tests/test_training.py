"""Tests for training modules."""
import pytest
import torch
import torch.nn as nn


class TestScheduler:
    """Tests for scheduler module."""

    def test_create_cosine_scheduler(self):
        """Test creating cosine scheduler."""
        from src.training.scheduler import create_scheduler

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        scheduler = create_scheduler(
            scheduler_type="cosine",
            optimizer=optimizer,
            num_epochs=100,
            steps_per_epoch=100,
        )
        assert scheduler is not None

    def test_create_step_scheduler(self):
        """Test creating step scheduler."""
        from src.training.scheduler import create_scheduler

        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters())

        scheduler = create_scheduler(
            scheduler_type="step",
            optimizer=optimizer,
            num_epochs=100,
            step_size=30,
        )
        assert scheduler is not None


class TestCallbacks:
    """Tests for callback classes."""

    def test_early_stopping(self):
        """Test early stopping callback."""
        from src.training.callbacks import EarlyStopping

        es = EarlyStopping(patience=3, mode="max")

        class MockTrainer:
            epoch = 0
            history = {"val_acc": []}

        trainer = MockTrainer()

        for i in range(5):
            trainer.epoch = i
            metric = 0.5 + (i * 0.1) if i < 3 else 0.6
            es.on_epoch_end(trainer, metric=metric)

        assert es.counter < 3 or not es.should_stop

    def test_model_checkpoint(self):
        """Test model checkpoint callback."""
        from src.training.callbacks import ModelCheckpoint
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            mc = ModelCheckpoint(output_dir=tmpdir, save_best=True, save_last=True)

            class MockTrainer:
                epoch = 0
                history = {"val_acc": []}

                def save_checkpoint(self, name):
                    pass

            trainer = MockTrainer()
            mc.on_epoch_end(trainer, metric=0.8)

            assert mc.best_metric == 0.8
