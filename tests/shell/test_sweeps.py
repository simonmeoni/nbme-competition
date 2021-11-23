import pytest

from tests.helpers.run_command import run_command

"""
A couple of tests executing hydra sweeps.

Use the following command to skip slow tests:
    pytest -k "not slow"
"""


@pytest.mark.slow
def test_default_sweep():
    """Test default Hydra sweeper."""
    command = [
        "run.py",
        "-m",
        "datamodule.batch_size=2,4",
        "model.lr=0.01,0.02",
        "trainer=default",
        "++trainer.fast_dev_run=true",
    ]
    run_command(command)
