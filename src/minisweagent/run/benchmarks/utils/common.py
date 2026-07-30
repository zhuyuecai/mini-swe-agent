"""Shared agent utilities for benchmark runners."""

from minisweagent.agents import get_agent_class
from minisweagent.agents.default import DefaultAgent
from minisweagent.run.benchmarks.utils.batch_progress import RunBatchProgressManager


class ProgressTrackingMixin:
    """Agent that reports per-step progress via :class:`RunBatchProgressManager`."""

    def __init__(self, *args, progress_manager: RunBatchProgressManager, instance_id: str = "", **kwargs):
        super().__init__(*args, **kwargs)
        self.progress_manager = progress_manager
        self.instance_id = instance_id

    def step(self) -> dict:
        self.progress_manager.update_instance_status(self.instance_id, f"Step {self.n_calls + 1:3d} (${self.cost:.2f})")
        return super().step()


class ProgressTrackingAgent(ProgressTrackingMixin, DefaultAgent):
    pass


def get_progress_tracking_agent_class(agent_class: str = ""):
    if not agent_class:
        return ProgressTrackingAgent
    base_class = get_agent_class(agent_class)
    return type(f"ProgressTracking{base_class.__name__}", (ProgressTrackingMixin, base_class), {})
