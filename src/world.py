from scheduler import Scheduler
from models.registry import ModelGraph
from config import Device, ParallelConfig
from pydantic import BaseModel


class Node(BaseModel):
    num_devices: int
    device_type: Device

    parallel_cfg: ParallelConfig

    scheduler: Scheduler
    model: ModelGraph
