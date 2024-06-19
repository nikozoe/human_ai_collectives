from dataclasses import dataclass


@dataclass
class MachineModel:
    org: str
    name: str
    parameters: dict
