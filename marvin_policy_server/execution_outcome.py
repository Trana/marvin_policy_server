"""Transport-independent result of a policy lifecycle request."""

from dataclasses import dataclass


@dataclass
class ExecutionOutcome:
    success: bool = False
    message: str = ''
