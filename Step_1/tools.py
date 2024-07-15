from __future__ import annotations

import random
import time


def get_current_wait_time(hospital: str) -> int | str:
    """Dummy function to generate fake wait times"""
    if hospital not in ["A", "B", "C", "D"]:
        return f"Hospital {hospital} does not exist"
    time.sleep(1)
    return random.randint(0, 1000)