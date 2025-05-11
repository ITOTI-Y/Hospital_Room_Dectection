# src/analysis/__init__.py
"""Analysis module for calculating travel times and other graph metrics."""
from .travel_time import calculate_room_travel_times

__all__ = ["calculate_room_travel_times"]