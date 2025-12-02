# utils.py

import random

def get_color(allocated):
    """Return a color for allocated/free blocks."""
    if not allocated:
        return "#000000"  # light grey
    # random pastel colors
    return f"hsl({random.randint(0,360)}, 70%, 75%)"
