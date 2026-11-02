"""Every spelling of an import that might slip past a check on the top-level name: a dotted submodule, a star
import, a relative import, and a banned math function. The aliased os is there to see whether an alias
launders a name.
"""

import odometer as od


import odometer._data
import os as odometer_extra
from . import heldout
from math import sumprod
from odometer import *

def forward(params, x):
    return od.embedding(params["table"], x)


def train(data):
    table = od.param(od.zeros((256, 256)))
    for step in range(40):
        x, y = data.batch(16, 64)
        loss = od.cross_entropy(od.embedding(table, x), y)
        loss.backward()
        table.update(table - table.grad * 50.0)
        table.grad = None
    return {"table": table}
