#!/usr/bin/env python

import argparse
import random
import time
import datetime
import json
import numpy as np

import sys
import pandas as pd

sys.path.append("../../")
from scripts.sawmill.sawmill import Sawmill, SawmillEdge
from scripts.generate.xyzw_gen import xyzw_gen, DEFAULT_ARGS
import datetime
import json



def temp_gen(l, v, v_max):
    # Dump arguments
    ts = datetime.datetime.utcnow()
    filename = f"~/causal-log/datasets_raw/micro-causal-unit/log_{ts.year:04d}-{ts.month:02d}-{ts.day:02d}_{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}"
    print(f"l: {l}, v: {v}")
    with open(filename + ".log", "a+") as f:
        ts_str = f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}T{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}.{ts.microsecond:06d}Z DATA"
        first_line =   ts_str + " 1" * v + "\n"
        f.write(first_line)
        ts += datetime.timedelta(milliseconds=1)

        for i in range(l-1):
            ts_str = f"{ts.year:04d}-{ts.month:02d}-{ts.day:02d}T{ts.hour:02d}:{ts.minute:02d}:{ts.second:02d}.{ts.microsecond:06d}Z DATA"
            other_line = ts_str + " 0" * v + "\n"
            f.write(other_line)

            ts += datetime.timedelta(milliseconds=1)

    return filename + ".log"


def main():
    sys.path.append("../")
    
    v_min = 10
    v_max = 100
    vs = [i for i in range(v_min, v_max+1, 10)]
    l = 10000

    f = open("micro-causal-unit_runlog.txt", "w+")
    fr1 = open("micro-causal-unit_resultslog1.txt", "w+")
    fr1.write("Length, Lines Per Causal Unit, Parse Time, Prep Time\n")
    sys.stdout = f

    filename = temp_gen(l, 1000, 1000)
    s = Sawmill(filename, workdir="~/causal-log/datasets/micro-causal-unit", skip_writeout=True)
    parse_time = s.parse(sim_thresh=2/(1000+3))
    
    for v in vs:
        s.set_causal_unit(var_name="Timestamp", time_granularity=v)
        prep_time = s.prepare(custom_imp=s.ffill_imp)
        print(s.prepared.head(10))

        fr1.write(f"{l},{v},{parse_time},{prep_time}\n")
        fr1.flush()

    f.close()
    fr1.close()


if __name__ == "__main__":
    main()
