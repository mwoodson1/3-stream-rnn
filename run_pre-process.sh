#!/bin/bash
python pre-process/Cropping/crop.py
python pre-process/make_opt_flow.py
python pre-process/make_opt_flow.py -bw
python load_ucf.py
