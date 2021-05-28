#!/usr/bin/env python

import os,sys,shlex,glob,subprocess,argparse
import numpy as np
from shutil import copyfile, rmtree


def parse_config(path_cfile):
    """
    INPUT: Path to the configuration file
    """
    
    config_params = {}
    with open (str(path_cfile)) as cfile:
        for line in cfile.readlines():
            sline = line.split("=")
            attr = (sline[0].rstrip())
            if attr == 'input_path':
                config_params["input_path"] = sline[1].rstrip().lstrip(' ')
            if attr == 'output_path':
                config_params["output_path"] = sline[1].rstrip().lstrip(' ')
            if attr == "user":
                config_params["user"] = sline[1].rstrip().lstrip(' ')
            if attr == "ram":
                config_params["ram"] = sline[1].rstrip().lstrip(' ')
            if attr == "tasks":
                config_params["tasks"] = sline[1].rstrip().lstrip(' ')
            if attr == "time":
                config_params["time"] = sline[1].rstrip().lstrip(' ')
            if attr == "mail":
                config_params["mail"] = sline[1].rstrip().lstrip(' ')

    cfile.close()
    
    return config_params



