#!/usr/bin/env python

import os,sys,glob,argparse,subprocess,shlex
import numpy as np
import psrchive as ps
from utils_tpa_scint import parse_config,run_scintpipe
import pandas as pd
import time


parser = argparse.ArgumentParser(description="Run Launch TPA Scintillation pipeline")
parser.add_argument("-cfile", dest="configfile", help="Path to the configuration file")
parser.add_argument("-psr", dest="psrname", help="Process the specified pulsar")
parser.add_argument("-utc", dest="utc", help="Process a particular UTC. Should be in conjunction with a pulsar name")
parser.add_argument("-list", dest="list", help="Process the pulsar-utc list")
parser.add_argument("-slurm", dest="slurm", help="Processes using Slurm",action="store_true")
args = parser.parse_args()

#Parsing the configuration file
config_params = parse_config(str(args.configfile))

if args.list:
    psr_utc_df = pd.read_csv(str(args.list), delimiter=" ", header=None, dtype=str, comment="#",names=["psrname","utc"])

if args.utc:
    obs_path = os.path.join(os.path.join(config_params["input_path"],str(args.psrname)),str(args.utc))
else:
    if args.list:
        obs_list = []
        for index, row in psr_utc_df.iterrows():
            psr_utc_path = os.path.join(os.path.join(config_params["input_path"],row["psrname"]),row["utc"])
            obs_list.append(psr_utc_path)
    else:
        obs_list = sorted(glob.glob(os.path.join(config_params["input_path"],"{0}/2*".format(args.psrname))))

#Checking validity of the input and output paths
if args.utc:
    if not os.path.exists(obs_path):
        print ("Pulsar observation path not valid. Quitting.")
        sys.exit()

if not os.path.exists(config_params["output_path"]):
    print ("Output path not valid. Quitting.")
    sys.exit()

#Creating output directory structure
if args.utc:
    output_psr_utc_path = "{0}/{1}/{2}/".format(config_params["output_path"],args.psrname,args.utc)
    if not os.path.exists(output_psr_utc_path):
        os.makedirs(output_psr_utc_path)
        print ("Output PSR-UTC directory structure created")

else:
    output_psr_utc_list = []
    for obs in obs_list:
        utcname = os.path.split(obs)[-1]
        psrname = os.path.split(os.path.split(obs)[0])[-1]
        output_psr_utc_path = "{0}/{1}/{2}/".format(config_params["output_path"],psrname,utcname)
        if not os.path.exists(output_psr_utc_path):
            os.makedirs(output_psr_utc_path)
        output_psr_utc_list.append(output_psr_utc_path)
        print ("Output directories created for {0}:{1}".format(psrname,utcname))
      q
#Obtaining the cleaned archive
if args.utc:
    cleaned_archive = glob.glob(os.path.join(obs_path,"*/*/cleaned/*.ar"))
    if len(cleaned_archive) > 0:
        print ("Processing {0}".format(cleaned_archive[0]))
    else:
        print ("Cleaned archive for {0}:{1} does not exist. Quitting".format(args.psrname,args.utc))
        sys.exit()
else:
    cleaned_archive_list = []
    for obs in obs_list:
        utcname = os.path.split(obs)[-1]
        psrname = os.path.split(os.path.split(obs)[0])[-1]
        cleaned_archive = glob.glob(os.path.join(obs,"*/*/cleaned/*.ar"))
        if len(cleaned_archive) > 0:
            print ("Adding {0} to processing list".format(cleaned_archive))
            cleaned_archive_list.append(cleaned_archive[0])
        else:
            print ("Cleaned archive for {0}:{1} does not exist. Skipping this UTC".format(psrname,utcname))
            output_path = "{0}/{1}/{2}/".format(config_params["output_path"],psrname,utcname)
            if os.path.exists(output_path):
                os.rmdir(output_path)


#Launching the Scintillation pipeline 
soft_path = "/fred/oz002/aparthas/TPA/Scintillation/meerkat-scintpipe"

if args.slurm:
    if args.utc:
        run_scintpipe(cleaned_archive[0],output_psr_utc_path,args.psrname,args.utc,config_params,soft_path,"slurm")
    else:
        for num,obs in enumerate(obs_list):
            utcname = os.path.split(obs)[-1]
            psrname = os.path.split(os.path.split(obs)[0])[-1]
            run_scintpipe(cleaned_archive_list[num],output_psr_utc_list[num],psrname,utcname,config_params,soft_path,"slurm")
else:
    if args.utc:
        run_scintpipe(cleaned_archive[0],output_psr_utc_path,args.psrname,args.utc,config_params,soft_path,"direct")

    else:
        for num,obs in enumerate(obs_list):
            utcname = os.path.split(obs)[-1]
            psrname = os.path.split(os.path.split(obs)[0])[-1]
            run_scintpipe(cleaned_archive_list[num],output_psr_utc_list[num],psrname,utcname,config_params,soft_path,"direct")



