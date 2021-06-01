#!/usr/bin/env python

import os,sys,shlex,glob,subprocess,argparse
import numpy as np
from shutil import copyfile, rmtree
import time


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

def run_scintpipe(cleaned_archive,output_psr_utc_path,psrname,utcname,config_params,soft_path,job_type):

    panorama_png = glob.glob(os.path.join(output_psr_utc_path,"*panorama*png"))
    
    if job_type == "slurm":
        
        if not len(panorama_png) > 0:
            command = "python ScintPipeline.py -archivefile {0} -outdir {1}".format(cleaned_archive,output_psr_utc_path)
            job_name = "Scint_{0}_{1}.bash".format(psrname,utcname)
            with open(os.path.join(output_psr_utc_path,str(job_name)),'w') as job_file:
                job_file.write("#!/bin/bash \n")
                job_file.write("#SBATCH --job-name=Scint_{0}_{1} \n".format(psrname,utcname))
                job_file.write("#SBATCH --output={0}/ScintPipe_{1}_{2}.out \n".format(output_psr_utc_path,psrname,utcname))
                job_file.write("#SBATCH --ntasks={0} \n".format(config_params["tasks"]))
                job_file.write("#SBATCH --mem={0} \n".format(config_params["ram"]))
                job_file.write("#SBATCH --time={0} \n".format(config_params["time"]))
                job_file.write("#SBATCH --mail-type=FAIL --mail-user={0} \n".format(config_params["mail"]))
                job_file.write('cd {0} \n'.format(soft_path))
                job_file.write('{0}'.format(command))

            print ("Slurm job - {0} created".format(job_name))

            print ("Deploying {0}".format(job_name))
            com_sbatch = 'sbatch {0}'.format(os.path.join(output_psr_utc_path,str(job_name)))
            args_sbatch = shlex.split(com_sbatch)
            proc_sbatch = subprocess.Popen(args_sbatch)
            time.sleep(1)
            print("{0} deployed.".format(job_name))

        else:
            print ("Already processed")

    elif job_type == "direct":
        
        if not len(panorama_png) > 0:
            print ("Launching scintillation pipeline for {0}:{1}".format(psrname,utcname))
            command = "python {0}/ScintPipeline.py -archivefile {1} -outdir {2}".format(soft_path,cleaned_archive,output_psr_utc_path)
            args_pipe = shlex.split(command)
            proc_pipe = subprocess.call(args_pipe)

        else:
            print ("Already processed")


