#!/usr/bin/env python

import os,sys,glob,argparse,subprocess,shlex
import numpy as np
import psrchive as ps
from utils_tpa_scint import parse_config


parser = argparse.ArgumentParser(description="Run Launch TPA Scintillation pipeline")
parser.add_argument("-cfile", dest="configfile", help="Path to the configuration file")
parser.add_argument("-psr", dest="psrname", help="Process the specified pulsar")
parser.add_argument("-utc", dest="utc", help="Process a particular UTC. Should be in conjunction with a pulsar name")
parser.add_argument("-slurm", dest="slurm", help="Processes using Slurm",action="store_true")
args = parser.parse_args()

#Parsing the configuration file
config_params = parse_config(str(args.configfile))

if args.utc:
    obs_path = os.path.join(os.path.join(config_params["input_path"],str(args.psrname)),str(args.utc))
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
        output_psr_utc_path = "{0}/{1}/{2}/".format(config_params["output_path"],args.psrname,utcname)
        if not os.path.exists(output_psr_utc_path):
            os.makedirs(output_psr_utc_path)
        output_psr_utc_list.append(output_psr_utc_path)
        print ("Output directories created for {0}:{1}".format(args.psrname,utcname))
       

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
        cleaned_archive = glob.glob(os.path.join(obs,"*/*/cleaned/*.ar"))
        if len(cleaned_archive) > 0:
            print ("Adding {0} to processing list".format(cleaned_archive))
            cleaned_archive_list.append(cleaned_archive[0])
        else:
            print ("Cleaned archive for {0}:{1} does not exist. Skipping this UTC".format(args.psrname,utcname))
            output_path = "{0}/{1}/{2}/".format(config_params["output_path"],args.psrname,utcname)
            if os.path.exists(output_path):
                os.rmdir(output_path)
            
#Launching the Scintillation pipeline 
soft_path = "/fred/oz002/aparthas/TPA/Scintillation/meerkat-scintpipe"

if args.slurm:

    if args.utc:
        command = "python ScintPipeline.py -archivefile {0} -outdir {1}".format(cleaned_archive[0],output_psr_utc_path)
        job_name = "Scint_{0}_{1}.bash".format(args.psrname,args.utc)
        with open(os.path.join(output_psr_utc_path,str(job_name)),'w') as job_file:
            job_file.write("#!/bin/bash \n")
            job_file.write("#SBATCH --job-name=Scint_{0}_{1} \n".format(args.psrname,args.utc))
            job_file.write("#SBATCH --output={0}/ScintPipe_out_{1}_{2} \n".format(output_psr_utc_path,args.psrname,args.utc))
            job_file.write("#SBATCH --ntasks={0} \n".format(config_params["tasks"]))
            job_file.write("#SBATCH --mem={0} \n".format(config_params["ram"]))
            job_file.write("#SBATCH --time={0} \n".format(config_params["time"]))
            job_file.write("#SBATCH --mail-type=FAIL --mail-user={0} \n".format(config_params["mail"]))
            job_file.write('cd {0} \n'.format(soft_path))
            job_file.write('tcsh load_scintmodules.sh \n')
            job_file.write('{0}'.format(command))

        print ("Slurm job - {0} created".format(job_name))

        #print ("Deploying {0}".format(job_name))
        #com_sbatch = 'sbatch {0}'.format(os.path.join(output_psr_utc_path,str(job_name)))
        #args_sbatch = shlex.split(com_sbatch)
        #proc_sbatch = subprocess.Popen(args_sbatch)
        #time.sleep(1)
        #print("{0} deployed.".format(job_name))

    else:

        for num,obs in enumerate(obs_list):
            utcname = os.path.split(obs)[-1]
            command = "python ScintPipeline.py -archivefile {0} -outdir {1}".format(cleaned_archive_list[num],output_psr_utc_list[num])
            job_name = "Scint_{0}_{1}.bash".format(args.psrname,utcname)
            with open(os.path.join(output_psr_utc_list[num],str(job_name)),'w') as job_file:
                job_file.write("#!/bin/bash \n")
                job_file.write("#SBATCH --job-name=Scint_{0}_{1} \n".format(args.psrname,utcname))
                job_file.write("#SBATCH --output={0}/ScintPipe_out_{1}_{2} \n".format(output_psr_utc_list[num],args.psrname,utcname))
                job_file.write("#SBATCH --ntasks={0} \n".format(config_params["tasks"]))
                job_file.write("#SBATCH --mem={0} \n".format(config_params["ram"]))
                job_file.write("#SBATCH --time={0} \n".format(config_params["time"]))
                job_file.write("#SBATCH --mail-type=FAIL --mail-user={0} \n".format(config_params["mail"]))
                job_file.write('cd {0} \n'.format(soft_path))
                job_file.write('{0}'.format(command))

            print ("Slurm job - {0} created".format(job_name))

            #print ("Deploying {0}".format(job_name))
            #com_sbatch = 'sbatch {0}'.format(os.path.join(output_psr_utc_path,str(job_name)))
            #args_sbatch = shlex.split(com_sbatch)
            #proc_sbatch = subprocess.Popen(args_sbatch)
            #time.sleep(1)
            #print("{0} deployed.".format(job_name))

else:

    if args.utc:

        print ("Launching scintillation pipeline for {0}:{1}".format(args.psrname,args.utc))
        command = "python ScintPipeline.py -archivefile {0} -outdir {1}".format(cleaned_archive[0],output_psr_utc_path)
        args_pipe = shlex.split(command)
        proc_pipe = subprocess.call(args_pipe)

    else:
        for num,obs in enumerate(obs_list):
            utcname = os.path.split(obs)[-1]
            command = "python ScintPipeline.py -archivefile {0} -outdir {1}".format(cleaned_archive_list[num],output_psr_utc_list[num])
            print ("Launching scintillation pipeline for {0}:{1}".format(args.psrname,utcname))
            args_pipe = shlex.split(command)
            proc_pipe = subprocess.call(args_pipe)





