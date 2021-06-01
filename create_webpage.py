#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys,glob
import numpy as np
import pandas as pd
import argparse
import shutil

parser = argparse.ArgumentParser(description="Create TPA scintillation webpage")
parser.add_argument("-path", dest="path", help="Path containing the TPA scintillation pipeline output",required=True)
parser.add_argument("-psr", dest="psrname", help="Update the pngs for the specified pulsar")
parser.add_argument("-utc", dest="utc", help="Update the pngs for the specific utc")
parser.add_argument("-list", dest="list", help="Use the pulsar-utc list")
parser.add_argument("-class", dest="classes", help="Classification list",required=True)
args = parser.parse_args()


scint_path = args.path

summary_file = os.path.join(scint_path,"usersummary.txt")
user_summary = pd.read_csv(summary_file,names=["psrname","comments"],delimiter=":",dtype=str)

if args.list:
    psr_utc_df = pd.read_csv(str(args.list), delimiter=" ", header=None, dtype=str, comment="#",names=["psrname","utc"])

if args.utc:
    data_path = os.path.join(scint_path,"{0}/{1}".format(args.psr,args.utc))
    png = glob.glob(os.path.join(data_path,"*panorama*png"))[0]
    png_list = png

else:
    if args.list:
        png_list = []
        for index, row in psr_utc_df.iterrows():
            data_path = glob.glob(os.path.join(os.path.join(scint_path,row["psrname"]),"{0}*".format(row["utc"])))[0]
            png = glob.glob(os.path.join(data_path,"*panorama*.png"))
            if len(png) > 0:
                png_list.append(png[0])
    else:
        png_list = sorted(glob.glob(os.path.join(scint_path,"{0}/2*/*panorama*png".format(args.psrname))))


if not os.path.exists(os.path.join(scint_path,"png_summary")):
    os.makedirs(os.path.join(scint_path,"png_summary"))

png_summary_dir = os.path.join(scint_path,"png_summary")
for image in png_list:
    source = image
    imagename = os.path.split(image)[-1]
    destination = os.path.join(png_summary_dir,imagename)
    if not os.path.exists(destination):
        shutil.copy(source,destination)


images = sorted(glob.glob(os.path.join(png_summary_dir,"*png")))

total = len(images)


#Reading classification text file
if args.classes:
    psr_classes = pd.read_csv(str(args.classes), delimiter=" ", header=None, dtype=str, comment="#",names=["psrname","class"])

    class0_pulsars = psr_classes.loc[psr_classes['class'] == "0"]
    class1_pulsars = psr_classes.loc[psr_classes['class'] == "1"]
    class2_pulsars = psr_classes.loc[psr_classes['class'] == "2"]
    class3_pulsars = psr_classes.loc[psr_classes['class'] == "3"]
    class4_pulsars = psr_classes.loc[psr_classes['class'] == "4"]
    class5_pulsars = psr_classes.loc[psr_classes['class'] == "5"]
    class6_pulsars = psr_classes.loc[psr_classes['class'] == "6"]


with open(os.path.join(scint_path,"scint.html"),"w") as f:

    f.write(""" <html>

    <head>
    <title>TPA Scintillation</title>
    <h1> Scintillation properties for the {0} TPA pulsars </h1>
    
    <p style="color:red; style="font-size:100px;"> Each pulsar has been classified into one/many classes based on their observed scintillation properties. <br/>
    These classes are defined below. Please click on the respective links to view the relevant plots. <br/> </p>
    
    <p style="color:black; style="font-size:80px;"> In each of these webpages the plots are arranged from left to right as described below: <br/>
    <br/>
    ->The first two panels show the Wiener filtered dynamic spectrum, and the corresponding secondary spectrum across the whole band 
    (the filtering includes inpainting the masked channels, and removing the intrinsic variations). <br/>
    <br/>
    ->The third panel, shows the dynamic and secondary spectra restricted to 970–1075 MHz (top) and 1290–1530 MHz (bottom), the two cleanest bands.  <br/>
    <br/>
    ->The fourth panels are 2D ACFs in these bands, fit with a model of the 2D ACF for a Kolmogorov spectrum along with phase gradient (fit shown in contours, and in blue lines), 
    and 1D cuts through the two axes (fit shown in orange).  <br/>
    <br/>
    ->The fifth panel show cuts through the 2D ACF, and their fits, in 8 equally spaced subbands (high to low - top to bottom).  <br/>
    <br/>
    ->The top-right plot in the last panel (when visible), shows the summed power as a function of arc curvature, and the best fit arc curvature. The bottom-right plots show the fit values of 
    t_scint (scintillation timescale) and nu_scint (scintillation bandwidth) across frequency. <br/> </p>

    <p style="color:blue; style="font-size:50px;">Contact: Robert Main (MPIfR) & Aditya Parthasarathy (MPIfR) </p>

    <h4>TPA PIs: Simon Johnston & Aris Karastergiou </h3>
    <body>
    <table frame=box border=3  cellpadding=6 >\n """.format(str(total)))


    f.write("<tr><th><font size=+1>CLASS<br></font></th><th><font size=+1>DESCRIPTION<br></font></th><th><font size=+1>#PULSARS</font></th></tr> \n")
    
    f.write("""<tr><td><a href=Class1.html><font size=+1>CLASS 1</font></a></td><td align=center><font size=+1>Observations with arc detections or constraints on 
    arc measurements.</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class1_pulsars)))
    
    f.write("""<tr><td><a href=Class2.html><font size=+1>CLASS 2</font></a></td><td align=center><font size=+1>Scintles not resolved in time</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class2_pulsars)))
    
    f.write("""<tr><td><a href=Class3.html><font size=+1>CLASS 3</font></a></td><td align=center><font size=+1>Reprocessing required with different filtering parameters</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class3_pulsars)))
    
    f.write("""<tr><td><a href=Class4.html><font size=+1>CLASS 4</font></a></td><td align=center><font size=+1>Combination of classes (1) and (3)</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class4_pulsars)))
    
    f.write("""<tr><td><a href=Class5.html><font size=+1>CLASS 5</font></a></td><td align=center><font size=+1>Combination of classes (2) and (3)</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class5_pulsars)))
    
    f.write("""<tr><td><a href=Class6.html><font size=+1>CLASS 6</font></a></td><td align=center><font size=+1>Classification unsure</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class6_pulsars)))
    
    f.write("""<tr><td><a href=Class0.html><font size=+1>CLASS 0</font></a></td><td align=center><font size=+1>Not usable for scintillation studies</font></td><td align=center><font size=+1>{0}</font></td>""".format(len(class0_pulsars)))

    f.write(""" </table>
            <br>
            </body>
            </head>
            </html>""")

f.close()

class0_file = open(os.path.join(scint_path,"Class0.html"),"w")
class1_file = open(os.path.join(scint_path,"Class1.html"),"w")
class2_file = open(os.path.join(scint_path,"Class2.html"),"w")
class3_file = open(os.path.join(scint_path,"Class3.html"),"w")
class4_file = open(os.path.join(scint_path,"Class4.html"),"w")
class5_file = open(os.path.join(scint_path,"Class5.html"),"w")
class6_file = open(os.path.join(scint_path,"Class6.html"),"w")


class0_file.write("<title> Class 0 pulsars </title> <h1> Mostly not usable for scintillation studies </h1> <table frame=box border=3 cellpadding=6 > \n")
class1_file.write("<title> Class 1 pulsars </title> <h1> Pulsars with resolved scintles in time - can be used to detect or constrain arc properties </h1> <table frame=box border=3 cellpadding=6 > \n")
class2_file.write("<title> Class 2 pulsars </title> <h1> Pulsars with unresolved scintles </h1> <table frame=box border=3 cellpadding=6 > \n")
class3_file.write("<title> Class 3 pulsars </title> <h1> Pulsars which require tweaking of the Wiener filter </h1> <table frame=box border=3 cellpadding=6 > \n")
class4_file.write("<title> Class 4 pulsars </title> <h1> Pulsars that could belong to Class 1 but require slight tweaking </h1> <table frame=box border=3 cellpadding=6 > \n")
class5_file.write("<title> Class 5 pulsars </title> <h1> Pulsars that could belong to Class 2 but require slight tweaking </h1>  <table frame=box border=3 cellpadding=6 > \n")
class6_file.write("<title> Class 6 pulsars </title> <h1> Pulsars with unclear classification </h1> <table frame=box border=3 cellpadding=6 > \n")


for index, row in class0_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class0_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class0_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class0_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class0_file.close()




for index, row in class1_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class1_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class1_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class1_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class1_file.close()




for index, row in class2_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class2_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class2_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class2_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class2_file.close()




for index, row in class3_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class3_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class3_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class3_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class3_file.close()





for index, row in class4_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class4_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class4_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class4_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class4_file.close()






for index, row in class5_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class5_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class5_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class5_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class5_file.close()





for index, row in class6_pulsars.iterrows():
    psrname = row["psrname"]
    image = [s for s in images if psrname in s][0] #CURRENTLY ASSUMES ONE OBS PER PULSAR
    obsname = os.path.split(image)[-1].split(".filtered")[0].split("_")[1]
    image_name = os.path.split(image)[-1]
    
    comment = user_summary.loc[user_summary['psrname'] == psrname]["comments"].tolist()
    if not len(comment) == 0:
        comment = comment[0]
        class6_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))        
    else:
        comment = "None"
        class6_file.write('<tr><td align=left><font size=+2>{0} - {1} - Comments: {2}<br></font></td></tr> \n'.format(psrname,obsname,comment))           
          
    class6_file.write('<tr><td><img src="png_summary/{0}" width="2600" height="1000" </td></tr> \n'.format(image_name))            

class6_file.close()


