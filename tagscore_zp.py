#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:52:21 2022

@author: sgoswami

This Python3 script produces ROC comparison plots for multiple p tags.
Must supply another python file called taglist.py with the tagger list.
"""

import h5py
import numpy as np
import pandas as pd
import numpy as np
from puma import Histogram, HistogramPlot
from puma.utils import get_good_linestyles, global_config
from puma.utils import logger
import glob

from taglist import *

###############################################################################
sig_eff = np.linspace(0.49, 1, 20)
###############################################################################
f_c_temp=[]
#f_c_temp.append([0.070 for i in range(len(tagger_list_rnnip))])
f_c_temp.append([0.018 for i in range(len(tagger_list_dl1))])
f_c_temp.append([0.005 for i in range(len(tagger_list_dips))])
f_c_temp.append([0.050 for i in range(len(tagger_list_GN1))])

f_c = [element for sublist in f_c_temp for element in sublist]
###############################################################################
logger.info("This code produces Tagger Score comparison plots for multiple datasets")
logger.info("Starting ROC plotting process.....")
logger.info("Reading h5 files......")
###############################################################################
base_addr="/home/sammy/eos/user/s/sgoswami/public/pflowstudy/newsamples/"
base_list=[
         "user.pgadow.800030.e7954_s3681_r13144_p5169.tdd.EMPFlow.22_2_110.23-03-05_pflow_oldr21_output.h5/*.h5",
         "user.pgadow.800030.e7954_s3681_r13144_p5169.tdd.EMPFlow.22_2_110.23-03-05_pflow_newr22_output.h5/*.h5"
        ]
file_list= list(map(lambda m : base_addr + m, base_list))
print(len(file_list))

ptag=["800030 (R21 Calibration)","800030 (R22 Calibration)"]
###############################################################################
jets=[]
is_l=[]
is_c=[]
is_b=[]
n_jets_l=[]
n_jets_c=[]
n_jets_b=[]
###############################################################################
for i in range(0,len(file_list)):
    f1=[]
    f2=[]
    flist=glob.glob(file_list[i])
    print(flist)
    for fiter in range(0,len(flist)):
        with h5py.File(flist[fiter]) as h5file:
            dfjet=pd.DataFrame(h5file["jets"][:])
            if fiter==0:
                f1=pd.DataFrame(dfjet[ (dfjet['pt'] >= 250e3) & (np.abs(dfjet['eta']) <= 2.5)])
            else:
                f2=pd.DataFrame(dfjet[ (dfjet['pt'] >= 250e3) & (np.abs(dfjet['eta']) <= 2.5)])
    ftemp=pd.concat([f1,f2],ignore_index=True)
    jets.append(ftemp)

for i in range(0,len(file_list)):
    is_l.append(jets[i]["HadronConeExclTruthLabelID"]==0)
    is_c.append(jets[i]["HadronConeExclTruthLabelID"]==4)
    is_b.append(jets[i]["HadronConeExclTruthLabelID"]==5)

for i in range(0,len(file_list)):
    n_jets_l.append(sum(is_l[i]))
    n_jets_c.append(sum(is_c[i]))
    n_jets_b.append(sum(is_b[i]))
flav_cat = global_config["flavour_categories"]
linestyles = get_good_linestyles()[:2]
###############################################################################

disc=[]
for i in range(0,len(file_list)):
    sub=[]
    for index,tagger in enumerate(tagger_list):
        sub.append(np.apply_along_axis(
            lambda a: np.nan_to_num(np.log(a[2] / ( f_c[index]*a[1]+(1-f_c[index])* a[0]))),
            1,
            jets[i][[tagger+"_pu",tagger+"_pc",tagger+"_pb"]].values
            )
        )
    disc.append(sub)

###############################################################################

for tagger in range(0,len(tagger_list)):
    refr=False
    name1=str(tagger_list[tagger])
    fcval=str(f_c[tagger])
    s1=f"Plotting ROC curves for tagger {name1}"
    st = f"Internal\nZ'ext; $f_{{c}}={fcval}$\nPFlow jets;$\sqrt{{s}}$=13TeV\n$p_T \epsilon [250GeV,6TeV]$"
    logger.info(s1)
    lstyle=["-","--","-.",":"]
    plot_histo = HistogramPlot(
                                n_ratio_panels=1,
                                ylabel="Normalised number of jets",
                                ylabel_ratio=["Ratio"],
                                xlabel="$b$-jet discriminant",
                                logy=False,
                                leg_ncol=1,
                                figsize=(8, 6),
                                bins=np.linspace(-10, 20, 100),
                                y_scale=1.5,
                                ymax_ratio=[1.5],
                                ymin_ratio=[0.5],
                                atlas_first_tag="Simulation",
                                atlas_second_tag=st,
                        )
    for i in range(0,len(file_list)):
        refr=True if i==0 else False
        plot_histo.add(
            Histogram(
                disc[i][tagger][is_l[i]],
                label="Light jets" if i == 0 else None,
                colour=flav_cat["ujets"]["colour"],
                ratio_group="ujets",
                linestyle=lstyle[i],
                ),
            reference=refr,
            )
        plot_histo.add(
            Histogram(
                disc[i][tagger][is_b[i]],
                label="$b$-jets" if i == 0 else None,
                colour=flav_cat["cjets"]["colour"],
                ratio_group="bjets",
                linestyle=lstyle[i],
            ),
            reference=refr,
        )
        plot_histo.add(
            Histogram(
                disc[i][tagger][is_c[i]],
                label="$c$-jets" if i == 0 else None,
                colour=flav_cat["bjets"]["colour"],
                ratio_group="cjets",
                linestyle=lstyle[i],
            ),
            reference=refr,
        )
    plot_histo.draw()
    plot_histo.make_linestyle_legend(
        linestyles=linestyles, labels=ptag, bbox_to_anchor=(0.55, 1)
    )
    plot_histo.savefig("Score_"+str(tagger_list[tagger])+"_zp.png")
    del plot_histo
