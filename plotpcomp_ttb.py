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
from puma import Roc, RocPlot
from puma.metrics import calc_rej
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
logger.info("This code produces ROC comparison plots for multiple datasets")
logger.info("Starting ROC plotting process.....")
logger.info("Reading h5 files......")
###############################################################################
base_addr="/home/sammy/eos/user/s/sgoswami/public/pflowstudy/newsamples/"
base_list=[
        "user.pgadow.410470.e6337_s3681_r13144_p5169.tdd.EMPFlow.22_2_110.23-03-05_pflow_oldr21_output.h5/*.h5",
        "user.pgadow.410470.e6337_s3681_r13144_p5169.tdd.EMPFlow.22_2_110.23-03-05_pflow_newr22_output.h5/*.h5"
        ]
file_list= list(map(lambda m : base_addr + m, base_list))
print(len(file_list))

ptag=["410470 (R21 Calibration)","410470 (R22 Calibration)"]
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
    flist=glob.glob(file_list[i])
    print(flist)
    for fiter in range(0,len(flist)):
        with h5py.File(flist[fiter]) as h5file:
            dfjet=pd.DataFrame(h5file["jets"][:])
            if fiter==0:
                f1=pd.DataFrame(dfjet[ (dfjet['pt'] >= 20e3) & (dfjet['pt'] <= 250e3) & (np.abs(dfjet['eta']) <= 2.5)])
            else:
                f1.append(pd.DataFrame(dfjet[ (dfjet['pt'] >= 20e3) & (dfjet['pt'] <= 250e3) & (np.abs(dfjet['eta']) <= 2.5)]), ignore_index=True)
    jets.append(f1)

for i in range(0,len(file_list)):
    is_l.append(jets[i]["HadronConeExclTruthLabelID"]==0)
    is_c.append(jets[i]["HadronConeExclTruthLabelID"]==4)
    is_b.append(jets[i]["HadronConeExclTruthLabelID"]==5)

for i in range(0,len(file_list)):
    n_jets_l.append(sum(is_l[i]))
    n_jets_c.append(sum(is_c[i]))
    n_jets_b.append(sum(is_b[i]))
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
ujets_rej=[]
cjets_rej=[]
###############################################################################
for i in range(0,len(file_list)):
    sub1=[]
    sub2=[]
    for tagger in range(0,len(tagger_list)):
        sub1.append(calc_rej(
            disc[i][tagger][is_b[i]],
            disc[i][tagger][is_l[i]],
            sig_eff
            )
        )
        sub2.append(calc_rej(
            disc[i][tagger][is_b[i]],
            disc[i][tagger][is_c[i]],
            sig_eff
            )
        )
    ujets_rej.append(sub1)
    cjets_rej.append(sub2)

###############################################################################

for tagger in range(0,len(tagger_list)):
    refr=False
    name1=str(tagger_list[tagger])
    fcval=str(f_c[tagger])
    s1=f"Plotting ROC curves for tagger {name1}"
    st=f'Internal\n$t\\bar{{t}}$; $f_{{c}}={fcval}$\nPFlow jets;$\sqrt{{s}}$=13TeV\n$p_T \epsilon [20,250]GeV$'
    logger.info(s1)
    plot_roc = RocPlot(
            n_ratio_panels=2,
            ylabel="background rejection",
            xlabel="b-jets efficiency",
            atlas_first_tag='Simulation',
            atlas_second_tag= st,
            figsize=(6.5, 6),
            y_scale=1.4,
            )
    for i in range(0,len(file_list)):
        refr=True if i==0 else refr
        plot_roc.add_roc(
            Roc(
                sig_eff,
                ujets_rej[i][tagger],
                n_test= n_jets_l[i],
                rej_class="ujets",
                signal_class="bjets",
                label=str(name1+" "+str(ptag[i])),
                ),
            reference=refr,
            )
        refr=False
        if i==len(file_list)-1:
            plot_roc.set_ratio_class(1, "ujets")

    for i in range(0,len(file_list)):
        refr=True if i==0 else refr
        plot_roc.add_roc(
            Roc(
                sig_eff,
                cjets_rej[i][tagger],
                n_test= n_jets_c[i],
                rej_class="cjets",
                signal_class="bjets",
                label=str(name1+" "+str(ptag[i])),
                ),
            reference=refr,
            )
        refr=False
        if i==len(file_list)-1:
            plot_roc.set_ratio_class(2, "cjets")
    plot_roc.draw()
    plot_roc.savefig("ROC_"+str(tagger_list[tagger])+"_ttb.png")
    del plot_roc
