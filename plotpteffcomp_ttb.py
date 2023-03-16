import random

import h5py
import numpy as np
import pandas as pd
from puma import Roc, RocPlot, VarVsEff, VarVsEffPlot
from puma.metrics import calc_rej
from puma.utils import logger
import glob

from taglist import *

logger.info("Starting pT vs EFF & Rejection plotting process ....")
logger.info("Reading h5 files")

###############################################################################
sig_eff = np.linspace(0.49, 1, 20)
###############################################################################
f_c_temp = []
#f_c_temp.append([0.070 for i in range(len(tagger_list_rnnip))])
f_c_temp.append([0.018 for i in range(len(tagger_list_dl1))])
f_c_temp.append([0.005 for i in range(len(tagger_list_dips))])
f_c_temp.append([0.050 for i in range(len(tagger_list_GN1))])

f_c = [element for sublist in f_c_temp for element in sublist]
#print(f_c)
###############################################################################
logger.info(
    "This code produces pt vs EFF and pt vs rejection comparison plots for multiple datasets"
)
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
###############################################################################
jets = []
is_l = []
is_c = []
is_b = []
n_jets_l = []
n_jets_c = []
n_jets_b = []
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

pt = []

for ptind in range(len(file_list)):
    pt.append(jets[ptind]["pt"] / 1e3)

for i in range(0, len(file_list)):
    is_l.append(jets[i]["HadronConeExclTruthLabelID"] == 0)
    is_c.append(jets[i]["HadronConeExclTruthLabelID"] == 4)
    is_b.append(jets[i]["HadronConeExclTruthLabelID"] == 5)

for i in range(0, len(file_list)):
    n_jets_l.append(sum(is_l[i]))
    n_jets_c.append(sum(is_c[i]))
    n_jets_b.append(sum(is_b[i]))
###############################################################################

disc = []
for i in range(0, len(file_list)):
    sub = []
    for index, tagger in enumerate(tagger_list):
        sub.append(
            np.apply_along_axis(
                lambda a:np.nan_to_num( np.log(a[2] / (f_c[index] * a[1] +
                                         (1 - f_c[index]) * a[0]))), 1,
                jets[i][[tagger + "_pu", tagger + "_pc",
                         tagger + "_pb"]].values))
    disc.append(sub)

###############################################################################
ujets_rej = []
cjets_rej = []
###############################################################################
for i in range(0, len(file_list)):
    sub1 = []
    sub2 = []
    for tagger in range(0, len(tagger_list)):
        sub1.append(
            calc_rej(disc[i][tagger][is_b[i]], disc[i][tagger][is_l[i]],
                     sig_eff))
        sub2.append(
            calc_rej(disc[i][tagger][is_b[i]], disc[i][tagger][is_c[i]],
                     sig_eff))
    ujets_rej.append(sub1)
    cjets_rej.append(sub2)

###############################################################################
wp=[0.60,0.70,0.77,0.85]
fixedornot=["True","False"]
fnamesuffix=["fixed","inclusive"]
for wpindex in range(0,len(wp)):
    for boolindex in range(0,len(fixedornot)):
        for tagger in range(0, len(tagger_list)):
            name1 = str(tagger_list[tagger])
            fcval = str(f_c[tagger])
            s1 = f"Plotting $p_T$ vs Eff curves for tagger {name1}"
            st = f'Internal\n$t\\bar{{t}}$; WP{int(100*wp[wpindex])}; $f_{{c}}={fcval}$\nPFlow jets;$\sqrt{{s}}$=13TeV\n$p_T \epsilon [20,250]GeV$'
            logger.info(s1)
            logger.info("Initializing plots as a function of pt.")
            plots_l = []
            plots_c = []
            plot_labels = [" 410470 (R21 Calibration)"," 410470 (R22  Calibration)"]
######################################################################
            for i in range(len(file_list)):
                plots_l.append(
                    VarVsEff(
                        x_var_sig=pt[i][is_b[i]],
                        disc_sig=disc[i][tagger][is_b[i]],
                        x_var_bkg=pt[i][is_l[i]],
                        disc_bkg=disc[i][tagger][is_l[i]],
                        bins=[
                            20, 30, 40, 60, 85, 110, 140, 200, 250,
                        ],
                        working_point=wp[wpindex],
                        disc_cut=None,
                        fixed_eff_bin=fixedornot[boolindex],
                        label=str(name1 + str(plot_labels[i])),
                    )
                )
######################################################################
            for i in range(len(file_list)):
                plots_c.append(
                    VarVsEff(
                        x_var_sig=pt[i][is_b[i]],
                        disc_sig=disc[i][tagger][is_b[i]],
                        x_var_bkg=pt[i][is_c[i]],
                        disc_bkg=disc[i][tagger][is_c[i]],
                        bins=[
                            20, 30, 40, 60, 85, 110, 140, 200, 250,
                        ],
                        working_point=wp[wpindex],
                        disc_cut=None,
                        fixed_eff_bin= fixedornot[boolindex],
                        label=str(name1 + str(plot_labels[i])),
                    )
                )
####################################################################
            logger.info(
                f"Plotting light bkg rejection for {fnamesuffix[boolindex]} efficiency as a function of pt."
            )
            # You can choose between different modes: "sig_eff", "bkg_eff", "sig_rej", "bkg_rej"
            plot_bkg_rej_l = VarVsEffPlot(
                n_ratio_panels=1,
                mode="bkg_rej",
                ylabel="Light jets rejection",
                xlabel=r"$p_{T}$ [GeV]",
                logy=False,
                atlas_first_tag="Simulation",
                atlas_second_tag=st,
                ymin_ratio=0.5,
                ymax_ratio=1.5,
                figsize=(6.5, 6),
                y_scale=1.4,
            )
######################################################################
            for ctr in range(len(plots_l)):
                plot_bkg_rej_l.add(plots_l[ctr], reference=not bool(ctr))
            plot_bkg_rej_l.draw()
            plot_bkg_rej_l.savefig(f"{name1}_pt_vs_l_rej_WP_{int(100*wp[wpindex])}_{fnamesuffix[boolindex]}_ttb.png")

######################################################################
            logger.info(
                f"Plotting c bkg rejection for {fnamesuffix[boolindex]} efficiency as a function of pt."
            )
            plot_bkg_rej_c = VarVsEffPlot(
                n_ratio_panels=1,
                mode="bkg_rej",
                ylabel="c jets rejection",
                xlabel=r"$p_{T}$ [GeV]",
                logy=False,
                atlas_first_tag="Simulation",
                atlas_second_tag=st,
                ymin_ratio=0.5,
                ymax_ratio=1.5,
                figsize=(6.5, 6),
                y_scale=1.4
            )
######################################################################
            for ctr in range(len(plots_c)):
                plot_bkg_rej_c.add(plots_c[ctr], reference=not bool(ctr))
            plot_bkg_rej_c.draw()
            plot_bkg_rej_c.savefig(f"{name1}_pt_vs_c_rej_WP_{int(100*wp[wpindex])}_{fnamesuffix[boolindex]}_ttb.png")
######################################################################
            plot_sig_eff = VarVsEffPlot(
                                        n_ratio_panels=1,
                                        mode="sig_eff",
                                        ylabel="b-jets efficiency",
                                        xlabel=r"$p_{T}$ [GeV]",
                                        logy=False,
                                        atlas_first_tag="Simulation",
                                        atlas_second_tag=st,
                                        ymin_ratio=0.5,
                                        ymax_ratio=1.5,
                                        figsize=(6.5, 6),
                                        y_scale=1.4,
                                        )

            for ctr in range(len(plots_c)):
                plot_sig_eff.add(plots_c[ctr], reference=not bool(ctr))

            # If you want to inverse the discriminant cut you can enable it via
            # plot_sig_eff.set_inverse_cut()
            plot_sig_eff.draw()
            # Drawing an hline indicating inclusive efficiency
            plot_sig_eff.draw_hline(wp[wpindex])
            plot_sig_eff.savefig(f"{name1}_pt_vs_b_eff_WP{int(100*wp[wpindex])}_{fnamesuffix[boolindex]}_ttb.png")
