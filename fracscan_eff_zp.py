import h5py
import numpy as np
import pandas as pd
from puma.utils import logger
from puma import Line2D, Line2DPlot
from puma.metrics import calc_eff
import glob

from taglist import *

def body(sgindex: int):
    ###############################################################################
    logger.info("This code produces fraction scan rejection vs rejection plots for multiple taggers and datasets")
    logger.info("Starting frac scan plotting process.....")
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
    ###############################################################################

    tag_prob=[]
    for i in range(0,len(file_list)):
        sub=[]
        for index,tagger in enumerate(tagger_list):
            sub.append(
                np.array(jets[i][[tagger+"_pu",tagger+"_pc",tagger+"_pb"]].values)
            )
        tag_prob.append(sub)

    ###############################################################################
    fc_values = np.linspace(0.0, 1.0, 101)
    sig_eff = [0.60,0.70,0.77,0.85]
    sigeffindex=sgindex
    def calc_effs(fc_value: float,tag_prob: float, findex: int, tagindex: int,sig_eff: float,sigindex: int):
        """Tagger efficiency for fixed working point

        Parameters
        ----------
        fc_value : float
            Value for the charm fraction used in discriminant calculation.

        Returns
        -------
        tuple
            Tuple of shape (, 3) containing (fc_value, ujets_eff, cjets_eff)
        """

        arr = tag_prob[findex][tagindex][:]
        disc = np.nan_to_num(arr[:, 2] / (fc_value * arr[:, 1] + (1 - fc_value) * arr[:, 0]))
        ujets_eff = calc_eff(disc[is_b[findex]], disc[is_l[findex]], sig_eff[sigindex])
        cjets_eff = calc_eff(disc[is_b[findex]], disc[is_c[findex]], sig_eff[sigindex])

        return [fc_value, ujets_eff, cjets_eff]

    ###############################################################################
    eff_results=[]
    for i in range(0,len(file_list)):
        sub=[]
        for tagger in range(0,len(tagger_list)):
            sub.append(
                np.array([calc_effs(fcval,tag_prob,i,tagger,sig_eff,sigeffindex) for fcval in fc_values])
            )
        eff_results.append(sub)

    x_values=[]
    y_values=[]
    for i in range(0,len(file_list)):
        sub1=[]
        sub2=[]
        for tagger in range(0,len(tagger_list)):
            sub1.append(
                np.array(eff_results[i][index][:, 2])
            )
            sub2.append(
                np.array(eff_results[i][index][:, 1])
                )
        x_values.append(sub1)
        y_values.append(sub2)


    # You can give several plotting options to the plot itself

    # Now init a fraction scan plot
    cpal=["r","b","g","k"]
    lstyle=["-","--","-.",":"]

    for index,tagger in enumerate(tagger_list):
        s1="Plotting FracScan curves for taggers"
        st = f"Internal\nZ'ext; WP{int(100*sig_eff[sigeffindex])}; $f_{{c}}={fcval}$\nPFlow jets;$\sqrt{{s}}$=13TeV\n$p_T \epsilon [250GeV,6TeV]$"
        frac_plot = Line2DPlot(
            atlas_first_tag="Simulation",
            atlas_second_tag= st,
            figsize=(6.5, 6),
            y_scale=1.4,
            )
        for i in range(0,len(file_list)):
            frac_plot.add(
                Line2D(
                    x_values=x_values[i][index][:],
                    y_values=y_values[i][index][:],
                    label=str(ptag[i])+" "+str(tagger),
                    colour=cpal[i],
                    linestyle=lstyle[i],
                )
            )
            """
            frac_plot.add(
                Line2D(
                    x_values=MARKER_X,
                    y_values=MARKER_Y,
                    colour="r",
                    marker="x",
                    label=rf"$f_c={eff_results[30, 0]}$",
                    markersize=15,
                    markeredgewidth=2,
                ),
                is_marker=True,
            )
            """
        # Adding labels
        frac_plot.ylabel = "Light-flavour jets efficiency"
        frac_plot.xlabel = "$c$-jets efficiency"

        # Draw and save the plot
        frac_plot.draw()
        frac_plot.savefig("FractionScanPlot_"+str(tagger)+str("_WP"+str(int(100*sig_eff[sigeffindex])))+"_eff_zp.png")
        del frac_plot
sigeff = [0,1,2,3]
for stuff in sigeff:
    body(stuff)
