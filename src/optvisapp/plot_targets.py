import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
import pandas as pd
import argparse

def plot_targets(targets,show_xmm,show_er):
    tab = pd.read_csv(targets,header=None)
    fulltab = pd.read_csv("source_catalogs/4XMM_targets.csv")

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection="mollweide")
    ras = tab[2].to_numpy()
    ras[np.where(ras>180)] -= 360
    plt.scatter(ras*u.deg.to('rad'),tab[3]*u.deg.to('rad'),marker='s',s=50,zorder=2)

    if show_xmm:
        fulltab_xmm = pd.read_csv("source_catalogs/4XMM_targets.csv")
        ras_xmm = fulltab_xmm['ra'].to_numpy()
        ras_xmm[np.where(ras_xmm>180)] -= 360
        plt.scatter(ras_xmm*u.deg.to('rad'),fulltab_xmm['dec']*u.deg.to('rad'),
                    marker='.',s=50,color='grey',zorder=1,alpha=0.2)
    if show_er:
        fulltab_er = pd.read_csv("source_catalogs/eROSITA_targets.csv")
        ras_er = fulltab_er['ra'].to_numpy()
        ras_er[np.where(ras_er>180)] -= 360
        plt.scatter(ras_er*u.deg.to('rad'),fulltab_er['dec']*u.deg.to('rad'),
                    marker='.',s=50,color='green',zorder=1,alpha=0.02)


    plt.grid()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plots NICER target files.")
    parser.add_argument("-f", help='Target file to plot', type=str, required=True)
    parser.add_argument("-x", help="Show 4XMM survey points",
                        type=bool, default=False,
                        action=argparse.BooleanOptionalAction)

    parser.add_argument("-e", help="Show eROSITA survey points",
                        type=bool, default=False,
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    plot_targets(args.f, args.x, args.e)


if __name__ == '__main__':
    main()
