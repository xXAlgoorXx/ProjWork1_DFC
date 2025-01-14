import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from folderManagment import pathsToFolders as ptf
import seaborn as sns

# Print output distribution in form of a gaussian distribution.
# The fact the the distribution can be something other than gaussian.

def printOutputDistGauss(df_path,outputfolder):
    classes = ["In_Arch","In_Constr","Out_Constr","Out_Urban","Forest"]
    figname = df_path.split("/")[-1].split(".")[0]
    figname = figname.split("_")[1]
    df = pd.read_csv(df_path)
    mean_list = []
    var_list = []
    values_class = []
    for hailoClass in classes:
        mean_list.append(df.loc[:, hailoClass].mean())
        var_list.append(df.loc[:, hailoClass].std())
    
    
    
    # plot confusion matrix
    plt.figure(figsize=(8, 6))
    x = np.linspace(0,1, 1000)
    for i,(mean,std) in enumerate(zip(mean_list,var_list)):
        currentClass = classes[i]
        # plt.boxplot(df.loc[:, currentClass])
        plt.plot(x, stats.norm.pdf(x, mean, std),label = f"{classes[i]} Mean:{mean:.3} Std:{std:.3}")
        plt.fill(x,stats.norm.pdf(x, mean, std))
        
    for mean in mean_list:
        plt.axvline(mean,linestyle='dashed') 
    plt.title(figname + "Box")
    plt.xlabel('X')
    plt.ylabel('Prob')
    plt.legend()
    plt.grid(True)
    plt.savefig(outputfolder + "/" +  figname, dpi=600, bbox_inches='tight')
    plt.clf()


def printOutputDistHist(df_path,outputfolder):
    classes = ["In_Arch", "In_Constr", "Out_Constr", "Out_Urban", "Forest"]
    figname = df_path.split("/")[-1].split(".")[0]
    figname = figname.split("_")[1]
    df = pd.read_csv(df_path)
    bins = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 6))
    plt.hist(df.loc[:, "In_Arch"], bins, label = "In_Arch")
    plt.hist(df.loc[:, "In_Constr"], bins,label = "In_Constr")
    plt.hist(df.loc[:, "Out_Constr"], bins, label = "Out_Constr")
    plt.hist(df.loc[:, "Out_Urban"], bins, label = "Out_Urban")
    plt.hist(df.loc[:, "Forest"], bins, label = "Forest")
    
    plt.title(figname + "Box")
    plt.xlabel('X')
    plt.ylabel('Prob')
    plt.legend()
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.savefig(outputfolder + "/" +  figname, dpi=600, bbox_inches='tight')
    plt.show()
    plt.clf()

if __name__ == "__main__":
    
    df_path = str(ptf.evaluationFolder5Patch / "pred_TinyCLIP-ResNet-19M-Text-19M_5patches_5scentens.csv")
    outPath = "temp"
    
    printOutputDistHist(df_path, outPath)