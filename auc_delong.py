import numpy as np
import pandas as pd
from util.utils import auc_delong_test
import os
from time import time

def get_values_from_line(line):
    if len(line)==0:
        values=[]
    else:
        values=line.split("<")[2].split(">")[0]
        if values=='':
            values=[]
        else:
            values=[float(i) for i in values.split(",")]
    return values

def auc_delong_hospital_overtime(dir_path, out_filename):

    text_files=[fname for fname in os.listdir(dir_path) if fname.startswith('result_') and fname.endswith(".txt")]

    base_hospital='UPMCPUH'
    base_year=2011
    base_month=2
    stats=['N_base','auc_base','ci_base','N','auc','ci','pval']

    columns=[]
    for target in targets:
        for representation in representations:
            for model in models:
                for stat in stats:
                    columns.append((target, representation, model, stat))

    ind=[(hosp, yr, mnth) for hosp in hospitals for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'representation', 'model', 'stat'))
    df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for representation in representations:
            for text_file in sorted(text_files):
                if target not in text_file:
                    continue
                if ("_"+representation+"_" in text_file):
                    with open(os.path.join(dir_path, text_file), 'r') as f:
                        all_lines=f.readlines()

                    for modeltype in models:
                        model_lines=[line for line in all_lines if (", "+modeltype.upper()+"," in line)]

                        base_month_str='<'+str(base_month-1)+","+str(base_month)+">"
                        base_lines=[line for line in model_lines if (base_hospital in line.split(",")[3]) and
                                (str(base_year) in line.split(",")[5]) and (base_month_str in line)]

                        if len(base_lines)==0:
                            continue

                        base_labels=[l for l in base_lines if 'label,' in l][0]
                        base_labels=get_values_from_line(base_labels)

                        if (base_labels is None) or (np.unique(base_labels).size < 2):
                            continue

                        base_probs=[l for l in base_lines if 'y_pred_prob,' in l][0]
                        base_probs=get_values_from_line(base_probs)
                        
                        for hosp in hospitals:
                            for year in year_range:
                                for month in month_intervals:
                                    print(target, representation, modeltype, hosp, year, month)

                                    month_str='<'+str(month-1)+","+str(month)+">"
                                    lines=[line for line in model_lines if (hosp in line) and 
                                        (str(year) in line.split(",")[5]) and (month_str in line)]
                                    if len(lines)==0:
                                        continue
                                    
                                    labels=[l for l in lines if 'label,' in l][0]
                                    labels=get_values_from_line(labels)
                                    probs=[l for l in lines if 'y_pred_prob,' in l][0]
                                    probs=get_values_from_line(probs)

                                    if (labels is None) or (np.unique(labels).size < 2):
                                            continue

                                    auc_base,ci_base,auc,ci,pval = auc_delong_test(base_labels, base_probs,labels,probs)
                                    N_base=len(base_labels)
                                    N=len(labels)
                                    df.loc[(hosp,year,month), idx[target, representation, modeltype, ['N_base','auc_base','ci_base','N','auc','ci','pval']]]=(N_base,auc_base,ci_base,N,auc,ci,pval)
                        
                            df.to_csv(os.path.join(dir_path, out_filename))
                            df.to_pickle(os.path.join(dir_path, out_filename.split(".")[0]+'.pkl'))
    return df


if __name__ == "__main__":

    print("start")

    targets = ['mort_icu', 'los_3']
    representations = ['raw', 'pca']
    models=['rf','lr','nb','rbf-svm']

    hospitals = ['UPMCBED','UPMCEAS','UPMCHAM','UPMCHZN','UPMCMCK','UPMCMER','UPMCMWH','UPMCNOR','UPMCPAS','UPMCPUH','UPMCSHY','UPMCSMH']
    year_range = np.arange(2011, 2015)
    month_step = 2
    month_intervals = np.arange(month_step, 13, month_step)

    dir_path="../../output/HIDENIC_overtime_analysis/hospital_overtime/"
    out_filename="auc_delong_hospital_overtime.csv"

    idx=pd.IndexSlice

    t0=time()
    df=auc_delong_hospital_overtime(dir_path, out_filename)

    t1=time()
    print("Done.")
    print("Total time={:0.2f} seconds".format(t1-t0)) #Total time=3845.82 seconds
                