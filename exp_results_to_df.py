import pandas as pd
import numpy as np

import argparse
import os
import time

def main_df_hospital_overtime(dir_path):

    text_files=[fname for fname in os.listdir(dir_path) if fname.startswith('result_hospital-overtime-style') and fname.endswith(".txt")]

    columns=[]
    for target in targets:
        for model in models:
            for representation in representations:
                for measurement in measures:
                    columns.append((target,model, representation, measurement))

    ind=[(hosp, yr, mnth) for hosp in hospitals for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement'))
    df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for representation in representations:
            for text_file in sorted(text_files):
                if (target in text_file) and (representation in text_file):
                    with open(os.path.join(dir_path, text_file), 'rb') as f:
                        all_lines=f.readlines()
                            
                    for modeltype in models:
                        model_lines=[line.decode() for line in all_lines if ", "+modeltype.upper()+"," in line.decode()]
                        for measurement in measures:
                            lines=[line for line in model_lines if ", "+measurement+"," in line]
                            for line in lines:
                                for hosp in hospitals:
                                    if (hosp in line):
                                        for year in year_range:
                                            if ("year," in line) and (str(year) in line.split("year,")[1].split(',')[0]):
                                                for month in month_intervals:
                                                    if (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)])):
                                                        try:
                                                            value=float(line.split(",")[-1])
                                                        except:
                                                            print(line)
                                                            print(hosp, year, month, target, modeltype, representation, measurement)
                                                            raise
                                                        df.loc[(hosp, int(year), int(month)), 
                                                                (target, modeltype, representation, measurement)]=value
                    
    return df.apply(pd.to_numeric, errors='coerce')

def main_df_single_site(dir_path):

    text_files  = [fname for fname in os.listdir(dir_path) if fname.startswith('result_single-site-style') and fname.endswith(".txt")]
    sites=["UPMCPUH", "UPMCSHY", "CTICU", "MICU"]

    columns=[]
    for target in targets:
        for model in models:
            for representation in representations:
                for measurement in measures:
                    columns.append((target,model, representation, measurement))
                        
    index=[(s, y, m) for s in sites for y in year_range for m in month_intervals]
    ind=pd.MultiIndex.from_tuples(index, names=('site', 'year', 'month_interval'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement'))
    df=pd.DataFrame( index=ind, columns=cols) # 4 columns, 2 indices

    for target in targets:
        print(target)
        for modeltype in models:
            for representation in representations:
                for site in sites:
                    for text_file in sorted(text_files):
                        if target not in text_file:
                            continue
                        if ("_"+modeltype.upper()+"_" in text_file) and ("_"+representation+"_" in text_file) and ("_"+site+"_" in text_file):
                            with open(os.path.join(dir_path, text_file), 'rb') as f:
                                all_lines=f.readlines()                                                                
                            for measurement in measures:
                                lines=[line.decode() for line in all_lines if ", "+measurement+"," in line.decode()]
                                for line in lines:
                                    for year in year_range:
                                        if ("year," in line) and (str(year) in line.split("year,")[1].split(",")[0]):
                                            for month in month_intervals:
                                                # print(month)
                                                if (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)])):
                                                    value=float(line.split(",")[-1])
                                                    # print(year, month, target, modeltype, representation, measurement, value)
                                                    df.loc[(site, int(year), int(month)), (target, modeltype, representation, measurement)]=value


    df=df.apply(pd.to_numeric, errors='coerce')
    return df

def main_df_overall_overtime(dir_path):

    text_files=[fname for fname in os.listdir(dir_path) if fname.startswith('result_overall-overtime-style') and fname.endswith(".txt")]

    columns=[]
    for target in targets:
        for model in models:
            for representation in representations:
                for measurement in measures:
                    columns.append((target,model, representation, measurement))

    ind=[(yr, mnth) for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement'))
    df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for representation in representations:
            for modeltype in models:
                for text_file in sorted(text_files):
                    if (target in text_file) and ("_"+representation+"_" in text_file) and ("_"+modeltype.upper()+"_" in text_file):
                        with open(os.path.join(dir_path, text_file), 'rb') as f:
                            all_lines=f.readlines()
                        
                        print(target, representation, modeltype)
                        
                        for measurement in measures:
                            lines=[line.decode() for line in all_lines if ", "+measurement+"," in line.decode()]
                            for line in lines:
                                for year in year_range:
                                    if ("year," in line) and (str(year) in line.split("year,")[1].split(',')[0]):
                                        for month in month_intervals:
                                            if (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)])):
                                                try:
                                                    value=float(line.split(",")[-1])
                                                except:
                                                    print(line)
                                                    print(year, month, target, modeltype, representation, measurement)
                                                    raise
                                                df.loc[(int(year), int(month)), (target, modeltype, representation, measurement)]=value
            
    return df.apply(pd.to_numeric, errors='coerce')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='summarize the results of experiments into dataframes')
    parser.add_argument('--dir_path', type=str, default="", help="full path to results directory")
    parser.add_argument('--out_dir', type=str, default="", help="full path to output directory")
    parser.add_argument('--train_type', type=str, default=None, help="['overall_overtime', 'hospital_wise', 'icu_type', 'single_site', 'hospital_overtime']")

    args = parser.parse_args()

    idx=pd.IndexSlice

    site_info = pd.read_pickle("E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_0/ITEMID_REP/site_info.pkl")
    hospitals = sorted(site_info["hospital"].unique().tolist())
    models = ['rf', 'lr', 'rbf-svm', 'nb']
    targets = ['mort_icu', 'los_3']
    representations = ['raw', 'pca']
    measures = ['AUC', 'APR', 'Acc', 'F1', 'ECE', 'MCE', 'O_E']
    # measures = main_measures + [m+'_base' for m in main_measures] + [m+'_diff' for m in main_measures]
    year_range = np.arange(2011, 2015)
    month_step = 2
    month_intervals = np.arange(month_step, 13, month_step)

    dir_path=args.dir_path
    out_dir=args.out_dir
    if out_dir=="":
        out_dir=dir_path
    out_filename="results_df_"+args.train_type

    if args.train_type=="hospital_overtime":
        df=main_df_hospital_overtime(dir_path)
    elif args.train_type=="single_site":
        df=main_df_single_site(dir_path)
    elif args.train_type=="overall_overtime":
        df=main_df_overall_overtime(dir_path)
    else:
        raise "method for "+args.train_type+" not found!"

    df.to_pickle(os.path.join(out_dir, out_filename+".pkl"))
    df.to_csv(os.path.join(out_dir, out_filename+".csv"))