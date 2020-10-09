# Imports
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from skmultiflow.drift_detection import *
from hyppo.ksample import KSample

import argparse
import os
import time

static_data=None

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

def load_static_data(data_dir=""):
    global static_data
    if data_dir=="":
        data_dir="E:/Data/HIDENIC_EXTRACT_OUTPUT_DIR/POP_SIZE_0/ITEMID_REP/"
    static_data=pd.read_csv(os.path.join(data_dir, "static_data.csv"))
    static_data["intime"]=pd.to_datetime(static_data["intime"])
    static_data=static_data.set_index("subject_id")
    return

def to_error_stream(label, probs):
    label=np.array(label)
    probs=np.array(probs)
    preds=(probs > 0.5).astype(int)
    ## XNOR label and preds
    error_stream=(~np.logical_xor(label, preds)).astype(int)
    return error_stream

def get_stream_drifts(method, base_error_stream, cur_error_stream, **kwargs):

    method_names={
        "adwin": ADWIN,
        "ddm": DDM,
        "eddm": EDDM,
        "hddm_a": HDDM_A,
        "hddm_w": HDDM_W,
        "kswin": KSWIN,
        "pagehinkley": PageHinkley
    }
    method=method_names[method.lower()]

    dd_method=method(**kwargs)

    all_warn_idx=[]; all_drift_idx=[]; base_warn_idx=[]; base_drift_idx=[]; cur_warn_idx=[]; cur_drift_idx=[]

    stream=np.concatenate((base_error_stream, cur_error_stream), axis=0)
    n=len(stream); b=len(base_error_stream)

    for i,e in enumerate(stream):
        dd_method.add_element(e)
        if dd_method.detected_warning_zone():
            all_warn_idx.append(i)
            if(i < b):
                base_warn_idx.append(i)
            else:
                cur_warn_idx.append(i-b)
        if dd_method.detected_change():
            all_drift_idx.append(i)
            if(i < b):
                base_drift_idx.append(i)
            else:
                cur_drift_idx.append(i-b)
 
    return all_warn_idx, all_drift_idx, base_warn_idx, base_drift_idx, cur_warn_idx, cur_drift_idx 

def get_sorted_labels(labels, y_pred_probs, subject_ids):
    in_times=get_intimes(subject_ids)
    df=pd.DataFrame({'subject_id': subject_ids, 'intime':in_times, 'label':labels, 'prob':y_pred_probs}).sort_values("intime", ascending=True)
    return df.label.values, df.prob.values

def get_intimes(subject_ids):
    global static_data
    in_times=static_data.loc[subject_ids, "intime"]
    return in_times.values

def main_class_dist_change_detection(dir_path, out_dir):

    text_files=[fname for fname in os.listdir(dir_path) if fname.startswith('result_') and fname.endswith(".txt")]

    base_hospital='UPMCPUH'
    base_year=2011
    base_month=2

    columns=[]
    for target in targets:
        for rep in representations:
            for stat in ['base_prop', 'prop', 'fisher_pval']:
                columns.append((target, rep, stat))

    ind=[(hosp, yr, mnth) for hosp in hospitals for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'representation', 'stat'))
    df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for representation in representations:
            for text_file in sorted(text_files):
                if target not in text_file:
                    continue
                if (representation in text_file):
                    print(target, representation)

                    with open(os.path.join(dir_path, text_file), 'rb') as f:
                        all_lines=f.readlines()

                        ## read labels only from one model (labels are the same for all models)
                        label_lines=[line.decode() for line in all_lines if ('label,' in line.decode()) and ('RF,' in line.decode())]

                        base_line=[line for line in label_lines if (base_hospital in line.split(",")[3]) and
                            (str(base_year) in line.split(",")[5]) and 
                            (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(base_month-month_step+1, base_month+1, 1)]))][0]
                        base_label=get_values_from_line(base_line)
                        if len(base_label)==0:
                            continue

                        for hosp in hospitals:
                            for year in year_range:
                                for month in month_intervals:
                                    lines=[line for line in label_lines if (hosp in line) and 
                                        (str(year) in line.split(",")[5]) and
                                        (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)]))
                                        ]

                                    label=None
                                    for line in lines:
                                        try:
                                            if ('label,' in line):
                                                label=get_values_from_line(line)
                                        except:
                                            print(line)
                                            print(hosp, year, month, target, representation)
                                            raise
                                    if (label is None) or (np.unique(label).size < 2):
                                        continue
                                    base_label=pd.Series(base_label)
                                    label=pd.Series(label)
                                    table=pd.concat([base_label.value_counts(), label.value_counts()], axis=1, sort=False).values
                                    _, pval=fisher_exact(table, alternative='two-sided')
                                    print(target, representation, hosp, year, month)
                                    print("base_prop={:0.2f}, prop={:0.2f}, pval={:0.3f}".format(base_label.mean(), label.mean(), pval))
                                    
                                    df.loc[(hosp, year, month), idx[target, representation, 'base_prop']]=base_label.mean()
                                    df.loc[(hosp, year, month), idx[target, representation, 'prop']]=label.mean()
                                    df.loc[(hosp, year, month), idx[target, representation, 'fisher_pval']]=pval


                            df.to_csv(os.path.join(out_dir, "class_dist_test.csv"))
                            df.to_pickle(os.path.join(out_dir, "class_dist_test.pkl"))
    return df

def main_error_rate_change_detection(method, out_dir, **kwargs):
    global static_data
    load_static_data()

    text_files=[fname for fname in os.listdir(dir_path) if fname.startswith('result_') and fname.endswith(".txt")]

    base_hospital='UPMCPUH'
    base_year=2011
    base_month=2

    with open(os.path.join(out_dir, method.upper()+"_error_rate_changes.txt"), "w") as outfile:
        for target in targets:
            for representation in representations:
                for text_file in sorted(text_files):
                    if target not in text_file:
                        continue
                    if (representation in text_file):
                        print(target, representation)

                        with open(os.path.join(dir_path, text_file), 'rb') as f:
                            all_lines=f.readlines()

                            for modeltype in models:
                                model_lines=[line.decode() for line in all_lines if (" "+modeltype.upper()+"," in line.decode())]

                                base_lines=[line for line in model_lines if (base_hospital in line.split(",")[3]) and
                                    (str(base_year) in line.split(",")[5]) and 
                                    (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(base_month-month_step+1, base_month+1, 1)]))]

                                if len(base_lines)==0:
                                    continue
                                base_label=[l for l in base_lines if 'label,' in l][0]
                                base_label=get_values_from_line(base_label)
                                base_probs=[l for l in base_lines if 'y_pred_prob,' in l][0]
                                base_probs=get_values_from_line(base_probs)
                                
                                base_error_stream=to_error_stream(base_label, base_probs)
                                if len(base_error_stream)==0:
                                    continue

                                for hosp in hospitals:
                                    print(modeltype, hosp)
                                    for year in year_range:
                                        for month in month_intervals:
                                            lines=[line for line in model_lines if (hosp in line) and 
                                                (str(year) in line.split(",")[5]) and
                                                (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)]))
                                                ]

                                            label=None
                                            for line in lines:
                                                try:
                                                    if ('label,' in line):
                                                        label=get_values_from_line(line)
                                                    if ('y_pred_prob,' in line):
                                                        probs=get_values_from_line(line)
                                                except:
                                                    print(line)
                                                    print(hosp, year, month, target, representation)
                                                    raise
                                            if (label is None) or (np.unique(label).size < 2):
                                                continue
                                            
                                            error_stream=to_error_stream(label, probs)
                                            _,_,base_warn_idx,base_drift_idx,warn_idx,drift_idx=get_stream_drifts(method, base_error_stream, error_stream, **kwargs)
                                            
                                            outfile.write("target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, label, <{}> \r\n".format(target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in label])))
                                            outfile.write("target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, probs, <{}> \r\n".format(target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in probs])))
                                            outfile.write("target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, error_stream, <{}> \r\n".format(target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in error_stream])))
                                            outfile.write("target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, warning_index, <{}> \r\n".format(target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in warn_idx])))
                                            outfile.write("target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, drift_index, <{}> \r\n".format(target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in drift_idx])))

def main_mv_test_hospital_overtime(data_dir, out_dir, out_filename, n_threads, reps):

    df_results = pd.read_pickle(os.path.join(data_dir, "results_df_hospital_overtime.pkl"))
    data_files=[f for f in os.listdir(data_dir) if ('hospital-overtime-style' in f)]
    
    main_measures = ['AUC', 'APR', 'ECE']
    train_hospital='UPMCPUH'

    columns=[]
    for target in targets:
            for representation in representations:
                for modeltype in models:
                    for indep_test in ['base_rows', 'base_cols', 'data_rows', 'data_cols'] + independent_tests:
                        columns.append((target, representation, modeltype, indep_test))

    ind=[(hosp, yr, mnth) for hosp in hospitals for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'representation', 'model', 'indep_test'))
    indep_test_df=pd.DataFrame(index=ind, columns=cols)

    for indep_test in independent_tests:
        print('indep_test:', indep_test)
        for target in targets:
            for rep in representations:
                for f in data_files:
                    if (target in f) and ("_"+rep+"_" in f):
                        print(target, rep)
                        for modeltype in models:
                            print(modeltype)
                            data_key= "X_train_"+train_hospital+"_"+modeltype.upper()
                            try:
                                X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                            except KeyError as ke:
                                data_key= "X_train_"+train_hospital+"_"+modeltype
                                X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                
                            print('X1_shape:', X1.shape)

                            indep_test_df.loc[:, idx[target, rep, modeltype, 'base_rows']] = X1.shape[0]
                            indep_test_df.loc[:, idx[target, rep, modeltype, 'base_cols']] = X1.shape[1]

                            for hosp in hospitals:
                                for year in year_range:
                                    for month in month_intervals:
                                        if ~df_results.loc[(hosp, year, month), idx[target, modeltype, rep, main_measures]].isna().all():
                                            ## if any non-null measures is available for (hosp, year, month) index
                                            print(indep_test, target, rep, modeltype, hosp, year, month)
                                            data_key="X_test_"+hosp+"_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])+"_"+modeltype.upper()
                                            try:
                                                X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                            except KeyError as ke:
                                                data_key="X_test_"+hosp+"_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])+"_"+modeltype
                                                X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                            print('X2_shape:', X2.shape)
                                            indep_test_df.loc[(hosp, year, month), idx[target, rep, modeltype, 'data_rows']] = X2.shape[0]
                                            indep_test_df.loc[(hosp, year, month), idx[target, rep, modeltype, 'data_cols']] = X2.shape[1]

                                            t0=time.time()
                                            np.random.seed(0)
                                            stat, pvalue = KSample(indep_test).test(X1.values, X2.values, workers=n_threads, reps=reps, auto=True)
                                            t1=time.time()
                                            print("runtime=", str((t1-t0)), "seconds")
                                            print("stat, pval= {:0.3f}, {:0.3f}".format(stat, pvalue))

                                            indep_test_df.loc[(hosp, year, month), idx[target, rep, modeltype, indep_test]] = pvalue
                                            indep_test_df.to_csv(os.path.join(out_dir, out_filename+".csv"))
                                            indep_test_df.to_pickle(os.path.join(out_dir, out_filename+".pkl"))
                                            print('*'*30)

def main_mv_test_overall_overtime(data_dir, out_dir, out_filename, n_threads, reps):

    df_results = pd.read_pickle(os.path.join(data_dir, "results_df_overall_overtime.pkl"))
    data_files=[f for f in os.listdir(data_dir) if ('overall-overtime-style' in f)]

    main_measures = ['AUC', 'APR', 'ECE']
    training_years=[2008,2009,2010]
    
    columns=[]
    for target in targets:
            for representation in representations:
                for modeltype in models:
                    for indep_test in ['base_rows', 'base_cols', 'data_rows', 'data_cols'] + independent_tests:
                        columns.append((target, representation, modeltype, indep_test))

    ind=[(yr, mnth) for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'representation', 'model', 'indep_test'))
    indep_test_df=pd.DataFrame(index=ind, columns=cols)

    for indep_test in independent_tests:
        print('indep_test:', indep_test)
        for target in targets:
            for rep in representations:
                for f in data_files:
                    if (target in f) and ("_"+rep+"_" in f):
                        print(target, rep)
                        for modeltype in models:
                            print(modeltype)
                            data_key= "X_train_" + "-".join([str(i) for i in training_years]) + "_" + modeltype.upper()
                            try:
                                X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                            except KeyError as ke:
                                data_key="X_train_" + "-".join([str(i) for i in training_years])
                                X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                
                            print('X1_shape:', X1.shape)

                            indep_test_df.loc[:, idx[target, rep, modeltype, 'base_rows']] = X1.shape[0]
                            indep_test_df.loc[:, idx[target, rep, modeltype, 'base_cols']] = X1.shape[1]

                            for year in year_range:
                                for month in month_intervals:
                                    if ~df_results.loc[(year, month), idx[target, modeltype, rep, main_measures]].isna().all():
                                        ## if any non-null measures is available for (hosp, year, month) index
                                        print(indep_test, target, rep, modeltype, year, month)
                                        data_key="X_test_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])+"_"+modeltype.upper()
                                        try:
                                            X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                        except KeyError as ke:
                                            data_key="X_test_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])
                                            X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                        print('X2_shape:', X2.shape)
                                        indep_test_df.loc[(year, month), idx[target, rep, modeltype, 'data_rows']] = X2.shape[0]
                                        indep_test_df.loc[(year, month), idx[target, rep, modeltype, 'data_cols']] = X2.shape[1]

                                        t0=time.time()
                                        np.random.seed(0)
                                        stat, pvalue = KSample(indep_test).test(X1.values, X2.values, workers=n_threads, reps=reps, auto=True)
                                        t1=time.time()
                                        print("runtime=", str((t1-t0)), "seconds")
                                        print("stat, pval= {:0.3f}, {:0.3f}".format(stat, pvalue))

                                        indep_test_df.loc[(year, month), idx[target, rep, modeltype, indep_test]] = pvalue
                                        indep_test_df.to_csv(os.path.join(out_dir, out_filename+".csv"))
                                        indep_test_df.to_pickle(os.path.join(out_dir, out_filename+".pkl"))
                                        print('*'*30)

def main_mv_test_single_site(data_dir, out_dir, out_filename, n_threads, reps, sites):
    df_results = pd.read_pickle(os.path.join(data_dir, "results_df_single_site.pkl"))
    data_files=[f for f in os.listdir(data_dir) if ('single-site-style' in f)]
    
    main_measures = ['AUC', 'APR', 'ECE']
    training_years=[2008,2009,2010]

    columns=[]
    for target in targets:
            for representation in representations:
                for modeltype in models:
                    for indep_test in ['base_rows', 'base_cols', 'data_rows', 'data_cols'] + independent_tests:
                        columns.append((target, representation, modeltype, indep_test))

    ind=[(site, yr, mnth) for site in sites for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'representation', 'model', 'indep_test'))
    indep_test_df=pd.DataFrame(index=ind, columns=cols)

    for indep_test in independent_tests:
        for site in sites:
            for target in targets:
                for rep in representations:
                    for f in data_files:
                        if (target in f) and ("_"+rep+"_" in f) and ("_"+site+"_" in f):
                            print(target, rep)
                            for modeltype in models:
                                print(modeltype)
                                
                                data_key= "X_train_"+"-".join([str(i) for i in training_years])+"_"+modeltype.upper()
                                try:
                                    X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                except KeyError as ke:
                                    data_key= "X_train_"+"-".join([str(i) for i in training_years])
                                    X1=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                    
                                print('X1_shape:', X1.shape)

                                indep_test_df.loc[idx[site,:,:], idx[target, rep, modeltype, 'base_rows']] = X1.shape[0]
                                indep_test_df.loc[idx[site,:,:], idx[target, rep, modeltype, 'base_cols']] = X1.shape[1]

                                for year in year_range:
                                    for month in month_intervals:
                                        if ~df_results.loc[(site, year, month), idx[target, modeltype, rep, main_measures]].isna().all():
                                            ## if any non-null measures is available for (site, year, month) index
                                            print(indep_test, site, target, rep, modeltype, year, month)
                                            data_key="X_test_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])+"_"+modeltype.upper()
                                            try:
                                                X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                            except KeyError as ke:
                                                data_key="X_test_"+str(year)+"_"+"-".join([str(i) for i in [month-1, month]])
                                                X2=pd.read_hdf(os.path.join(data_dir, f), key=data_key)
                                            print('X2_shape:', X2.shape)
                                            indep_test_df.loc[(site, year, month), idx[target, rep, modeltype, 'data_rows']] = X2.shape[0]
                                            indep_test_df.loc[(site, year, month), idx[target, rep, modeltype, 'data_cols']] = X2.shape[1]

                                            t0=time.time()
                                            np.random.seed(0)
                                            stat, pvalue = KSample(indep_test).test(X1.values, X2.values, workers=n_threads, reps=reps, auto=True)
                                            t1=time.time()
                                            print("runtime=", str((t1-t0)), "seconds")
                                            print("stat, pval= {:0.3f}, {:0.3f}".format(stat, pvalue))

                                            indep_test_df.loc[(site, year, month), idx[target, rep, modeltype, indep_test]] = pvalue
                                            indep_test_df.to_csv(os.path.join(out_dir, out_filename+".csv"))
                                            indep_test_df.to_pickle(os.path.join(out_dir, out_filename+".pkl"))
                                            print('*'*30)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='drift detection for the overtime-hospital experiments')
    parser.add_argument('--exp_type', type=str, default=None, choices=["error_rate_change_detection", "class_dist_change_detection", "mv_test"])
    parser.add_argument('--drift_detection_method', type=str, default="ADWIN", choices=["ADWIN", "DDM", "EDDM", "HDDM_A", "HDDM_W", "KSWIN", "PageHinkley"])
    parser.add_argument('--dir_path', type=str, default="", help="full path to directory containing probability and label files")
    parser.add_argument('--out_dir', type=str, default="", help="full path to output directory")
    parser.add_argument('--source_train_type', type=str, default=None, help="['overall_overtime', 'hospital_wise', 'icu_type', 'single_site', 'hospital_overtime']")
    parser.add_argument('--n_threads', type=int, default=1, help="used for multivariate tests")
    parser.add_argument('--sites', type=str, nargs='+', default=None, help="required when source_train_type=single_site, example: ['MICU', 'CTICU', 'UPMCPUH', 'UPMCSHY']")
    parser.add_argument('--reps', type=int, default=None, help="number of repeats in permutation multivariate tests")


    args = parser.parse_args()

    idx=pd.IndexSlice

    if isinstance(args.sites, str):
        args.sites=[args.sites]

    dir_path=args.dir_path
    out_dir=args.out_dir
    if out_dir=="":
        out_dir=dir_path

    targets = ['mort_icu', 'los_3']
    representations = ['raw', 'pca']
    models=['rf','lr','nb','rbf-svm']

    # site_info = pd.read_pickle(os.path.join(dir_path, "site_info.pkl"))
    hospitals = ['UPMCBED','UPMCEAS','UPMCHAM','UPMCHZN','UPMCMCK','UPMCMER','UPMCMWH','UPMCNOR','UPMCPAS','UPMCPUH','UPMCSHY','UPMCSMH']
    # icu_units = sorted(site_info["icu_category"].unique().tolist())
    year_range = np.arange(2011, 2015)
    month_step = 2
    month_intervals = np.arange(month_step, 13, month_step)

    # independent_tests = ["CCA", "Dcorr", "RV", "Hsic", "HHG", "MGC"]
    independent_tests = ["CCA", "Dcorr", "RV"]

    t0=time.time()
    
    if args.exp_type=="error_rate_change_detection":
        main_error_rate_change_detection(args.drift_detection_method, out_dir)
    elif args.exp_type=="class_dist_change_detection":
        main_class_dist_change_detection(dir_path, out_dir)
    elif args.exp_type=="mv_test":
        out_filename="indep_tests_reps_"+str(args.reps)+"_"+args.source_train_type
        if args.source_train_type=="hospital_overtime":
            main_mv_test_hospital_overtime(dir_path, out_dir, out_filename, args.n_threads, reps=args.reps)
        elif args.source_train_type=="single_site":
            main_mv_test_single_site(dir_path, out_dir, out_filename, args.n_threads, reps=args.reps, sites=args.sites)
        elif args.source_train_type=="overall_overtime":
            main_mv_test_overall_overtime(dir_path, out_dir, out_filename, args.n_threads, reps=args.reps)
        else:
            raise "method for "+args.source_train_type+" not found!"
    else:
        raise "method for "+args.exp_type+" not found!"

    t1=time.time()
    print("Done. Total time={:0.1f} seconds".format(t1-t0))