import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils import resample
from util.utils import get_calibration_metrics, stat_ci, stat_pval

import argparse
import os
import time

def get_values_from_line(line):
    values=[]
    if len(line)>0:
        values=line.split("<")[-1].split(">")[0]
        if values!='':
            values=[float(i) for i in values.split(",")]
    return np.array(values)

def generate_bootstrap(label, y_pred_prob, n_bootstrap):
    '''
    generate stratified bootstrapping samples of predictions and return metrics for each sample
    '''
    auroc_list=[]
    auprc_list=[]
    ece_list=[]

    ## add the original metrics
    auroc_list.append(roc_auc_score(label, y_pred_prob))
    auprc_list.append(average_precision_score(label, y_pred_prob))
    try:
        _,_,ece,_=get_calibration_metrics(label, y_pred_prob,n_bins=10,bin_strategy='quantile')
        ece_list.append(ece)
    except:
        pass
    for i in range(n_bootstrap):
        indices=resample(range(len(label)),random_state=i,n_samples=len(label),replace=True,stratify=label)
        y_true=label[indices]
        probs=y_pred_prob[indices]
        auroc_list.append(roc_auc_score(y_true, probs))
        auprc_list.append(average_precision_score(y_true, probs))
        try:
            _,_,ece,_=get_calibration_metrics(y_true, probs,n_bins=10,bin_strategy='quantile')
            ece_list.append(ece)
        except:
            pass
    return auroc_list, auprc_list, ece_list

def main_bootstrap_hospOvertime(n_bootstrap, data_dir, out_filename, out_dir=""):

    text_files=[fname for fname in os.listdir(data_dir) if fname.startswith('result_hospital-overtime-style') and fname.endswith(".txt")]

    with open(os.path.join(out_dir, out_filename), "w") as out_file:
        for target in targets:
            for representation in representations:
                for text_file in sorted(text_files):
                    if target not in text_file:
                        continue
                    if (representation in text_file):
                        print(target, representation)

                        with open(os.path.join(data_dir, text_file), 'rb') as f:
                            all_lines=f.readlines()

                        for modeltype in models:
                            print('modeltype=', modeltype)
                            model_lines=[line.decode() for line in all_lines if ", "+modeltype.upper()+"," in line.decode()]

                            ## generate bootstrap for training metrics if any
                            train_label=[l for l in model_lines if 'train_label' in l]
                            train_prob=[l for l in model_lines if 'train_y_pred_prob' in l]

                            if len(train_label)>0 and len(train_prob)>0:
                                train_label=get_values_from_line(train_label[0])
                                train_prob=get_values_from_line(train_prob[0])
                                auroc_list, auprc_list, ece_list = generate_bootstrap(train_label, train_prob, n_bootstrap)

                                out_file.write('target, {}, representation, {}, model, {}, train_AUROC, <{}> \r\n'.format(
                                        target, representation, modeltype.upper(), ",".join([str(i) for i in auroc_list])))

                                out_file.write('target, {}, representation, {}, model, {}, train_AUPRC, <{}> \r\n'.format(
                                target, representation, modeltype.upper(), ",".join([str(i) for i in auprc_list])))

                                out_file.write('target, {}, representation, {}, model, {}, train_ECE, <{}> \r\n'.format(
                                target, representation, modeltype.upper(), ",".join([str(i) for i in ece_list])))

                            for hosp in hospitals:
                                for year in year_range:
                                    for month in month_intervals:
                                        lines=[line for line in model_lines if (hosp in line) and 
                                            (str(year) in line.split(",")[5]) and
                                            (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)]))
                                            ]
                                        if len(lines)==0:
                                            continue
                                        label=pred=y_pred_prob=None
                                        for line in lines:
                                            try:
                                                if ('label,' in line):
                                                    label=line.split("label")[1].split("<")[1].split(">")[0].split(",")
                                                    label=np.array([int(float(i)) for i in label])
                                                if ('pred,' in line):
                                                    pred=line.split("pred,")[1].split("<")[1].split(">")[0].split(",")
                                                    pred=np.array([float(i) for i in pred])
                                                if ('y_pred_prob,' in line):
                                                    y_pred_prob=line.split("y_pred_prob,")[1].split("<")[1].split(">")[0].split(",")
                                                    y_pred_prob=np.array([float(i) for i in y_pred_prob])
                                            except:
                                                print(line)
                                                print(hosp, year, month, target, modeltype, representation)
                                                raise
                                        if (label is None) or (np.unique(label).size < 2):
                                            continue
                                        print('bootstrapping (',hosp, year, month,')')
                                        auroc_list, auprc_list, ece_list = generate_bootstrap(label, y_pred_prob, n_bootstrap)

                                        out_file.write('target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, AUROC, <{}> \r\n'.format(
                                        target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in auroc_list])))

                                        out_file.write('target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, AUPRC, <{}> \r\n'.format(
                                        target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in auprc_list])))

                                        out_file.write('target, {}, representation, {}, model, {}, hospital, {}, year, {}, month, {}, ECE, <{}> \r\n'.format(
                                        target, representation, modeltype.upper(), hosp, str(year), str(month), ",".join([str(i) for i in ece_list])))

    return

def main_bootstrap_overall_overtime(n_bootstrap, data_dir, out_filename, out_dir=""):

    text_files=[fname for fname in os.listdir(data_dir) if fname.startswith('result_overall-overtime-style') and fname.endswith(".txt")]

    with open(os.path.join(out_dir, out_filename), "w") as out_file:
        for target in targets:
            for representation in representations:
                for modeltype in models:
                    for text_file in sorted(text_files):
                        if target not in text_file:
                            continue
                        if (representation in text_file) and ("_"+modeltype.upper()+"_" in text_file):
                            print(target, representation, modeltype)

                            with open(os.path.join(data_dir, text_file), 'rb') as f:
                                all_lines=f.readlines()

                            for year in year_range:
                                year_lines=[line.decode() for line in all_lines if ("year, " + str(year) in line.decode())]
                                for month in month_intervals:
                                    lines=[line for line in year_lines if (line.split("<")[1].split(">")[0] == ','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)]))]
                                    if len(lines)==0:
                                        continue
                                    label=pred=y_pred_prob=None
                                    for line in lines:
                                        try:
                                            if ('label,' in line):
                                                label=line.split("label")[1].split("<")[1].split(">")[0].split(",")
                                                label=np.array([int(float(i)) for i in label])
                                            if ('pred,' in line):
                                                pred=line.split("pred,")[1].split("<")[1].split(">")[0].split(",")
                                                pred=np.array([float(i) for i in pred])
                                            if ('y_pred_prob,' in line):
                                                y_pred_prob=line.split("y_pred_prob,")[1].split("<")[1].split(">")[0].split(",")
                                                y_pred_prob=np.array([float(i) for i in y_pred_prob])
                                        except:
                                            print(line)
                                            print(year, month, target, representation, modeltype)
                                            raise
                                    if (label is None) or (np.unique(label).size < 2):
                                        continue
                                    auroc_list=[]
                                    auprc_list=[]
                                    ece_list=[]

                                    auroc_list.append(roc_auc_score(label, y_pred_prob))
                                    auprc_list.append(average_precision_score(label, y_pred_prob))
                                    try:
                                        _,_,ece,_=get_calibration_metrics(label, y_pred_prob,n_bins=10,bin_strategy='quantile')
                                        ece_list.append(ece)
                                    except:
                                        pass
                                    print('bootstrapping (', year, month,')')
                                    for i in range(n_bootstrap):
                                        # indices=np.random.randint(0, len(label), len(label))
                                        indices=resample(range(len(label)),random_state=i,n_samples=len(label),replace=True,stratify=label)
                                        y_true=label[indices]
                                        y_pred=pred[indices]
                                        probs=y_pred_prob[indices]
                                        auroc_list.append(roc_auc_score(y_true, probs))
                                        auprc_list.append(average_precision_score(y_true, probs))
                                        try:
                                            _,_,ece,_=get_calibration_metrics(y_true, probs,n_bins=10,bin_strategy='quantile')
                                            ece_list.append(ece)
                                        except:
                                            pass

                                    out_file.write('target, {}, representation, {}, model, {}, year, {}, month, {}, AUROC, <{}> \r\n'.format(
                                    target, representation, modeltype.upper(), str(year), str(month), ",".join([str(i) for i in auroc_list])))

                                    out_file.write('target, {}, representation, {}, model, {}, year, {}, month, {}, AUPRC, <{}> \r\n'.format(
                                    target, representation, modeltype.upper(), str(year), str(month), ",".join([str(i) for i in auprc_list])))

                                    out_file.write('target, {}, representation, {}, model, {}, year, {}, month, {}, ECE, <{}> \r\n'.format(
                                    target, representation, modeltype.upper(), str(year), str(month), ",".join([str(i) for i in ece_list])))

    return

def main_bootstrap_single_site(n_bootstrap, data_dir, out_filename, out_dir=""):

    text_files=[fname for fname in os.listdir(data_dir) if fname.startswith('result_single-site-style') and fname.endswith(".txt")]

    with open(os.path.join(out_dir, out_filename), "w") as out_file:
        for target in targets:
            for representation in representations:
                for site in hospitals+icu_types:
                    for modeltype in models:
                        for text_file in sorted(text_files):
                            if (target in text_file) and (representation in text_file) and (site in text_file) and (modeltype.upper() in text_file):
                                print(target, representation, site, modeltype.upper())

                                with open(os.path.join(data_dir, text_file), 'rb') as f:
                                    all_lines=f.readlines()
                                all_lines=[line.decode() for line in all_lines]

                                ## generate bootstrap for training metrics if any
                                train_label=[l for l in all_lines if 'train_label' in l]
                                train_prob=[l for l in all_lines if 'train_y_pred_prob' in l]

                                if len(train_label)>0 and len(train_prob)>0:
                                    train_label=get_values_from_line(train_label[0])
                                    train_prob=get_values_from_line(train_prob[0])
                                    auroc_list, auprc_list, ece_list = generate_bootstrap(train_label, train_prob, n_bootstrap)

                                    out_file.write('target, {}, representation, {}, site, {}, model, {}, train_AUROC, <{}> \r\n'.format(
                                            target, representation, site.upper(), modeltype.upper(), ",".join([str(i) for i in auroc_list])))

                                    out_file.write('target, {}, representation, {}, site, {}, model, {}, train_AUPRC, <{}> \r\n'.format(
                                    target, representation, site.upper(), modeltype.upper(), ",".join([str(i) for i in auprc_list])))

                                    out_file.write('target, {}, representation, {}, site, {}, model, {}, train_ECE, <{}> \r\n'.format(
                                    target, representation, site.upper(), modeltype.upper(), ",".join([str(i) for i in ece_list])))

                                for year in year_range:
                                    for month in month_intervals:
                                        lines=[line for line in all_lines if ('year, '+str(year) in line) and
                                            ('months, <'+','.join([str(m) for m in np.arange(month-month_step+1, month+1, 1)])+'>' in line)
                                            ]
                                        if len(lines)==0:
                                            continue
                                        label=pred=y_pred_prob=None
                                        for line in lines:
                                            try:
                                                if ('label,' in line):
                                                    label=line.split("label")[1].split("<")[1].split(">")[0].split(",")
                                                    label=np.array([int(float(i)) for i in label])
                                                if ('y_pred_prob,' in line):
                                                    y_pred_prob=line.split("y_pred_prob,")[1].split("<")[1].split(">")[0].split(",")
                                                    y_pred_prob=np.array([float(i) for i in y_pred_prob])
                                            except:
                                                print(line)
                                                print(site, year, month, target, modeltype, representation)
                                                raise
                                        if (label is None) or (np.unique(label).size < 2):
                                            continue
                                        print('bootstrapping (', target, representation, site, modeltype.upper(), year, month,')')
                                        auroc_list, auprc_list, ece_list = generate_bootstrap(label, y_pred_prob, n_bootstrap)

                                        out_file.write('target, {}, representation, {}, site, {}, model, {}, year, {}, month, {}, AUROC, <{}> \r\n'.format(
                                        target, representation, site.upper(), modeltype.upper(), str(year), str(month), ",".join([str(i) for i in auroc_list])))

                                        out_file.write('target, {}, representation, {}, site, {}, model, {}, year, {}, month, {}, AUPRC, <{}> \r\n'.format(
                                        target, representation, site.upper(), modeltype.upper(), str(year), str(month), ",".join([str(i) for i in auprc_list])))

                                        out_file.write('target, {}, representation, {}, site, {}, model, {}, year, {}, month, {}, ECE, <{}> \r\n'.format(
                                        target, representation, site.upper(), modeltype.upper(), str(year), str(month), ",".join([str(i) for i in ece_list])))

    return

def main_stats_hospOvertime(data_dir, bs_filename, out_filename, stat_test="mannwhitneyu", out_dir=""):
    '''
    supported stat_tests are ["wicoxon", "mannwhitneyu"]
    '''
    
    with open(os.path.join(data_dir, bs_filename), "r") as f:
        all_lines=f.readlines()

    columns=[]
    for target in targets:
        for model in models:
            for rep in representations:
                for measurement in measures:
                    for stat in stats:
                        columns.append((target,model, rep, measurement, stat))

    ind=[(hosp, yr, mnth) for hosp in hospitals for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('hospital', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement', 'stat'))
    result_df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for modeltype in models:
            for rep in representations:
                lines=[l for l in all_lines if (target in l) and (", "+rep+"," in l) and
                        (", "+modeltype.upper()+"," in l)]
                for measure in measures:
                    print(target, rep, modeltype, measure)
                    base_line=[line for line in lines if ("train_"+measure.upper() in line)]
                    if len(base_line)==0:
                        continue
                    base_line=base_line[0]
                    base_values=get_values_from_line(base_line)
                    if len(base_values)==0:
                        continue
                    base_mean, base_ci_l, base_ci_u=stat_ci(base_values)
                    for line in lines:
                        if (", "+measure.upper()+"," in line):
                            values=get_values_from_line(line)
                            if len(values)==0:
                                continue
                            hosp=line.split(",")[7].strip()
                            year=int(line.split(",")[9].strip())
                            month=int(line.split(",")[11].strip())
                            mean_score, ci_lower, ci_upper=stat_ci(values)
                            _, pval=stat_pval(values, base_values, test=stat_test)
                            # print(hosp,year,month,target, rep, modeltype, measure, mean_score, ci_lower, ci_upper, pval)
                            result_df.loc[(hosp,year,month), idx[target, modeltype, rep, measure, ['N', 'base_mean', 'base_CI_L', 'base_CI_U', 'mean', 'CI_L','CI_U', 'pval']]]=(len(values), base_mean, base_ci_l, base_ci_u, mean_score, ci_lower, ci_upper, pval)
                result_df.to_csv(os.path.join(out_dir, out_filename))
                result_df.to_pickle(os.path.join(out_dir, out_filename.split(".")[0]+'.pkl'))
    return

def main_stats_overall_overtime(data_dir, bs_filename, out_filename, stat_test="mannwhitneyu", out_dir=""):
    '''
    supported stat_tests are ["wicoxon", "mannwhitneyu"]
    '''
    
    with open(os.path.join(data_dir, bs_filename), "r") as f:
        all_lines=f.readlines()
    
    base_year=2011
    base_month=2

    columns=[]
    for target in targets:
        for model in models:
            for rep in representations:
                for measurement in measures:
                    for stat in stats:
                        columns.append((target,model, rep, measurement, stat))

    ind=[(yr, mnth) for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement', 'stat'))
    result_df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for modeltype in models:
            for rep in representations:
                for measure in measures:
                    lines=[l for l in all_lines if (target in l) and (rep in l.split(",")[3]) and
                        (modeltype.upper() in l.split(",")[5]) and (measure in l.split(",")[10])]
                    if len(lines)==0:
                        continue
                    print(target, rep, modeltype, measure)
                    base_line=[line for line in lines if (str(base_year) in line.split(",")[7]) and (int(line.split(",")[9].strip()) == base_month)][0]
                    base_values=get_values_from_line(base_line)
                    if len(base_values)==0:
                        continue
                    base_mean, base_ci_l, base_ci_u=stat_ci(base_values)
                    for line in lines:
                        values=get_values_from_line(line)
                        if len(values)==0:
                            continue
                        year=int(line.split(",")[7].strip())
                        month=int(line.split(",")[9].strip())
                        mean_score, ci_lower, ci_upper=stat_ci(values)
                        _, pval=stat_pval(values, base_values, test=stat_test)
                        result_df.loc[(year,month), idx[target, modeltype, rep, measure, ['N', 'base_mean', 'base_CI_L', 'base_CI_U', 'mean', 'CI_L','CI_U', 'pval']]]=(len(values), base_mean, base_ci_l, base_ci_u, mean_score, ci_lower, ci_upper, pval)

                result_df.to_csv(os.path.join(out_dir, out_filename))
                result_df.to_pickle(os.path.join(out_dir, out_filename.split(".")[0]+'.pkl'))
    return

def main_stats_single_site(data_dir, bs_filename, out_filename, stat_test="mannwhitneyu", out_dir=""):
    '''
    supported stat_tests are ["wicoxon", "mannwhitneyu"]
    '''
    
    with open(os.path.join(data_dir, bs_filename), "r") as f:
        all_lines=f.readlines()

    columns=[]
    for target in targets:
        for model in models:
            for rep in representations:
                for measurement in measures:
                    for stat in stats:
                        columns.append((target,model, rep, measurement, stat))

    sites=hospitals+icu_types
    ind=[(site, yr, mnth) for site in sites for yr in year_range for mnth in month_intervals]
    ind=pd.MultiIndex.from_tuples(ind, names=('site', 'year', 'month'))
    cols=pd.MultiIndex.from_tuples(columns, names=('target', 'model', 'representation', 'measurement', 'stat'))
    result_df=pd.DataFrame(index=ind, columns=cols)

    for target in targets:
        for modeltype in models:
            for rep in representations:
                lines=[l for l in all_lines if (target in l) and (", "+rep+"," in l) and
                        (", "+modeltype.upper()+"," in l)]
                for measure in measures:
                    print(target, rep, modeltype, measure)
                    base_line=[line for line in lines if ("train_"+measure.upper() in line)]
                    if len(base_line)==0:
                        continue
                    base_line=base_line[0]
                    base_values=get_values_from_line(base_line)
                    if len(base_values)==0:
                        continue
                    base_mean, base_ci_l, base_ci_u=stat_ci(base_values)
                    for line in lines:
                        if (", "+measure.upper()+"," in line):
                            values=get_values_from_line(line)
                            if len(values)==0:
                                continue
                            site=line.split("site,")[1].split(",")[0].strip()
                            year=int(line.split("year,")[1].split(",")[0].strip())
                            month=int(line.split("month,")[1].split(",")[0].strip())
                            mean_score, ci_lower, ci_upper=stat_ci(values)
                            _, pval=stat_pval(values, base_values, test=stat_test)
                            # print(hosp,year,month,target, rep, modeltype, measure, mean_score, ci_lower, ci_upper, pval)
                            result_df.loc[(site,year,month), idx[target, modeltype, rep, measure, ['N', 'base_mean', 'base_CI_L', 'base_CI_U', 'mean', 'CI_L','CI_U', 'pval']]]=(len(values), base_mean, base_ci_l, base_ci_u, mean_score, ci_lower, ci_upper, pval)
                result_df.to_csv(os.path.join(out_dir, out_filename))
                result_df.to_pickle(os.path.join(out_dir, out_filename.split(".")[0]+'.pkl'))

    result_df.dropna(axis=0, how="all", inplace=True)
    result_df.to_csv(os.path.join(out_dir, out_filename))
    result_df.to_pickle(os.path.join(out_dir, out_filename.split(".")[0]+'.pkl'))
    return



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='booststrap predictions and generate CI and pvals for the metrics')
    parser.add_argument('--run_bootstrap', type=int, default=0, help="run bootstrap or read from existing file. 0: read from existing file, 1: run bootstrap")
    parser.add_argument('--n_bootstrap', type=int, default=100, help="num of bootstrap samples")
    parser.add_argument('--generate_stats', type=int, default=0, help="generate stats from bootstrap. 0: False, 1: True")
    parser.add_argument('--stat_test', type=str, default="mannwhitneyu", choices=["mannwhitneyu", "wilcoxon"], help="independent test to use to compare vector of metrics")
    parser.add_argument('--data_dir', type=str, default="", help="full path to directory containing probability and label files and/or generated bootstraps")
    parser.add_argument('--out_dir', type=str, default="", help="full path to output directory")
    parser.add_argument('--source_train_type', type=str, default=None, help="['overall_overtime', 'hospital_wise', 'icu_type', 'single_site', 'hospital_overtime']")


    args = parser.parse_args()
    
    targets = ['mort_icu', 'los_3']
    representations = ['raw', 'pca']
    models=['rf','lr','nb','rbf-svm']
    measures=['AUROC', 'AUPRC', 'ECE']
    stats=['N', 'base_mean', 'base_CI_L', 'base_CI_U', 'mean', 'CI_L', 'CI_U', 'pval']

    hospitals = ['UPMCBED','UPMCEAS','UPMCHAM','UPMCHZN','UPMCMCK','UPMCMER','UPMCMWH','UPMCNOR','UPMCPAS','UPMCPUH','UPMCSHY','UPMCSMH']
    icu_types = ['CTICU', 'MICU']
    year_range = np.arange(2011, 2015)
    month_step = 2
    month_intervals = np.arange(month_step, 13, month_step)

    data_dir=args.data_dir
    out_dir=args.out_dir
    if out_dir=="":
        out_dir=data_dir
    
    bs_file="bootstrap_" + str(args.n_bootstrap) + "_" + args.source_train_type + ".txt"
    stats_file="bootstrap_stats_" + str(args.n_bootstrap) + "_" + args.source_train_type + "_" + args.stat_test + ".csv"
    
    idx=pd.IndexSlice

    t0=time.time()
    if(args.run_bootstrap==1):
        print("running bootstrap ...")
        if args.source_train_type=="overall_overtime":
            main_bootstrap_overall_overtime(n_bootstrap=args.n_bootstrap, data_dir=data_dir, out_filename=bs_file, out_dir=out_dir)
        elif args.source_train_type=="hospital_overtime":
            main_bootstrap_hospOvertime(n_bootstrap=args.n_bootstrap, data_dir=data_dir, out_filename=bs_file, out_dir=out_dir)
        elif args.source_train_type=="single_site":
            main_bootstrap_single_site(n_bootstrap=args.n_bootstrap, data_dir=data_dir, out_filename=bs_file, out_dir=out_dir)
        else:
            raise args.source_train_type + "is not implemented yet!"
    if(args.generate_stats==1):
        print("generating stats from bootstrap samples...")
        if args.source_train_type=="overall_overtime":
            main_stats_overall_overtime(data_dir=data_dir, bs_filename=bs_file, out_filename=stats_file, stat_test=args.stat_test, out_dir=out_dir)
        elif args.source_train_type=="hospital_overtime":
            main_stats_hospOvertime(data_dir=data_dir, bs_filename=bs_file, out_filename=stats_file, stat_test=args.stat_test, out_dir=out_dir)
        elif args.source_train_type=="single_site":
            main_stats_single_site(data_dir=data_dir, bs_filename=bs_file, out_filename=stats_file, stat_test=args.stat_test, out_dir=out_dir)
        else:
            raise args.source_train_type + "is not implemented yet!"
    
    t1=time.time()
    print("Done. in {:0.2f} seconds".format(t1-t0))




