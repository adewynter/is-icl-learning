import re
import numpy as np
import pandas as pd
import os
import json
from copy import deepcopy
from sklearn.metrics import auc



def get_label_for_vm_sum_fn(entry: dict):
    """ 
    Compute the appropriate label for a given entry of Vending_Machine_Sum

    ---
    Params:
    entry (dict) the line item for a vending machine (sum) dataset 
    """
    vals = {"biscuit": 20,
            "coffee": 15,
            "soda": 25}
    ex = entry["Entry"] 
    balance = 0
    for e in ex.split(","):
        if e == "":
            continue
        if e[0] == "+":
            balance += int(e[1:])
        elif e[0] in "0123456789":
            balance += int(e)
        else:
            balance -= vals[e]
    return int(balance)


def compute_accuracy(arr: list, problem: str, labels:list=[0, 1], punish:bool=True, is_cot:bool=False, 
                    do_trace:bool=None, output_auc:bool=False):
    """
    Compute accuracy for a specific problem. 

    ---
    Params
    arr (list): a list of [{"Response": ..., "Label": ...}] dictionaries.
    problem (str): the problem evaluated
    labels (list): acceptable label set. Probably unneeded
    punish (bool): if True, parsing errors are mapped to the opposite label.
    do_trace (bool): pipe errors to a file for further analysis
    is_cot (bool): if this is one of CoT or SoT, or another thing.
    output_auc (bool): stat tests
    """
    if is_cot:
        return compute_acc_cot(arr, problem, labels=labels, punish=punish, do_trace=do_trace, output_auc=output_auc)

    preds = []
    failure_rate = 0
    for entry in arr:

        if problem == "vending_machine_with_sum":
            entry["Label"] = get_label_for_vm_sum_fn(entry)

        pred = None
        if "Response" not in entry:
            failure_rate += 1
            if do_trace is not None: record_trace(f"response not in entry: {resp}\n", do_trace)
            continue
        if type(entry["Response"]) == list:
            entry["Response"] = entry["Response"][0]
        resp = helper_fn_for_parsing(entry["Response"])
        
        # Our code for out of tokens, broken requests, etc.
        if resp == "fail": 
            failure_rate += 1
            continue
        if resp == "" or resp is None:
            failure_rate += 1
            if do_trace is not None: record_trace("response is empty\n", do_trace)
            if punish:
                if entry["Label"] == 0:
                    preds.append((entry["Label"], 1))
                else:
                    preds.append((entry["Label"], 0))
            continue
        try:
            resp = resp.split("\n")[0].split(" ")[0]
            pred = int(resp)
            preds.append((int(entry["Label"]), pred))
        except:
            failure_rate += 1
            if do_trace is not None: record_trace(f"response not an integer: {resp}\n", do_trace)
            if punish:
                if entry["Label"] == 0:
                    preds.append((entry["Label"], 1))
                else:
                    preds.append((entry["Label"], 0))
            continue

    # Smaller models like Phi-2 (which we decided to drop)
    # have incredibly high failure rates.
    if preds == []:
        r_auc = [e["Label"] for e in arr], [1 if e["Label"] == 0 else 0 for e in arr]
        accuracy = 0
        frate = 100
    else:
        r_auc = [x[0] for x in preds], [x[1] for x in preds]
        accuracy = sum([x[0] == x[1] for x in preds])*100./len(preds) # Always over the preds
        frate = failure_rate*100./len(arr) # Always over the array
    if output_auc:
        return round(accuracy, 2), r_auc, frate
    return round(accuracy, 2), frate


def helper_fn_for_parsing(entry: dict):
    """
    Wee parser built based on the traces

    ---
    Params:
    entry (dict) the line item for a dataset 
    """
    entry = entry.replace(".", "").replace(",", "")
    # <s> is for mixtral, llama; <bos> for gemma
    entry = entry.replace("<s>", "").replace("<bos>", "")
    return entry.lower().strip()


def compute_acc_cot(arr:list, problem:str, labels:list=[0,1], punish:bool=False, do_trace:str=None, output_auc:bool=False):
    """
    Compute accuracy for CoT/SoT specifically.

    ---
    Params:
    arr (list): a list of [{"Response": ..., "Label": ...}] dictionaries.
    problem (str): the problem evaluated (only for CoT)
    labels (list): acceptable label set. Probably unneeded
    punish (bool): if True, parsing errors are mapped to zero.
    do_trace (bool): pipe errors to a file for further analysis
    output_auc (bool): stat tests
    """
    preds = []
    failure_rate = 0
    check_labels = True
    for entry in arr:
        if problem == "vending_machine_with_sum":
            check_labels = False
            entry["Label"] = get_label_for_vm_sum_fn(entry)

        pred = None
        # Search for the output string:
        if "Response" not in entry:
            failure_rate += 1
            if do_trace is not None: record_trace(f"response not in entry: {resp}\n", do_trace)
            if punish:
                if entry["Label"] == 0:
                    preds.append((entry["Label"], 1))
                else:
                    preds.append((entry["Label"], 0))
            continue
        # Sometimes we also got the latency... entry["Response"] = (resp, 0)
        if type(entry["Response"]) == list:
            entry["Response"] = entry["Response"][0]
        out_string = helper_fn_for_parsing(entry["Response"])

        # Our code for out of tokens, broken requests, etc.
        if out_string == "fail": 
            failure_rate += 1
            continue
        # Get the CoT-formatted answer
        if problem.lower() in ["parity", "pattern_matching", "stack", "maze_complete", "maze_solve", "reversal"]:
            resp = out_string.split("so the answer is ")
        elif problem.lower() in ["vending_machine", "vending_machine_with_sum", "hamiltonian"]:
            resp = out_string.split("the answer is then")
        else:
            print(f"Problem \"{problem}\" not implemented!")
            raise

        try:
            intresp = re.search(r'\d', resp[-1][::-1])
            if intresp is not None:
                intresp = intresp.group()[::-1]
            pred = int(intresp) #resp[-1])
        except:
            if do_trace is not None: record_trace(f"response is not an integer. Parsed {intresp} | {resp}\n", do_trace)
            failure_rate += 1
            if punish:
                if entry["Label"] == 0:
                    preds.append((entry["Label"], 1))
                else:
                    preds.append((entry["Label"], 0))
            continue
        if pred not in labels and check_labels:
            failure_rate += 1
            if do_trace is not None: record_trace(f"label is not in set: {pred}\n", do_trace)
            if punish:
                if entry["Label"] == 0:
                    preds.append((entry["Label"], 1))
                else:
                    preds.append((entry["Label"], 0))
            continue
        preds.append((entry["Label"], pred))

    # Smaller models like Phi-2 have incredibly high failure rates.
    if preds == []:
        r_auc = [e["Label"] for e in arr], [1 if e["Label"] == 0 else 0 for e in arr]
        accuracy = 0
        frate = 100
    else:
        r_auc = [x[0] for x in preds], [x[1] for x in preds]
        accuracy = sum([x[0] == x[1] for x in preds])*100./len(preds)
        frate = failure_rate*100./len(arr)

    if output_auc:
        return round(accuracy, 2), r_auc, frate
    return round(accuracy, 2), frate


def check_for_skip(path_func, shots: list):
    """
    Check if we have to skip this entry (because Turbo is missing one entry)
    ---
    Params:
    path_func (call): a function that gives a path given a shot
    shots (list): list of shots
    """
    fname = path_func(shots[0])
    if "turbo" in fname:
        if "MazeComplete" in fname:
            if "word_salad" in fname:
                return {s: {d: (0, 0) for d in [0, 0.2, 0.45, 0.65, 0.85]} for s in shots}
    return None


def record_trace(message: str, fname: str):
    """
    Record the type of error observed during parsing.
    ---
    Params:
    message (str): message to write
    fname (str): the filename for logging
    """
    with open(f"trace_{fname}.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


def collect(path_dict: str, shots: list, is_cot:bool=False, problem:str=None, do_trace:str=None, 
            punish:bool=False, output_auc:bool=False) -> dict:
    """
    Open a predictions file and compute the accuracy.
    Returns a dict of the form {shot: {delta: (accuracy, error)}}

    ---
    Params:
    path_dict (str): a function that retrieves filenames based on the shot chosen
    shots (list): a list of the shots that this function will collect
    is_cot (bool, defaults to False): proof is left to the reader
    problem (str, defaults to None): the problem you are solving (for parsing)
    do_trace (str, defaults to None): if not None, this acts as a filename and enable logging of every failed datapoint to do_trace
    punish (bool, defaults to False): flips the labels 
    """
    check = check_for_skip(path_dict, shots)
    if check is not None:
        return check

    results = {}
    for shot in shots:
        file = path_dict(shot)
        results[shot] = {}
        predictions = json.load(open(file, "r", encoding="utf-8"))
        for delta in [0, 0.2, 0.45, 0.65, 0.85]:
            _predictions = predictions[f"delta_{delta}"]
            acc = compute_accuracy(_predictions, problem, is_cot=is_cot, do_trace=do_trace, 
                                   punish=punish, output_auc=output_auc)
            if output_auc:
                results[shot][delta] = (acc[0], acc[1], acc[-1])
            else:
                results[shot][delta] = (acc[0], acc[-1])
    return results


def write_to_excel(excel_filename, suff, problem, row, accs, errs, header=None):
    """ Write the predictions to an Excel file for further analysis.
    ---
    Params:
     
    """
    # This function is overly verbose because pandas' API is the worst
    if header is None:
        header = {"Problem": suff}

    if os.path.exists(excel_filename):
        with pd.ExcelWriter(excel_filename, mode="a", if_sheet_exists="overlay") as f:
            df = pd.DataFrame(header, index=[0])
            df.to_excel(f, sheet_name=f"{problem} Accuracy", header=None, startrow=row)
            df.to_excel(f, sheet_name=f"{problem} Errors", header=None, startrow=row)
            row += 2
            df = pd.DataFrame(accs)
            df.to_excel(f, sheet_name=f"{problem} Accuracy", startrow=row)
            df = pd.DataFrame(errs)
            df.to_excel(f, sheet_name=f"{problem} Errors", startrow=row)
            row += len(df) + 4
    else:
        # ExcelWriter is so dumb on this
        with pd.ExcelWriter(excel_filename, mode="w")as f:
            df = pd.DataFrame(header, index=[0])
            df.to_excel(f, sheet_name=f"{problem} Accuracy", header=None, startrow=row)
            row += 2
            df = pd.DataFrame(accs)
            df.to_excel(f, sheet_name=f"{problem} Accuracy", startrow=row)
        with pd.ExcelWriter(excel_filename, mode="a", if_sheet_exists="overlay")as f:
            df = pd.DataFrame({"Problem": suff}, index=[0])
            df.to_excel(f, sheet_name=f"{problem} Errors", header=None, startrow=row - 2)
            df = pd.DataFrame(errs)
            df.to_excel(f, sheet_name=f"{problem} Errors", startrow=row)
        row += 34
    return row


def inject_stats_sheets(names: list, problems: list, models: list, excel_filename: str):
    """
    Return the formulae (as a dataframe) for the Excel averages

    Params:
    names (list): a list of the sheets from which to compute averages
    problems (list): the problems (prompt strategies)
    models (list): the models (table headers)
    excel_filename (str): the name of the excel file to dump this to
    """
    def format_str(lx, ix, names, error_mode=None):
        ss = "=AVERAGE('"
        for n in names:
            ss += f"{n} Accuracy'!{lx}{ix}, '"
        ss = ss[:-3] + ")"
        if error_mode == "AccuracyCount":
            ss = "=IFERROR(SUM('"
            for n in names:
                ss += f"{n} Accuracy'!{lx}{ix}, '"
            ss = ss[:-3] + f")/('Failure Counts Accuracy'!D1-'Failure Counts Accuracy'!{lx}{ix}),0)"
        if error_mode == "ErrorCount":
            ss = [f"COUNTIF('{n} Errors'!{lx}{ix}, 100)" for n in names]
            ss = " + ".join(ss)
            ss = "=" + ss
        return ss
    
    def retrieve_dict_for_problem(start_ix, prompt_style, error_mode=False):
        start_ix = start_ix + 4
        dic_accuracy = {_m: {} for _m in ["shots", "delta"] + models + ["name"]}
        dic_errors = {_m: {} for _m in ["shots", "delta"] + models + ["name"]}
        shots = [0, 5, 10, 20, 50, 100]
        if prompt_style == "modus_ponens": 
            shots = [2, 5, 10, 20, 50, 100]
        alphanumeric = ["D", "E", "F", "G", "H", "I", "J", "K"][:len(models)]
        counter = 0
        for s in shots:
            for d in [0, 0.2, 0.45, 0.65, 0.85]:
                dic_accuracy["shots"][counter] = s
                dic_accuracy["delta"][counter] = d
                dic_errors["shots"][counter] = s
                dic_errors["delta"][counter] = d
                for alpha, _m in zip(alphanumeric, models):
                    dic_accuracy[_m][counter] = format_str(alpha, start_ix, names, error_mode=error_mode)
                    dic_errors[_m][counter] = format_str(alpha, start_ix, names, error_mode=error_mode).replace("Accuracy", "Errors")
                dic_accuracy["name"][counter] = f'=B{start_ix}&" (δ="&C{start_ix}&")"'
                dic_errors["name"][counter] = f'=B{start_ix}&" (δ="&C{start_ix}&")"'
                start_ix += 1
                counter += 1
        return dic_accuracy, dic_errors

    row = 0
    for p in problems:
        # Write standard averages
        dic_accuracy, dic_errors = retrieve_dict_for_problem(row, p)
        _ = write_to_excel(excel_filename, p, "Averages", row, dic_accuracy, dic_errors)
        # Write averages without accounting for failed states
        dic_accuracy, dic_errors = retrieve_dict_for_problem(row, p, error_mode="AccuracyCount")
        _ = write_to_excel(excel_filename, p, "Avgs no fails", row, dic_accuracy, dic_errors)
        dic_accuracy, dic_errors = retrieve_dict_for_problem(row, p, error_mode="ErrorCount")
        row = write_to_excel(excel_filename, p, "Failure Counts", row, dic_accuracy, dic_errors, 
                             header={"Problem": p, "blank": "", "len": len(names)})


def print_table_helper(data: dict, accs: dict, errs: dict, model2printname: dict, latex: bool, output_auc: bool, 
                       delta_avg: bool, mode: str, shuffled: bool, problem: str, shots: list):
    """ 
    A thing to print tables to the console. Not entirely necessary but good for eyeballing.
    Data is of the shape model: xaxis_acc/err : value

    ---
    Params
    data (dict): the full dataset
    accs (dict): the subset of accuracies -- likely we only need data but this is how I originally wrote it.
    errs (dict): the subset of errors -- see above
    model2printname (dict): a dictionary of raw keys for models to pretty print names
    latex (bool): output a latex table (this doesn't work because plots are easier to read)
    delta_avg (bool): is it an average over deltas?
    mode (str): the type of prompt
    shuffled (bool): whether this is the shuffled or unshuffled subset
    problem (str): the problem to be worked on
    shots (list): the list of shots
    """
    yaxis = "Shots" if delta_avg else "Delta"
    models = [k for k in data.keys()]

    if latex:
        title_line = f"c|" + "|".join(["c" for _ in models]) + "\\\\ \n"
        title_line += f"{yaxis}\t|" + "\t|".join([model2printname[m] for m in models]) + "\\\\ \n"

        for i in range(len(accs[0])):
            d = shots[i]
            arr = [str(accs[j][i]) for j in range(len(accs))]
            print(f"{d}\t|{'\t|'.join(arr)}")
        print("----- Errors ---------")
        print(title_line)
        for i in range(len(accs[0])):
            d = shots[i]
            #print(f"{d}\t|{errs[0][i]}\t|{errs[1][i]}\t|{errs[2][i]}")
            arr = [str(errs[j][i]) for j in range(len(accs))]
            print(f"{d}\t|{'\t|'.join(arr)}")
        pass

    else:
        title_line = f"{yaxis}\t|" + "\t|".join([model2printname[m] for m in models])
        suff = f"({mode})" if not shuffled else f"({mode}, shuffled)"
        header = f"======== {problem} {suff} ========="
        print(header)
        print("----- Accuracy ---------")
        print(title_line)
        model = models[0]
        for xaxis_key in data[model].keys():
            # 0_shot, 2_shot... etc
            if "acc" not in xaxis_key: continue
            key_name = xaxis_key.split("_")[0]
            performance_row = [str(data[m][xaxis_key]) for m in models]
            print(f"{key_name}\t|{'\t|'.join(performance_row)}")
        print("----- Errors ---------")
        print(title_line)
        for xaxis_key in data[model].keys():
            # 0_shot, 2_shot... etc
            if "acc" in xaxis_key or "preds" in xaxis_key: continue
            key_name = xaxis_key.split("_")[0]
            performance_row = [str(data[m][xaxis_key]) for m in models]
            print(f"{key_name}\t|{'\t|'.join(performance_row)}")


def output_auc_helper(results_dic: dict, model2printname: dict, delta_avg: bool, mode: str, 
                      shuffled: bool, problem: str, shots: list):

    suff = f"({mode})" if not shuffled else f"({mode}, shuffled)"
    header = f"======== {problem} {suff} ========="
    print(header)
    print("----- AUCS ---------")
    models = [m for m in results_dic.keys()]
    for i in range(len(models) - 1):
        a_model = models[i]
        for j in range(i, len(models)):
            if i == j: continue
            b_model = models[j]
            print(f"{model2printname[a_model]} / {model2printname[b_model]}")
            for _, delta_dic in results_dic[a_model].items():
                title_string = "\t|".join([str(delta) for delta in delta_dic.keys()])
                break

            print(title_string)
            for shot, delta_dic in results_dic[a_model].items():
                print(f"{shot}:")
                runing_string = ""
                for delta, preds in delta_dic.items():
                    auc_pred1 = auc(preds[-1], preds[0]) #model A
                    auc_pred2 = auc(results_dic[b_model][shot][delta][0], results_dic[b_model][shot][delta][-1]) # model b
                    runing_string += f"\t| {round(auc_pred1, 2)} {round(auc_pred2, 2)} {round(auc_pred1, 2) - round(auc_pred2, 2)}"
                print(runing_string)


def print_table(problem: str, mode: str, models: list, problem2fname: dict, model2printname: dict,
                latex: bool=False, delta_avg: bool=False, per_ood: bool=False, rename_gpt4o: bool = False,
                shot_avg: bool=False, shuffled: bool=False, punish: bool=False, do_print_table: bool=True,
                output_auc: bool=False, excel_filename: str=None, row: int=1):
    """
    Pretty print a table with performances. LaTeX optional. You can also get the datasets (accuracy and errors), 
    or an Excel file with the datasets. 

    ---
    Params:    
    problem (str): the problem that we are printing for
    mode (str): one of the subproblems (prompt strats; modus ponens, CoT, etc.)
    models (list): a list of the models (top-level directories) for which we will print stuff.
    problem2fname (dict): the dict translating problems to filenames
    model2printname (dict): the dict translating model names to pretty print models
    latex (bool, defaults to False): bool: print that table in latex instead.
    delta_avg (bool, defaults to False): display an averaged out (over shots) performance per delta
    shot_avg (bool, defaults to False): display an averaged out (over deltas) performance per shots. This also allows you to output AUC, so keep it on.
    per_ood (bool, defaults to False): aggregate by delta and then by shot
    rename_gpt4o (bool, defaults to False): whether to rename the GPT-4o entry to the pretty print version for the ablation studies
    shuffled (bool, defaults to False): whether we are doing the shuffled versions or not.
    punish (bool, defaults to False): whether to flip labels when there are parsing errors
    output_auc (bool, defaults to False): stat tests -- disabled because it asks for increasing values???
    do_print_table (bool, defaults to False): whether to print the table to the console
    excel_filename (str, defaults to None): the name of the Excel file to dump predictions to (if not None)
    row (int, defaults to 1): for Excel writing, this thing returns an integer denoting the last row
    """
    if mode == "modus_ponens": 
        shots = [2, 5, 10, 20, 50, 100]
    else:
        # This was an accident from the consolidation and 
        # all preds with shots = 2 are lost... ):
        shots = [0, 5, 10, 20, 50, 100]
    is_cot = "cot" in mode.lower() or "sot" in mode.lower()
    is_vending = "vending" in problem.lower()

    paths = []
    data = {}
    double_data = {}
    for m in models:
        data[m] = {}
        double_data[m] = {}
        is_shuffled = "" if not shuffled else "_shuffled"
        if is_vending:
            if "sum" in problem.lower():
                pass                
        paths.append(lambda d,m=m: f"{m}/{problem}/predictions_{mode.lower()}_{d}_shot{is_shuffled}.json")

    fname = problem2fname[problem]
    accs, errs = {}, {}
    for ix, p in enumerate(paths):
        m = paths[ix](0).split("/")[0].strip()
        do_trace = None #if not is_cot else f"{m}_{mode}_{problem}"
        result = collect(p, shots, is_cot=is_cot, problem=fname, do_trace=do_trace, punish=punish)
        # Shot view (delta average): overall aggregated perf over shots
        if delta_avg:
            for shot, delta_dic in result.items():
                avg_deltas = np.array([v[0] for _, v in delta_dic.items()]).mean()
                avg_errors = np.array([v[-1] for _, v in delta_dic.items()]).mean()
                data[m][f"{shot}_acc"] = round(avg_deltas, 2)
                data[m][f"{shot}_err"] = round(avg_errors, 2)
                #if output_auc:
                #    for k, v in delta_dic.items():
                #        data[m][f"{shot}_preds"][k] = v[1]
        # Delta view (shot average): overall aggregated perf over OOD
        if shot_avg:
            delta_keys = result[[k for k in result.keys()][0]].keys()
            resultT_acc = {k: [] for k in delta_keys}
            resultT_err = {k: [] for k in delta_keys}
            for shot, delta_dic in result.items():
                for delta_key, value in delta_dic.items():
                    resultT_acc[delta_key].append(value[0])
                    resultT_err[delta_key].append(value[-1])
            for delta in resultT_acc.keys():
                avg_deltas = round(np.array(resultT_acc[delta]).mean(), 2)
                avg_errors = round(np.array(resultT_err[delta]).mean(), 2)
                data[m][f"{delta}_acc"] = round(avg_deltas, 2)
                data[m][f"{delta}_err"] = round(avg_errors, 2)
        # Fully unrolled
        if per_ood:
            for shot, delta_dic in result.items():
                for delta, item in delta_dic.items():
                    data[m][f"{shot}_{delta}_acc"] = round(item[0], 2)
                    data[m][f"{shot}_{delta}_err"] = round(item[-1], 2)

        counter = 0
        if "shots" not in errs:
            errs["shots"] = {}
            errs["delta"] = {}
            accs["shots"] = {}
            accs["delta"] = {}
            skip_shots = False
        else:
            skip_shots = True
        _m = model2printname[m]
        if _m not in accs: accs[_m] = {}
        if _m not in errs: errs[_m] = {}
        for shot, delta_dic in result.items():
            for delta, item in delta_dic.items():
                if not skip_shots:
                    accs["shots"][counter] = shot
                    accs["delta"][counter] = delta
                    errs["shots"][counter] = shot
                    errs["delta"][counter] = delta
                accs[_m][counter] = round(item[0], 2)
                errs[_m][counter] = round(item[-1], 2)
                counter += 1

    if do_print_table:
        print_table_helper(data, accs, errs, model2printname, latex, output_auc, delta_avg, 
                           mode, shuffled, problem, shots)
    if excel_filename is not None:
        suff = f"({mode})" if not shuffled else f"({mode}, shuffled)"
        if rename_gpt4o:
            accs["GPT-4o (random exemplars)"] = deepcopy(accs["GPT-4o"])
            errs["GPT-4o (random exemplars)"] = deepcopy(errs["GPT-4o"])
            del accs["GPT-4o"]
            del errs["GPT-4o"]
        row = write_to_excel(excel_filename, suff, problem, row, accs, errs)

    return row, data


