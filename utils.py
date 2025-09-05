import json


def to_file(arr: list, fname: str, dumpall=False):
    """Dump the predictions array (`arr`) to `fname`
    """
    if dumpall:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(json.dumps(arr, ensure_ascii=False))
    else:
        with open(fname, "w", encoding="utf-8") as f:
            _ = [f.write(json.dumps(l, ensure_ascii=False) +"\n") for l in arr]


def compute_acc(arr: list, labels: list = [0,1], is_apo=False):
    """Compute accuracy for a set of predictions. Only used in APO.
    """
    preds = []
    failure_rate = 0
    for entry in arr:
        pred = None
        if entry["Response"][0].isnumeric():
            pred = int(entry["Response"][0])
        else:
            failure_rate += 1
            continue
        if pred not in labels:
            failure_rate += 1
            continue
        preds.append((entry["Label"], pred))

    # Smaller models like Phi-2 have incredibly high failure rates.
    if preds == []:
        accuracy = 0
        frate = 100
    else:
        accuracy = sum([x[0] == x[1] for x in preds])*100./len(preds)
        frate = failure_rate*100./len(preds)
    return round(accuracy, 2), frate
