
def print_a_table_per_prompt():
    # This one is for the shot/slope comparison per prompt:
    # Go to the excel, copy-paste, say, L23-26 for cols P-W, and paste them as in the example.
    # Then run this baby.
    arrs = ["""14.63230816	-0.293732143	12.69335	-0.598270833	5.862	-0.598125	5.714427438	-0.250833333
    31.35484524	31.35484524	47.32429167	47.32429167	56.02416667	56.02416667	53.19437169	53.19437169
    25.78533828	0.754323808	22.43102117	0.848109679	10.12847231	0.852338083	10.02132831	0.515528347"""
    ,
    """3.349464286	-0.483166667	1.504785714	-1.080208333	6.212357143	-0.403541667	9.788384354	-0.163492063
    61.18045833	61.18045833	63.66891667	63.66891667	55.92875	55.92875	51.28568122	51.28568122
    5.885236252	0.761069028	3.014642121	1.561902921	11.08068503	0.691249121	21.67072363	0.308916092"""
    ,
    """11.60805	-0.553029762	13.46592857	-0.100625	12.93928571	-0.265625	11.6508322	-0.290492725
    34.37066071	34.37066071	47.82625	47.82625	43.625	43.625	47.68784392	47.68784392
    21.32295217	0.810340136	24.87028413	0.290320106	23.17341315	0.39356808	22.65808398	0.654435413"""							
    ,
    """6.732428571	-0.494583333	2.290285714	-1.136041667	5.084	-0.620833333	10.28692857	-0.084546958
    57.36083333	57.36083333	62.555	62.555	57.085	57.085	51.62845899	51.62845899
    12.4836925	0.744892796	4.180357341	1.634200769	10.26759628	1.000622549	21.47787875	0.381945242"""
    ,
    """4.156943197	-0.803136905	1.764328231	-3.132671627	0.833993197	-1.517757937	8.155479365	-1.010851852
    50.78859325	50.78859325	60.61190675	60.61190675	49.73230159	49.73230159	39.24474074	39.24474074
    7.275998308	1.65868383	4.463296594	4.462844024	1.742370841	2.235497372	15.39792309	1.433124942"""						
    ,
    """2.087635714	0.012037698	3.642664762	-0.734677778	-0.124387755	-0.187916667	1.983611111	0.313
    21.700625	21.700625	26.64325556	26.64325556	27.1277381	27.1277381	22.13966667	22.13966667
    4.552447034	1.020092778	7.701120763	1.056260533	4.349178869	0.333828629	5.088108669	0.695120258"""
    ]

    aggregates = {
        "shots": [
            (9.1, 2.9),
            (5.0, 1.9),
            (5.2, 1.9),
            (12.3, 3.1,), 
            (6.1, 2.0),
            (3.6, 2.3),
            (0.7, 2.2)
        ],
        "deltas": [
            (-0.4, 0.4),
            (-0.6, 0.4),
            (-0.5, 0.6),
            (-0.3, 0.3),
            (-0.6, 0.7),
            (-1.5, 1.9),
            (-0.2, 0.7)
        ]
    }

    aggregates_ablation = {
        "shots": [
            (8.3, 3.9),
            (4.4, 2.2),
            (4.5, 2.4),
            (11, 4.6,), 
            (5.4, 2.6),
            (3.3, 2.4),
            (1.6, 2.2)
        ],
        "deltas": [
            (-0.4, 0.4),
            (-0.5, 0.5),
            (-0.5, 0.6),
            (-0.2, 0.3),
            (-0.5, 0.7),
            (-1.4, 1.9),
            (0.0, 0.6)
        ]
    }

    gpt4o_baseline = {
        "shots": {
            "slope": [9, 5.0, 5.2, 12.3, 6.1, 3.6, 0.7],
            "acc": [47, 58, 58, 43, 57, 50, 24],
            "sigma": [17, 2, 9, 23, 11, 7, 4]
        },
        "delta": {
            "slope": [-0.4, -0.6, -0.5, -0.3, -0.6, 1.5, -0.2],
            "acc": [47, 58, 58, 43, 57, 50, 25],
            "sigma": [1, 1, 1, 1, 1, 2, 0]
        },
    }

    headers = ["Modus Ponens", "Description", "DE", "\\textbf{Word Salad}", "APO", "CoT", "\\textbf{SoT}"]

    gray = "" #"\\cellcolor{gray!25}"
    blue = "" #"\\cellcolor{RoyalPurple!20}"

    all_shots, all_deltas = [], []
    for ix, (h, a) in enumerate(zip(headers, arrs)):
        deltas, accs, stds = a.split("\n")
        shot_str = []
        delta_str = []
        for i, (_slope, _acc, _sd) in enumerate(zip(
            deltas.split("\t"), accs.split("\t"), stds.split("\t")
        )):
            slope = round(float(eval(_slope)), 1)
            acc = int(eval(_acc))
            sd = int(eval(_sd))

            colour = ""
            if i%2 == 0:
                if slope > gpt4o_baseline["shots"]["slope"][ix]:
                    colour = gray
                shot_str.append(colour + str(slope))
                shot_str.append("&")

                if acc >= gpt4o_baseline["shots"]['acc'][ix]:
                    colour = gray
                if sd > gpt4o_baseline["shots"]['sigma'][ix]:
                    colour = blue

                if sd != 0:
                    shot_str.append(colour + str(acc)+"$\\pm$"+str(sd))
                else:
                    shot_str.append(colour + str(acc))
                shot_str.append("&")
            else:
                if slope > gpt4o_baseline["delta"]["slope"][ix]:
                    colour = gray
                delta_str.append(colour + str(slope))
                delta_str.append("&")

                if acc >= gpt4o_baseline["delta"]['acc'][ix]:
                    colour = gray
                if sd > gpt4o_baseline["delta"]['sigma'][ix]:
                    colour = blue

                if sd != 0:
                    delta_str.append(colour + str(acc)+"$\\pm$"+str(sd))
                else:
                    delta_str.append(colour + str(acc))
                delta_str.append("&")

        all_shots.append(h + "&" +" ".join(shot_str[:-1])   + f"& {aggregates['shots'][ix][0]}$\\pm${aggregates['shots'][ix][-1]}"+ "\\\\")
        all_deltas.append(h + "&" +" ".join(delta_str[:-1]) + f"& {aggregates['deltas'][ix][0]}$\\pm${aggregates['deltas'][ix][-1]}"+ "\\\\")

    for j, s in enumerate(all_shots):
        if j == 0: print("\\textbf{Shots} & " + s)
        else:
            print("&" + s)
    print("\\midrule")
    for j, s in enumerate(all_deltas):
        if j == 0: print("\\textbf{$\\delta$} & " + s)
        else:
            print("&" + s)

