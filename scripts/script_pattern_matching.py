#!/usr/bin/env python
# coding: utf-8

# # Data Collection - SLM Version
# This means no chatML API

# In[1]:
import time
import numpy as np
from tqdm import tqdm
import random
import json
from glob import glob
from transformers import pipeline
import transformers
transformers.logging.set_verbosity_error()
import argparse

class LLMClient():
    def __init__(self, params, model_id, is_dumb=False):
        self._pipeline = pipeline(
            "text-generation",
            model=model_id,
            device_map="auto",
        )
        # OpenAI's temperature range is [0,2] for some dumb reason.
        # For backwards compat with the GPT-4 code, we halve it if it is not dumb.
        self._is_dumb = is_dumb 
        self._params = params

    def send_request(self, assembled_prompt):
        #eos_token_id=terminators
        outputs = self._pipeline(assembled_prompt, do_sample=False,
                           #temperature=self._params["temperature"] if self._is_dumb else self._params["temperature"]/2, 
                           max_new_tokens=self._params["max_tokens"],
                           pad_token_id = self._pipeline.tokenizer.eos_token_id)
        return outputs

    def update_params(self, params):
        for k, v in params.items():
            self._params[k] = v


def parse_them_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--root", type=str)
    args = parser.parse_args()
    return args

args = parse_them_args()

# o1 has only one temperature and it's called max_completion_tokens
MODEL = args.model ##o-2024-05-13-chat-completions" #"dev-gpt-o1-preview"
root_out = args.root

tkey = "max_tokens" if "o1" not in MODEL else "max_completion_tokens"
is_async = False
async_batch_size = None

tokens = 5000 if "o1" in MODEL else 3
cot_tokens = 5000 if "o1" in MODEL else 1024
max_apo_tokens = 5000 if "o1" in MODEL else 512
MAX_DATA_TO_EVAL = 1000

params = {tkey: tokens,} 
if "o1" not in MODEL:
    params["temperature"] = 0.0
gpt4 = LLMClient(params, MODEL)



def to_file(arr, fname, dumpall=False):
    if dumpall:
        with open(fname, "w", encoding="utf-8") as f:
            f.write(json.dumps(arr, ensure_ascii=False))
    else:
        with open(fname, "w", encoding="utf-8") as f:
            _ = [f.write(json.dumps(l, ensure_ascii=False) +"\n") for l in arr]


def compute_acc(arr, labels=[0,1], is_apo=False):
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


def get_gpt4_response(model, assembled_prompt, debug=False):
    try:
        resp = model.send_request(assembled_prompt)
    except:
        return "FAIL"
    if type(assembled_prompt) == list:
        return resp[0]["generated_text"][-1]["content"]
    return resp[0]["generated_text"][len(assembled_prompt):]


def get_predictions(dataset, max_calls, problem="parity", max_exemplars=10, use_desc="", desc="", shuffle_exemplars=False,
                    start_at=None, is_cot=False, stream=False, max_token_override=None, fname=None, debug=False):
    predictions = []
    max_points = min(max_calls, len(dataset))
    if debug: print(f"Calling for max_points = {max_points}")

    if is_cot:
        gpt4.update_params({tkey: cot_tokens})
    else:
        gpt4.update_params({tkey: tokens})
    if max_token_override is not None:
        gpt4.update_params({tkey: max_token_override})

    batch = []
    _batch = []
    if start_at is None:
        start_at = 0
    for i in tqdm(range(start_at, max_points), desc=desc):
        point = dataset[i]
        prediction = {k:v for k,v in point.items()}

        if debug: print(f'Getting prompt for {problem}: entry: {point["Entry"]}')

        if problem == "parity":
            prompt = get_prompt_for_parity(point["Entry"], use_desc=use_desc, max_exemplars=max_exemplars, shuffle_exemplars=shuffle_exemplars, is_cot=is_cot)
        elif problem == "pattern_matching":
            prompt = get_prompt_for_pattern_matching(point["Entry"], use_desc=use_desc, max_exemplars=max_exemplars, shuffle_exemplars=shuffle_exemplars, is_cot=is_cot)
        elif problem == "vending_machine":
            prompt = get_prompt_for_vending_machine(point["Entry"], use_desc=use_desc, max_exemplars=max_exemplars, shuffle_exemplars=shuffle_exemplars, is_cot=is_cot)
        elif problem == "vending_machine_with_sum":
            prompt = get_prompt_for_vending_machine_with_sum(point["Entry"], use_desc=use_desc, max_exemplars=max_exemplars, shuffle_exemplars=shuffle_exemplars, is_cot=is_cot)
        elif problem == "hamiltonian":
            prompt = get_prompt_for_hamiltonian(point["Entry"], point["Path"], use_desc=use_desc, max_exemplars=max_exemplars, shuffle_exemplars=shuffle_exemplars, is_cot=is_cot)
        else:
            raise "Problem \"{}\" not implemented!".format(problem)

        if debug: print(f"Got prompt: {prompt}")
        if is_async:
            batch.append(prompt)
            _batch.append(prediction)
            if len(batch) == async_batch_size:
                actual_responses = get_gpt4_response(gpt4, batch, debug=debug)
                _tmp_preds = []
                for pred, actual_response in zip(_batch, actual_responses):
                    pred["Response"] = actual_response
                    predictions.append(pred)
                    _tmp_preds.append(pred)
                if stream:
                    with open(fname, "a", encoding="utf-8") as f:
                        for pred in _tmp_preds:
                            f.write(json.dumps(pred, ensure_ascii=False) + ",\n")
                batch, _batch = [], []
        else:
            actual_response = get_gpt4_response(gpt4, prompt, debug=debug)
            if debug: print(f"Got response: {actual_response}")
            prediction["Response"] = actual_response
            predictions.append(prediction)
            if stream:
                with open(fname, "a", encoding="utf-8") as f:
                    f.write(json.dumps(prediction, ensure_ascii=False) + ",\n")                

    if batch != []:
        actual_responses = get_gpt4_response(gpt4, batch, debug=debug)
        _tmp_preds = []
        for pred, actual_response in zip(_batch, actual_responses):
            pred["Response"] = actual_response
            predictions.append(pred)
            _tmp_preds.append(pred)
        if stream:
            with open(fname, "a", encoding="utf-8") as f:
                for pred in _tmp_preds:
                    f.write(json.dumps(pred, ensure_ascii=False) + ",\n")

    return predictions


def call_and_solve_for(problem, suff, dataset, SHOTS=[2, 5, 10, 20, 50, 100], 
                       zero_shot_prompt = None, few_shot_prompt = None, shuffle_exemplars=False,
                       do_ood=False, out_dir_root="", is_cot=False, stream=True):
    for shots in SHOTS:

        start_at = None
        start_delta = None
        if type(shots) == tuple:
            start_at = shots[-1]
            if len(shots) == 3: # OD
                start_delta = shots[1]
            shots = shots[0]

        system_prompt = ""
        if shots == 0 and zero_shot_prompt is not None:
            system_prompt = zero_shot_prompt
        if shots != 0 and few_shot_prompt is not None:
            system_prompt = few_shot_prompt

        if not do_ood:
            preds_raw_id = get_predictions(dataset, MAX_DATA_TO_EVAL, max_exemplars=shots, problem=problem, use_desc=system_prompt,
                                            desc="ID {}".format(shots), is_cot=is_cot, start_at=start_at, shuffle_exemplars=shuffle_exemplars,
                                           stream=stream, fname="{}preds_raw_id_{}_shot_{}.json".format(out_dir_root, shots, suff))
            if not stream:
                to_file(preds_raw_id, "{}preds_raw_id_{}_shot_{}.json".format(out_dir_root, shots, suff), dumpall=False)
        else:
            oods_negs = {}

            for delta in [0.2, 0.45, 0.65, 0.85]: 
                if start_delta is not None and delta < start_delta:
                    continue
                if start_delta != delta:
                    start_at = None
                corpus = dataset[0]["delta_{}".format(str(delta))]
                preds_raw_od = get_predictions(corpus, MAX_DATA_TO_EVAL, max_exemplars=shots, problem=problem, use_desc=system_prompt,
                                               desc="OD {} | {}".format(shots, str(delta)), is_cot=is_cot, start_at = start_at, shuffle_exemplars=shuffle_exemplars,
                                               stream=stream, fname="{}tmp_{}_preds_raw_od_{}_shot_{}.json".format(out_dir_root, str(delta), shots, suff))
                oods_negs["delta_{}".format(str(delta))] = preds_raw_od
                # Reenable this to have checkpoints if needed
                #if not stream:
                #    to_file(oods_negs, "{}tmp_{}_preds_raw_od_{}_shot_{}.json".format(out_dir_root, str(delta), shots, suff), dumpall=True)
            to_file(oods_negs, "{}preds_raw_od_{}_shot_{}.json".format(out_dir_root, shots, suff), dumpall=True)


# In[3]:


def apo(initial_prompt, dataset, problem="parity", beam_width=4, search_depth=6, max_token_override=None, debug=False):
    b0 = [(initial_prompt, 0)]
    for i in range(search_depth):
        candidates = []
        # Our train set is 4k -- model will only see 64.
        subset = random.sample(dataset[2000:], k=64)
        if debug: print(f"b0: {b0}")
        for prompts, scores in b0:
            candidates += expand(prompts, subset, problem, max_token_override=max_token_override, debug=debug)
        # Sampling to avoid overrun
        _cands = random.sample(candidates, k=8) if len(candidates) > 8 else candidates
        if debug: print(f"chosen candidates {_cands}")
        if _cands != []:
            b0 += select(_cands, dataset, problem, beam_width)
        if debug: break
    b0.sort(key=lambda x: x[-1], reverse=True)
    return b0


def expand(p_candidate, subset, problem, max_token_override=None, max_errors=4, debug=False):
    if debug: print("Calling for predictions")
    resps = get_predictions(subset, len(subset), problem=problem, max_exemplars=2, use_desc=p_candidate, debug=debug, desc="APO")
    errors = []
    accuracy = compute_acc(resps, is_apo=True)[0]
    if debug: print(f"Called {len(subset)} predictions and got accuracy {accuracy}")
    for entry in resps:
        if entry["Response"][0].isnumeric():
            pred = int(entry["Response"][0])
            if pred != entry["Label"]:
                errors.append(entry)
    errors = random.sample(errors, k=min(len(errors), max_errors))
    # Minor hack (not in the paper): if no errors, we've converged
    if debug: print(f"Selected {len(errors)}")
    if errors == []:
        successors = [p_candidate]
    else:
        successors = gradient_and_edit(p_candidate, errors, max_token_override=max_token_override, debug=debug)
    return successors


def gradient_and_edit(p_candidate, errors, max_token_override=None, num_reasons=4, edits_per_gradient=1, num_mc_samples=2, debug=False):
    # Defaults from the paper
    def gradient_prompt(p, e, f): 
        resp = [{"role": "user", "content": f"I'm trying to write a zero-shot classifier prompt.\nMy current prompt is:\n\"{p}\"\nBut this prompt gets the following examples wrong:\n{e}\ngive {f} reasons why the prompt could have gotten these examples wrong.\nWrap each reason with <START> and <END>"}]
        if is_async:
            resp = [resp]
        return resp

    def edition_prompt(p, e, g, n):
        resp = [{"role": "user", "content": f"I'm trying to write a zero-shot classifier.\n My current prompt is:\n\"{p}\"\nBut it gets the following examples wrong:\n{e}\nBased on these examples the problem with this prompt is that:\n{g}\nBased on the above information, I wrote {n} different improved prompts.\nEach prompt is wrapped with <START> and <END>.\nThe {n} new prompts are:"}]
        if is_async:
            resp = [resp]
        return resp
    
    def mc_prompt(p):
        resp = [{"role": "user", "content": f"Generate a variation of the following instruction while keeping the semantic meaning.\nInput: {p}\nOutput:"}]
        if is_async:
            resp = [resp]
        return resp

    gpt4.update_params({tkey: max_apo_tokens if max_token_override is None else max_token_override})
    
    # Direct call to model, postprocess "gradient"
    if "Path" not in errors[0]:
        error_string = "\n - ".join([p["Entry"] + ": " + str(p["Label"]) for p in errors])
    else:
        # Hamiltonian has a sliiiightly different signature
        error_string = "\n - ".join([f"{p['Entry']} | path: {p['Path']} : {p['Label']}" for p in errors])
    prompt = gradient_prompt(p_candidate, error_string, str(num_reasons))
    if debug: print(f"Prompt is \n{prompt}\n for the gradient step")
    response = get_gpt4_response(gpt4, prompt)
    if is_async:
        response = response[0]
    if debug: print(f"ot \n{response}\n from the gradient step")
    
    # Edit the prompt -- one edit per gradient.
    edited_prompts = []
    # Note: the original paper doesn't specify parsing. We need to keep it or else this algorithm won't work.
    response_processed = [r.replace("<END>", "").strip() for r in response.split("<START>") if "<END>" in r]
    for g in response_processed:
        prompt = edition_prompt(p_candidate, error_string, g.strip(), str(edits_per_gradient))
        response = get_gpt4_response(gpt4, prompt)
        if is_async:
            response = response[0]
        if "4o" in MODEL or "turbo" in MODEL:
            edited_prompts.append(response.strip())
        else:
            # this will inject boilerplate from the model in omni
            edited_prompts.append(response.strip().split("\n")[0].strip())
    # Do MC search
    # Two candidates per instruction:
    if debug: print(f"Here are the edited prompts:\n{edited_prompts}")
    candidates = []
    for c in edited_prompts:
        # I would like to make this a batch call but I don't want to alter the original work.
        for _ in range(num_mc_samples):
            response1 = get_gpt4_response(gpt4, mc_prompt(c))
            if is_async:
                response1 = response1[0]
            # Same as before: no parsing in paper
            response_processed = [r.replace("<END>", "").strip() for r in response1.split("<START>") if "<END>" in r]
            if response_processed != []:
                response1 = response_processed[0]
            candidates.append(response1)

    gpt4.update_params({tkey: tokens})

    return candidates


def select(candidates, dataset, problem, beam_size, B=12):
    # Paper states that B in 12-50 keeps it steady.
    # This is a very confusing and not-very-well-written algorithm.
    # We'll implement it verbatim from the paper though.
    S = [(p, 0) for p in candidates]
    old_s = [s for s in S]

    def get_n(i, T):
        # What is T lmao 
        # From the below, our maximum number of iterations (n) cannot be larger than T
        # Likewise, B gets too degenerate when close to T (T < B, otherwise you'll just get zeroes)
        left = 1/(0.5*sum([1/j for j in range(2, T + 1)]))
        right = (B-T)/(T + 1 - i)
        return int(np.ceil(left*right))

    # Says 1... n - 1 for n prompts; index shift python
    m = min(len(candidates) - 2, beam_size + 1)
    for i in range(1, m):
        subset = random.sample(dataset, k=get_n(i, m))
        # Original paper isn't very clear here (i is not defined; We'll evaluate all prompts)
        S_exp = []
        for prompt, _ in S:
            resps = get_predictions(subset, len(subset), problem=problem, max_exemplars=2, use_desc=prompt)
            accuracy = compute_acc(resps, is_apo=True)[0] 
            S_exp.append((prompt, accuracy))
        S_exp.sort(key = lambda x: x[-1], reverse=True) # Decreasing
        S = [p for p in S_exp[:-1]]
    # We also need to address this corner case, not addressed in the paper
    if S == []:
        S = old_s
    return [S[0]]


def insert_into(arr, x):
    line = random.choice(arr)
    for _x in x:
        index = random.choice(range(len(line.split(" "))))
        line = line.split(" ")
        line = " ".join(line[:index]).strip() + f" {_x} " + " ".join(line[index:]).strip() + "\n"
    return line


# # Pattern Matching Automata

# ## Shared functions

# In[4]:


corpus_id = [json.loads(l) for l in open("Pattern_Matching/corpus_pattern_matching_id_4k_new.json", "r", encoding="utf-8").readlines()]
corpus_id_test = [json.loads(l) for l in open("Pattern_Matching/corpus_pattern_matching_id_4k_new_test.json", "r", encoding="utf-8").readlines()]
corpus_od_test_w_negs = [json.loads(l) for l in open("Pattern_Matching/corpus_pattern_matching_ood_4k_new_test.json", "r", encoding="utf-8").readlines()]

problem = "pattern_matching"
out_dir_root = f"{root_out}/Pattern_Matching/"

def get_prompt_for_pattern_matching(point, use_desc="", max_exemplars=10, shuffle_exemplars=False, is_cot=False):
    pattern = "abcabb"

    def get_match(c, tally):
        if tally == "":
            return pattern[0] == c
        elif pattern == tally:
            return True
        elif len(tally) > len(pattern):
            return False
        else:
            return pattern[len(tally)] == c

    def build_cot_exemplar(ex, lab):
        has_pattern = str(lab) == "1" # Need this for noise as opposed to count % 2 == 0
        exemplar = "\n{}: \n".format(ex)
        exemplar += "Let's think and solve this step-by-step. "
        if type(use_desc) == str:
            exemplar += "We read the string character-by-character and keep a tally:\n"
        else:
            exemplar += f"{' '.join(random.sample(use_desc, k=2))}:\n"
        tally = ""
        if len(ex) >= len(pattern):
            for i, c in enumerate(ex):
                is_match = get_match(c, tally)
                if type(use_desc) == str:
                    exemplar += "We read \"{}\". It is {}. ".format(c, "not a match" if not is_match else "a match")
                else:
                    match_str = "not a match" if not is_match else "a match"
                    exemplar += f"{' '.join(random.sample(use_desc, k=2))} \"{c}\". {' '.join(random.sample(use_desc, k=2))} {match_str}. "
                if is_match:
                    tally += c
                    if type(use_desc) == str:
                        exemplar += "Our tally is: {}. ".format(tally)
                    else:
                        exemplar += f"{' '.join(random.sample(use_desc, k=1))}: {tally}. "
                    if tally == pattern and lab == "1":
                        if type(use_desc) == str:
                            exemplar += "Our tally matches the pattern. "
                        else:
                            exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
                        break
                else:
                    if tally != "":
                        if type(use_desc) == str:
                            exemplar += "We clear our tally. "
                        else:
                            exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
                    tally = ""
                if type(use_desc) == str:
                    exemplar += "Now we move to the next character.\n"
                else:
                    exemplar += f"{' '.join(random.sample(use_desc, k=1))}.\n"
            if lab == "0" or i == len(ex) - 1:
                if type(use_desc) == str:
                    exemplar += "We have reached the end of the string. "
                else:
                    exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
        else:
            if type(use_desc) == str:
                exemplar += "The length of the example is too short. "
            else:
                exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "

        if type(use_desc) == str:
            exemplar += "The pattern \"{}\" is {}.\n".format(pattern, "in the string" if has_pattern else "is not in the string")
        else:
            pat_in_str = "in the string" if has_pattern else "is not in the string"
            exemplar += f"{' '.join(random.sample(use_desc, k=2))} \"{pattern}\" {random.choice(use_desc)} {pat_in_str}.\n"
        exemplar += "So the answer is {}".format(lab)
        return exemplar

    prompt = []
    base_prompt = use_desc if type(use_desc) == str else " ".join(random.sample(use_desc, k=random.randint(15, 40)))
    if base_prompt != "":
         prompt.append({"role": "system", "content": base_prompt})
    exemplars = corpus_id[:max_exemplars]
    if shuffle_exemplars:
        random.shuffle(exemplars)
    for i in range(len(exemplars)):
        prompt.append({"role": "user", "content": corpus_id[i]["Entry"] + ": "})
        ex = str(corpus_id[i]["Label"])
        if is_cot:
            ex = build_cot_exemplar(corpus_id[i]["Entry"], str(corpus_id[i]["Label"]))
        prompt.append({"role": "assistant", "content": ex})
    user_str = f"{point}: "
    prompt.append({"role": "user", "content": user_str})
    return prompt

def get_prompt_for_pattern_matching_completion(point, use_desc="", max_exemplars=10, shuffle_exemplars=False, is_cot=False):
    pattern = "abcabb"

    def get_match(c, tally):
        if tally == "":
            return pattern[0] == c
        elif pattern == tally:
            return True
        elif len(tally) > len(pattern):
            return False
        else:
            return pattern[len(tally)] == c

    def build_cot_exemplar(ex, lab):
        has_pattern = lab == "1" # Need this for noise as opposed to count % 2 == 0
        exemplar = "\n".format(ex)
        exemplar += "Let's think and solve this step-by-step. "
        if type(use_desc) == str:
            exemplar += "We read the string character-by-character and keep a tally:\n"
        else:
            exemplar += f"{' '.join(random.sample(use_desc, k=1))}:\n"
        tally = ""
        if len(ex) >= len(pattern):
            for i, c in enumerate(ex):
                is_match = get_match(c, tally)
                if type(use_desc) == str:
                    exemplar += "We read \"{}\". It is {}. ".format(c, "not a match" if not is_match else "a match")
                else:
                    match_str = "not a match" if not is_match else "a match"
                    exemplar += f"{' '.join(random.sample(use_desc, k=1))} \"{c}\". {' '.join(random.sample(use_desc, k=1))} {match_str}. "
                if is_match:
                    tally += c
                    if type(use_desc) == str:
                        exemplar += "Our tally is: {}. ".format(tally)
                    else:
                        exemplar += f"{' '.join(random.sample(use_desc, k=1))}: {tally}. "
                    if tally == pattern and lab == "1":
                        if type(use_desc) == str:
                            exemplar += "Our tally matches the pattern. "
                        else:
                            exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
                        break
                else:
                    if tally != "":
                        if type(use_desc) == str:
                            exemplar += "We clear our tally. "
                        else:
                            exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
                    tally = ""
                if type(use_desc) == str:
                    exemplar += "Now we move to the next character.\n"
                else:
                    exemplar += f"{' '.join(random.sample(use_desc, k=1))}.\n"
            if lab == "0" or i == len(ex) - 1:
                if type(use_desc) == str:
                    exemplar += "We have reached the end of the string. "
                else:
                    exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "
        else:
            if type(use_desc) == str:
                exemplar += "The length of the example is too short. "
            else:
                exemplar += f"{' '.join(random.sample(use_desc, k=1))}. "

        if type(use_desc) == str:
            exemplar += "The pattern \"{}\" is {}.\n".format(pattern, "in the string" if has_pattern else "is not in the string")
        else:
            pat_in_str = "in the string" if has_pattern else "is not in the string"
            exemplar += f"{' '.join(random.sample(use_desc, k=1))} \"{pattern}\" {random.choice(use_desc)} {pat_in_str}.\n"
        exemplar += f"So the answer is {lab}\n"
        return exemplar

    prompt = ""
    base_prompt = use_desc if type(use_desc) == str else " ".join(random.sample(use_desc, k=random.randint(15, 40)))
    if base_prompt != "":
         prompt += base_prompt + "\n"
    exemplars = corpus_id[:max_exemplars]
    if shuffle_exemplars:
        random.shuffle(exemplars)
    for i in range(len(exemplars)):
        prompt += corpus_id[i]["Entry"] + ": "
        ex = str(corpus_id[i]["Label"])
        if is_cot:
            ex = build_cot_exemplar(corpus_id[i]["Entry"], str(corpus_id[i]["Label"]))
        prompt += ex
    user_str = f"{point}: "
    prompt += user_str
    return prompt


# ## Modus Ponens

# In[5]:
'''

suffix = f"modus_ponens_{problem}"
#0
call_and_solve_for(problem=problem, suff=suffix, dataset=corpus_id_test, out_dir_root=out_dir_root, shuffle_exemplars=False)
call_and_solve_for(problem=problem, suff=suffix, dataset=corpus_od_test_w_negs, do_ood=True, out_dir_root=out_dir_root, shuffle_exemplars=False)


# In[6]:


suffix = f"modus_ponens_{problem}_shuffled"

call_and_solve_for(problem=problem, suff=suffix, dataset=corpus_id_test, out_dir_root=out_dir_root, shuffle_exemplars=True)
call_and_solve_for(problem=problem, suff=suffix, dataset=corpus_od_test_w_negs, do_ood=True, out_dir_root=out_dir_root, shuffle_exemplars=True)


# ## With description

# In[7]:


parity_desc = "This is a pattern matching task. The strings in this task are generated from a probabilistic automaton.\n"
parity_desc += "Each string is labelled 0 or 1 depending on whether the pattern \"abcabb\" is (1) or is not (0) in the string.\n"
parity_desc += "Your job is to learn what is the likelihood of a string to be labelled 0 or 1, and output the correct label.\n"
parity_desc += "In the limit where the automaton is deterministic, if the pattern is present in the string, the label is always 1. Else, it is 0.\n"

parity_desc_zero_shot = parity_desc + "Give your answer as a single integer, and your reasoning in a new line.\n"
parity_desc_zero_shot += "For example:\naabcabbb:1\nThe pattern is present in the string.\n\n"

parity_desc += "Given the data below, determine what is the most likely label for the given string and output ONLY the label.\n"
parity_desc += "Data:\n\n"

suff = f"w_description_{problem}"

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],  
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)

suff = f"w_description_{problem}_shuffled"

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],  
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)


# ## Word salad

# In[8]:


parity_desc = [l.strip() + "." for l in open("words.txt", "r", encoding="utf-8").readlines()[0].split(".")]
parity_desc_zero_shot = parity_desc

suff = f"word_salad_{problem}"

call_and_solve_for(problem, suff=suff, dataset=corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)
call_and_solve_for(problem, suff=suff, dataset=corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100], #0, 2, 5, 10
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)


suff = f"word_salad_{problem}_shuffled"

call_and_solve_for(problem, suff=suff, dataset=corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)
call_and_solve_for(problem, suff=suff, dataset=corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100], #0, 2, 5, 10
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)


# ## CoT

# In[9]:


parity_desc = "This is a pattern matching task. The strings in this task are generated from a probabilistic automaton.\n"
parity_desc += "Each string is labelled 0 or 1 depending on whether the pattern \"abcabb\" is (1) or is not (0) in the string.\n"
parity_desc += "Your job is to learn what is the likelihood of a string to be labelled 0 or 1, and output the correct label.\n"
parity_desc += "In the limit where the automaton is deterministic, if the pattern is present in the string, the label is always 1. Else, it is 0.\n"

parity_desc_zero_shot = parity_desc + "Give your answer as a single integer, and your reasoning in a new line.\n"
parity_desc_zero_shot += "For example:\naabcabbb:1\nThe pattern is present in the string.\n\n"

parity_desc += "Given the data below, determine what is the most likely label for the given string and output ONLY the label.\n"
parity_desc += "Data:\n\n"

gpt4.update_params({tkey: cot_tokens})


# In[ ]:


suffix = f"CoT_{problem}"

call_and_solve_for(problem, suffix, corpus_id_test, SHOTS=[0, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=True) 

call_and_solve_for(problem, suffix, corpus_od_test_w_negs, SHOTS=[0, 5, 10, 20, 50, 100], 
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=True)


# In[ ]:


suffix = f"CoT_{problem}_shuffled"

call_and_solve_for(problem, suffix, corpus_id_test, SHOTS=[0, 5, 10, 20, 50, 100], #0, 5, 10, 20, 50, 100
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=True) #790

call_and_solve_for(problem, suffix, corpus_od_test_w_negs, SHOTS=[0, 5, 10, 20, 50, 100], #0, 5, 10, 20 #17
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=True)

'''
# ## Salad-of-Thought

# In[ ]:


parity_desc = list(set([l.strip() + "." for l in open("words.txt", "r", encoding="utf-8").readlines()[0].split(".")]))
parity_desc_zero_shot = parity_desc


suffix = f"SoT_{problem}"
gpt4.update_params({tkey: cot_tokens})

call_and_solve_for(problem, suffix, corpus_id_test, SHOTS=[0, 5, 10, 20, 50, 100], #0, 
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=True)

call_and_solve_for(problem, suffix, corpus_od_test_w_negs, SHOTS=[0, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=True)


# In[ ]:


parity_desc = list(set([l.strip() + "." for l in open("words.txt", "r", encoding="utf-8").readlines()[0].split(".")]))
parity_desc_zero_shot = parity_desc


suffix = f"SoT_{problem}_shuffled"
gpt4.update_params({tkey: cot_tokens})

call_and_solve_for(problem, suffix, corpus_id_test, SHOTS=[0, 5, 10, 20, 50, 100], #0, 
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=True)

call_and_solve_for(problem, suffix, corpus_od_test_w_negs, SHOTS=[0, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=True)


# ## Automata

# In[ ]:


ecm_str_desc = "def pattern_matching(P, is_neg=False, max_length = 24_000):\n    \"\"\"\n    Automaton generating strings for the pattern matching problem:\n    L = {w \in {a, b, c}* : \exists x, y \in {a, b, c}* (w = x abcabb y)}.\n    \"\"\"\n    # Transition matrix for the automaton\n    # T[i][j] means what you write on the tape when in Si and transitioning to Sj\n    # When there's a comma, random choices are needed.\n    # S0, # S1, # S2, # S3, # S4, # S5, # S6, # Sf\n    T = [[\"b,c\", \"a\",  None, None, None, None, None],   # S0\n         [\"c\",   \"a\",  \"b\",  None, None, None, None],   # S1\n         [\"b\",   \"a\",  None, \"c\",  None, None, None],    # S2\n         [\"b,c\", None, None, None, \"a\",  None, None],   # S3\n         [\"c\",   \"a\",  None, None, None, \"b\",  None],   # S4\n         [None,  \"a\",  None, \"c\",  None, None, \"b\"],    # S5\n         [None,  None, None, None, None, None, \"a,b,c\"]]# S6\n    if is_neg:\n        T = [T[6], T[0], T[1], T[2], T[3], T[4], T[5]]\n\n    state = \"S0\"\n    end_states = [\"S0\"] if is_neg else [\"S6\"]\n    tape = \"\" # Start with the empty tape \n    steps = 0\n\n    # Transitions _to_ the state in question. \n    # - weights are handled by P\n    arr = [0, 1, 2, 3, 4, 5, 6, \"f\"] \n\n    while True:\n        ix = int(state[-1])\n        x = random.choices(arr, weights=P[ix], k=1)[0]\n        if x == \"f\":\n            break\n\n        next_state = \"S\" + str(x)\n        write_to = T[ix][x]\n        if \",\" in write_to:\n            write_to = random.choice(write_to.split(\",\"))\n        tape += write_to\n        state = next_state\n        \n        steps += 1\n        if max_length is not None:\n            if steps >= max_length:\n                return None\n    if tape == \"\":\n        return None\n    return tape"

parity_desc = "This is a pattern matching task. The strings in this task are generated from a probabilistic automaton.\n"
parity_desc += "Each string is labelled 0 or 1 depending on whether the pattern \"abcabb\" is (1) or is not (0) in the string.\n"
parity_desc += "Your job is to learn what is the likelihood of a string to be labelled 0 or 1, and output the correct label.\n"
parity_desc += "In the limit where the automaton is deterministic, if the pattern is present in the string, the label is always 1. Else, it is 0.\n"

parity_desc += "Here's the automaton:\n\n"
parity_desc += ecm_str_desc
parity_desc += "\n\n"

parity_desc_zero_shot = parity_desc + "Give your answer as a single integer, and your reasoning in a new line.\n"
parity_desc_zero_shot += "For example:\naabcabbb:1\nThe pattern is present in the string.\n\n"

parity_desc += "Given the data below, determine what is the most likely label for the given string and output ONLY the label.\n"
parity_desc += "Data:\n\n"

gpt4.update_params({tkey: tokens})

suff = f"w_automaton_{problem}"

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=False,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)

suff = f"w_automaton_{problem}_shuffled"

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = parity_desc_zero_shot, few_shot_prompt = parity_desc, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)


# ## APO

# In[ ]:


parity_desc = "This is a pattern matching task. The strings in this task are generated from a probabilistic automaton.\n"
parity_desc += "Each string is labelled 0 or 1 depending on whether the pattern \"abcabb\" is (1) or is not (0) in the string.\n"
parity_desc += "Your job is to learn what is the likelihood of a string to be labelled 0 or 1, and output the correct label.\n"
parity_desc += "In the limit where the automaton is deterministic, if the pattern is present in the string, the label is always 1. Else, it is 0.\n"
parity_desc += "Given the data below, determine what is the most likely label for the given string and output ONLY the label.\n"
parity_desc += "Data:\n\n"

p_hat_candidates = apo(parity_desc, corpus_id, problem=problem) # Use training for optimisation
p_hat = p_hat_candidates[0][0]
score = p_hat_candidates[0][-1]

with open(f"{out_dir_root}/apo_prompt_{problem}.json", "w", encoding="utf-8") as f:
    f.write(json.dumps({"Prompt": p_hat, "Score": score, "InitialPrompt": parity_desc, "OtherCandidates": p_hat_candidates, "Problem": problem}, ensure_ascii=False))


# In[ ]:


# APO is a zero-shot problem, but we'll humour ourselves and try it out with shots anyway.
p_hat = [json.loads(l) for l in open(f"{out_dir_root}/apo_prompt_{problem}.json", "r", encoding="utf-8").readlines()][0]["Prompt"]
suff = f"apo_{problem}"
gpt4.update_params({tkey: tokens})

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = p_hat, few_shot_prompt = p_hat,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = p_hat, few_shot_prompt = p_hat,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)


suff = f"apo_{problem}_shuffled"

call_and_solve_for(problem, suff, corpus_id_test, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = p_hat, few_shot_prompt = p_hat, shuffle_exemplars=True,
                   do_ood=False, out_dir_root=out_dir_root, is_cot=False)

call_and_solve_for(problem, suff, corpus_od_test_w_negs, SHOTS=[0, 2, 5, 10, 20, 50, 100],
                   zero_shot_prompt = p_hat, few_shot_prompt = p_hat, shuffle_exemplars=True,
                   do_ood=True, out_dir_root=out_dir_root, is_cot=False)
