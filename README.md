# Is In-Context Learning Learning?

Code to reproduce the results from the paper Is In-Context Learning Learning?

>Note: this is a very expensive experiment to run. If at all possible, please avoid running the full suite of experiments. Do [consider](https://mlco2.github.io/impact/#compute) [the](https://llmemissions.com/) environmental impact of it before running this code.

## Outline

This repo contains everything you need to generate the data (`DataGenerator.ipynb`), reproduce the experiments (`Labeller.ipynb`, `Labeller-Ablation.ipynb`, and/or `scripts/`), and generate the evaluation results (`Evaluator.ipynb`). 

The raw predictions are under `raw_predictions`, if you want to avoid the whole 'taking a year to run' thing. You'll have to fiddle a bit with `Evaluator.ipynb` to re-generate the Excel files, but I also include these under `annotated_data`. 


## Generate the data

You just need to call the functions from within the `DataGenerator.ipynb` notebook. For MazeComplete you will also need to do `!pip install mazelib` in order to get it to... well, generate mazes. 


## Reproduce the experiment:

You may reproduce the work by either calling it via the notebooks or using the scripts provided in `scripts/`. 
Either way:
- For some models you will have to modify the `llmclient.py` class (i.e., if it is an OpenAI model, modify it to load your key and stuff).
- You need to move the datasets from under `datasets` to wherever your notebook/scripts are. Or modify their hardcoded path.

Every model will need you to manually make a specific directory (`root_out`) where all predictions will be dumped. You also need to add directories per problem (`<root_out>/PARITY`, `<root_out>/Pattern_Matching`... etc). To make your life easier, just copy-paste `template_folder` and replace the name.

_In case of failures_ (which will happen, especially if you are running Windows and the thing decides to update), you can restart the experiments where you left out. Simply replace the number of shots in the calls by doing one of:

```python
SHOTS=[(shot, last_datapoint), more, shots] # ID
SHOTS=[(shot, delta, last_datapoint), more, shots] # OOD
```


## Data analysis

It's very difficult to generate the data in a parseable, easy-to-handle format (there are a lot of interrupted high volume calls). To do this you will need to run `reshape_predictions.py` first:

```bash

python reshape_predictions.py --root <the root_out variable you set earlier>

```

It will leave you with a `<root_out>-consolidated` folder, OR tell you that some datapoints are incomplete/missing and that you should run them again. 

Then you can load the data in the `Evaluator.ipynb` notebook.

This notebook will dump everything into an Excel file with _most_ of the needed patterns for your analysis already there. However, you will need to create the plots (painstakingly) by yourself, in addition to any further eval. I would recommend just using mine.

Dumping files supports two modes via the `punish` and `shuffled` variables. `punish` is for our experiments on the distinction between learning and comprehension + learning. Punish the model to get comprehension + learning. `shuffled` is what it says: output the shuffled subsets instead of the unshuffled subsets. This one isn't very interesting.

## Citation

```
anonymised
```

## Licence

MIT for all code. The word set from Saparov and He used in word salad is [Apache-2.0](https://github.com/asaparov/prontoqa). 
