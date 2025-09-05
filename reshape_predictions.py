'''
## Reshape predictions

This is to make sure your files are in the right shape for the script.
It is also a nicer filesystem to work with (compared to the output system from the notebook, 
which is designed to be robust to interruptions while you collect the 4.4M or so datapoints)
'''
from data_utils import verify_files_for_model, consolidate_files
import argparse

problem2fname = {"PARITY": "parity",
                 "Pattern_Matching": "pattern_matching",
                 "Reversal": "reversal",
                 "Stack": "stack",
                 "Vending_Machine": "vending_machine",
                 "Vending_Machine_Sum": "vending_machine_with_sum",
                 "MazeComplete": "maze_complete",
                 "MazeSolve": "maze_solve",
                 "Hamiltonian": "hamiltonian",
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Reshape predictions')
    parser.add_argument('root', help='The root directory where all your predictions are dumped. This is `root_out` in the data collection notebook.')
    args = parser.parse_args()

    origin_folder = args.root
    # Usage:
    # 1. Check if the files are good -- you will have some missing, but that's ok
    # just keep it around 950+
    verify_files_for_model(origin_folder, problem2fname=problem2fname)
    # 2. Correct the files and write them to a directory
    verify_files_for_model(origin_folder, write_to=f"{origin_folder}-temp", problem2fname=problem2fname)
    # 3. Re-verify
    verify_files_for_model(f"{origin_folder}-temp", no_tmp=True, problem2fname=problem2fname)
    # 4. Consolidate -- don't skip this step, all the rest of the code uses this step!
    consolidate_files(f"{origin_folder}-temp", f"{origin_folder}-consolidated", problem2fname=problem2fname)
    print(f"Output folder is {origin_folder}-consolidated.\nDon't forget to copy apo_prompts from {origin_folder}-temp!")
