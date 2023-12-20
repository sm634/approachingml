"""
Docstring
---------

The main python script used to run all other src .py scripts.
"""
import argparse

from src.create_folds import save_mnist_df
from src.train import run

def main():
    # Creating an ArgumentParser object
    parser = argparse.ArgumentParser(description="parse arguments referencing the scripts we want to run.")

    # Adding the '--script' argument
    parser.add_argument('--script',
                        help='provide the name of the script you want to run',
                        type=str,
                        choices=['config.py', 'create_folds.py', 'inference.py', 'model_dispatcher.py', 'models.py', 'train.py'])
    parser.add_argument('--model',
                        help='select the model you want to use.',
                        type=str)
    parser.add_argument('--folds',
                        help='select the number of folds to create for the training set.',
                        type=int,
                        default=0)
    # Parsing the arguments
    args = parser.parse_args()

    # Accessing the value passed for '--scirpt' and running that scirpt.
    if args.script == 'config.py':
        pass
    if args.script == 'create_folds.py':
        save_mnist_df()
    if args.script == 'inference.py':
        pass
    if args.script == 'model_dispatcher.py':
        pass
    if args.script == 'models.py':
        pass
    if args.script == 'train.py':
        run(fold=args.folds, model=args.model)

if __name__ == '__main__':
    main()
