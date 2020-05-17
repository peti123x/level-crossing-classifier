import sys
import pandas as pd

num_args = len(sys.argv)

if num_args > 1:
    path = sys.argv[1]
    df = None
    try:
        df = pd.read_pickle(path)
    except FileNotFoundError:
        print("File does not exist.")
        exit()
    #Detail is in second parameter
    try:
        if sys.argv[2] == "-d":
            pd.set_option('display.expand_frame_repr', False)
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print(df)
        else:
            print(df)
    except IndexError:
        #Otherwise short one
        print(df)
