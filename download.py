import wget
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--aif360-folder',
                        help='The folder where aif360 packege is installed.',
                        default="myvenv/Lib/site-packages/aif360/")
    file_name = ""
    args = parser.parse_args()

    out = os.path.join(args.aif360_folder, 'data', 'raw', 'compas',
                           'compas-scores-two-years.csv')
    wget.download('https://raw.githubusercontent.com/propublica/'
                  'compas-analysis/master/compas-scores-two-years.csv',
                  out)