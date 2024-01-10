import argparse

from src.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--config_path')

args = parser.parse_args()
pipeline = Pipeline(args.config_path)
pipeline.start()