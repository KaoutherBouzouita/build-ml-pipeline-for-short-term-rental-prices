#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning,
exporting the result to a new artifact
"""
import argparse
import logging

import hydra
from omegaconf import DictConfig

import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


@hydra.main(config_name='config')
def go(args, config: DictConfig):
    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact.
    logger.info("Downloading input artifact...")

    run = wandb.init(project="nyc_airbnb", group="eda")

    local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Reading csv file...")

    df = pd.read_csv(local_path)

    logger.info("Removing outliers...")

    idx = df["price"].between(args.min_price, args.max_price)

    df = df[idx].copy()

    logger.info("Converting last_review string feature to datetime...")

    df["last_review"] = pd.to_datetime(df["last_review"])

    logger.info("Converting result to CSV file...")

    idx = df['longitude'].between(
        config["data_check"]["min_longitude"],
        config["data_check"]["max_longitude"]) & df['latitude'].between(
        config["data_check"]["min_latitude"],
        config["data_check"]["max_latitude"])
    df = df[idx].copy()

    df.to_csv(args.output_artifact, index=False)

    logger.info("Uploading output artifact to W&B")

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Data to be loaded and cleaned",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Cleaned data",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Type of the output",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="Minimum rental price",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="Maximum rental price",
        required=True
    )

    args = parser.parse_args()

    go(args)
