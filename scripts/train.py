#!/usr/bin/env python3
import hydra

from wam_h3.train.runtime import run_training


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
