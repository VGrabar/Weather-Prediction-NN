import dotenv
import hydra
import glob
from pathlib import Path
import torch
import os
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
from src import utils

log = utils.get_logger(__name__)

@hydra.main(config_path="configs/", config_name="test.yaml")
def main(config: DictConfig):

    # Imports can be nested inside @hydra.main to optimize tab completion
    # https://github.com/facebookresearch/hydra/issues/934
    from src import utils
    from src.utils import metrics
    from src.testing_pipeline import test

    # Applies optional utilities
    utils.extras(config)
    all_preds = []
    os.chdir("/Weather-Prediction-NN")
    # Evaluate model
    chkpts = []
    path = config.ckpt_folder
    for ck in Path(path).rglob("*.ckpt"):
        if not "last" in str(ck):
            chkpts.append(ck)
    for c in chkpts:
        config.ckpt_path = c
        preds, all_targets = test(config)
        all_preds.append(preds)

    all_preds = torch.stack((all_preds))
    all_preds = torch.mean(all_preds, dim=0)
    rocauc_table, ap_table, f1_table = metrics.metrics_celled(all_targets, all_preds)
    res_rocauc = torch.median(rocauc_table)
    res_ap = torch.median(ap_table)
    res_f1 = torch.median(f1_table)
    log.info(f"test_ensemble_median_rocauc: {res_rocauc}")
    log.info(f"test_ensemble_median_ap: {res_ap}")
    log.info(f"test_ensemble_median_f1: {res_f1}")
    with open("ens.txt", "a") as f:
        f.write(config.ckpt_folder + "\n")
        f.write("median_rocauc: " + str(res_rocauc) + "\n")
        f.write("\n")
        f.write("median_ap: " + str(res_ap) + "\n")
        f.write("\n")
        f.write("median_f1: " + str(res_f1) + "\n")
        f.write("\n")
    return


if __name__ == "__main__":
    
    
    main()
