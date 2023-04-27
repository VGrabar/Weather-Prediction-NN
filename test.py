import dotenv
import hydra
import glob
from omegaconf import DictConfig

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


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
    # Evaluate model
    chkpts = glob.glob(config.ckpt_folder + "**/*.ckpt", recursive=True)
    chkpts = [c for c in chkpts if c != "last.ckpt"]
    print(chkpts)
    for c in chkpts:
        config.ckpt_path = c
        preds, all_targets = test(config)
        all_preds.append(preds)
    return

    all_preds = torch.stack((all_preds))
    print(all_preds.shape)
    all_preds = torch.mean(all_preds, dim=0)
    print(all_preds.shape)
    rocauc_table = metrics.rocauc_celled(all_targets, all_preds)

    #return test(config)


if __name__ == "__main__":
    
    
    main()
