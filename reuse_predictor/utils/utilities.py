import os
import logging
import torch
logger = logging.getLogger(__name__)


def save_checkpoint(state, checkpoint_dir, checkpoint_prefix):
    """Saves model and training parameters at checkpoint: checkpoint_prefix.pth.tar'.
    Args:
        state: (dict) contains model's state_dict It may also contain other keys such as epoch,
        optimizer state_dict etc.
        checkpoint_dir: (str) is the location of the folder in which the checkpoint file will be stored.
        checkpoint_prefix: (str) is the prefix that will be used for the resulting checkpoint.
    """
    filepath = os.path.join(checkpoint_dir, '%s.pth.tar' % checkpoint_prefix)
    if not os.path.exists(checkpoint_dir):
        logging.info("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    else:
        logging.info("Checkpoint Directory exists!")
    torch.save(state, filepath)