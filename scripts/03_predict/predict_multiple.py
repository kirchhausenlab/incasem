import logging
import os
import sys
import json

import configargparse as argparse

from incasem.tracking.sacred import ex
from predict import (
    predict,
    observer_setup,
    get_config_from_database,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--checkpoint_dir',
        required=True,
        help='Base directory with model checkpoints.'
    )
    parser.add_argument(
        '--checkpoint_basename',
        default='model_checkpoint_',
        help='Prefix of all checkpoint filenames before the iteration number.'
    )
    parser.add_argument(
        '--start',
        type=int,
        required=True,
        help='First checkpoint to use.'
    )
    parser.add_argument(
        '--stop',
        type=int,
        required=True,
        help='Last checkpoint to use, this iteration is included.'
    )
    parser.add_argument(
        '--step',
        type=int,
        required=True,
        help='Interval between checkpoints.'
    )

    return parser.parse_known_args()


@ex.capture
def directory_structure_setup(_config, _run, iteration):

    predictions_out_path = _config['prediction']['directories']['prefix']
    if not os.path.isdir(predictions_out_path):
        os.makedirs(predictions_out_path)

    # training run id, then prediction run id as subfolder, then iteration
    # number as subfolder

    run_path = os.path.join(
        f"train_{int(_config['prediction']['run_id_training']):04d}",
        f"predict_{_run._id:04d}",
        f"iteration_{iteration:06d}"
    )
    return run_path


def append_iteration_to_name(argv, iteration):
    index = None
    try:
        index = argv.index('-n')
    except ValueError:
        pass
    try:
        index = argv.index('--name')
    except ValueError:
        pass
    if index is not None:
        argv[index + 1] = f"{argv[index+1]}_{iteration}"
    else:
        argv.append('--name')
        argv.append(f"iteration_{iteration}")

    return argv


@ex.main
def predict_multiple(_config, _run):
    checkpoint_dir = _config['prediction']['checkpoint_dir']
    checkpoint_basename = _config['prediction']['checkpoint_basename']
    start = _config['prediction']['iteration_start']
    stop = _config['prediction']['iteration_stop']
    step = _config['prediction']['iteration_step']

    # include end point
    iterations = list(range(start, stop + 1, step))
    logger.info(f"Predicting for iterations {iterations}.")

    for i in iterations:
        checkpoint = os.path.join(checkpoint_dir, f"{checkpoint_basename}{i}")
        run_path = directory_structure_setup(_config, _run, i)
        predict(
            _config=_config,
            _run=_run,
            checkpoint=checkpoint,
            iteration=i,
            run_path=run_path
        )


def main():
    args_multiple, remaining_argv_multiple = parse_args()

    ex.add_config({
        'prediction': {
            'checkpoint_dir': args_multiple.checkpoint_dir,
            'checkpoint_basename': args_multiple.checkpoint_basename,
            'iteration_start': args_multiple.start,
            'iteration_stop': args_multiple.stop,
            'iteration_step': args_multiple.step,
        },
    })
    ex.add_config('../02_train/config_training.yaml')
    ex.add_config('config_prediction.yaml')

    sys.argv = [
        sys.argv[0],
        *remaining_argv_multiple
    ]

    args, remaining_argv = observer_setup()

    with open(args.mongodb_training) as f:
        db_config = json.load(f)
    config = get_config_from_database(
        db_config['url'],
        db_config['db_name'],
        args.run_id)
    ex.add_config(config)

    sacred_default_flags = ['-C', 'no']
    argv = [
        sys.argv[0],
        *sacred_default_flags,
        *remaining_argv,
        f'prediction.run_id_training={args.run_id}'
    ]
    logger.info(argv)

    ex.run_commandline(argv)


if __name__ == '__main__':
    main()
