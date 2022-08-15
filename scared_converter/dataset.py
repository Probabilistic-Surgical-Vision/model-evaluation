import argparse
import glob
import os.path
from multiprocessing.pool import Pool

from .keyframe import SCAREDKeyframeConverter

parser = argparse.ArgumentParser()

parser.add_argument('source', type=str,
                    help='The path to the original SCARED dataset.')
parser.add_argument('--target', '-t', default=None, type=str,
                    help='The path to save the converted dataset to.')
parser.add_argument('--rectify', default=False, action='store_true',
                    help='Rectify the images when saving.')
parser.add_argument('--workers', '-w', default=4, type=int,
                    help='The number of processes to spawn.')
parser.add_argument('--image-size', type=int, nargs=2, default=(512, 256),
                    help='The size to make all video images.')


def extract_keyframe(config: dict) -> None:
    converter = SCAREDKeyframeConverter(**config)
    converter.extract()


def main(args: argparse.Namespace) -> None:
    pool = Pool(processes=args.workers)

    keyframe_glob = os.path.join(args.source, '*', 'dataset_*', 'keyframe_*')
    keyframes = glob.glob(keyframe_glob)

    for keyframe in keyframes:
        if '.zip' in os.path.basename(keyframe):
            print(f'Ignoring {keyframe} as it needs to be unzipped.')
            continue

        route = os.path.relpath(keyframe, start=args.source)
        keyframe_target = os.path.join(args.target, route)

        keyframe_config = {
            'source': keyframe,
            'target': keyframe_target,
            'rectify': args.rectify
        }

        pool.apply_async(extract_keyframe, args=(keyframe_config,))
        print(f'Queued {keyframe}')

    pool.close()
    pool.join()

    print('Conversions completed.')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
