from argparse import ArgumentParser
from tqdm import tqdm
import shutil
import glob
import os

parser = ArgumentParser(description="Move files from one directory to another")
parser.add_argument("source",      help="Source directory")
parser.add_argument("destination",  help="Destination directory")
parser.add_argument("-p", "--pattern", type=str, default='', help="Pattern to match files to move")
parser.add_argument("-e", "--exclude", type=str, default='', help="Pattern to exclude files to move")
parser.add_argument("-c", "--copy", action="store_true", help="Copy files instead of moving")
args = parser.parse_args()

src = args.source
dst = args.destination
ptn = args.pattern
exc = args.exclude

def get_files(src, ptn, exc):
    pattern = os.path.join(src, ptn)
    files = glob.glob(pattern)
    files = [f.replace('\\', '/') for f in files]
    if exc:
        exclude = os.path.join(src, exc)
        exclude_files = glob.glob(exclude)
        files = [f for f in files if f not in exclude_files]
    file_names = [f.replace(src, '').lstrip('/') for f in files]
    return files, file_names

def mass_action(files, names, destination):
    for file, name in tqdm(zip(files, names), total=len(files)):
        action(file, os.path.join(destination, name))

def action(*args):
    pass

def move(src, dst):
    shutil.move(src, dst)

def copy(src, dst):
    shutil.copy(src, dst)

def clean_path(path):
    return path.replace('\\', '/')

if args.copy:
    action = copy
else:
    action = move


if __name__ == '__main__':
    if not ptn:
        ptn = '*'

    src = clean_path(src).rstrip('/')
    dst = clean_path(dst).rstrip('/')
    ptn = clean_path(ptn).lstrip('/')
    exc = clean_path(exc).lstrip('/')

    files, names = get_files(src, ptn, exc)
    mass_action(files, names, dst)


