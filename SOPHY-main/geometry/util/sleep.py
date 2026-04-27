import time
import argparse
from datetime import datetime

def sleep(seconds):
    print(f'Current time: {datetime.now()}')
    print(f'Sleeping for {seconds} seconds...')
    time.sleep(seconds)

def arg_parser():
    parser = argparse.ArgumentParser(description='Sleep for a number of seconds')
    parser.add_argument('--seconds', '-s', type=int, help='Number of seconds to sleep')
    return parser

if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()
    sleep(args.seconds)
