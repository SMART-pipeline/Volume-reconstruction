import argparse
from VISoR_Reconstruction.reconstruction_executor.executor import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VISoR Reconstruction task runner')
    parser.add_argument('task_list', help='Task file.')
    args = parser.parse_args()
    with open(args.task_list) as d:
        main(d.read())
