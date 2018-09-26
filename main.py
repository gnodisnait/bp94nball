import bp94nball
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--restart", type=int, default=1,
                        help='whether need to restart: 0 for not restart, 1 for restart')
    args = parser.parse_args()
    print(args)
    restartTF = args.restart
    bp94nball.training_balls(restart=restartTF)
