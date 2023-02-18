import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--test", type=int, help="active test",
                    nargs="+", metavar=("nb_episode, nb_step"))
parser.add_argument("--train", help="active train", action="store_true")
args = parser.parse_args()
print(args.test, args.train)
print(args.test[0])
if args.test:
    print("work")
