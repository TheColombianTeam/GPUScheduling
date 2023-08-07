from PyOpenTCU import Tensor
from utils.args import args


import Schedulers as schedulers_list


def main():
    print(args)
    scheduler = getattr(schedulers_list, args.scheduler)()
    tensor = Tensor()
    print(f"Tensor {scheduler}")


if __name__ == "__main__":
    main()
