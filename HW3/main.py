import sys

from transformer import RollingAVGTransform


def main():
    # Transform a master DF into a DF of rolling averages.
    rolling = RollingAVGTransform()
    rolling_avg = rolling._transform()
    rolling_avg.show(100)
    return


if __name__ == "__main__":
    sys.exit(main())
