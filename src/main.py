import sys
from Acquisitor import Acquisitor
from datetime import datetime


def main() -> int:
    acquisitor = Acquisitor(write_h5=True)
    acquisitor.get_data(t_start=datetime(2015, 10, 16, 13, 7, 0), t_end=datetime(2015, 10, 16, 13, 8, 0))
    return 0

if __name__ == '__main__':
    sys.exit(main())