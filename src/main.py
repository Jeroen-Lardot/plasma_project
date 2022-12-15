import sys
from Acquisitor import Acquisitor
from datetime import datetime


def main() -> int:
    acquisitor = Acquisitor(write_vtk=True)
    acquisitor.get_data(t_start=datetime(2017, 8, 4, 8, 56, 0), t_end=datetime(2017, 8, 4, 11, 58, 0))
    return 0

if __name__ == '__main__':
    sys.exit(main())
