import sys
from Acquisitor import Acquisitor
from datetime import datetime


def main() -> int:
    acquisitor = Acquisitor(write_vtk=False)
    
    acquisitor.get_data(t_start=datetime(2015, 9, 8, 10, 0, 0), t_end=datetime(2015, 9, 8, 10, 8, 0), mms_analysis=True)
    return 0

if __name__ == '__main__':
    sys.exit(main())
