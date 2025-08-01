import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/main-user/Jetson-RT-Traffic-System/install/videopub'
