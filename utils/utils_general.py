import os
import time


def getPathList(path, suffix='png'):
    if (path[-1] != '/') & (path[-1] != '\\'):
        path = path + '/'
    pathlist = list()
    g = os.walk(path)
    for p, d, filelist in g:
        for filename in filelist:
            if filename.endswith(suffix):
                pathlist.append(os.path.join(p, filename))
    return pathlist

def mkdir(path):
    if not os.path.isdir(path):
        mkdir(os.path.split(path)[0])
    else:
        return
    if not os.path.isdir(path):
        os.mkdir(path)

def path_join(root, name):
    if root == '':
        return name
    if name[0] == '/':
        return os.path.join(root, name[1:])
    else:
        return os.path.join(root, name)

class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))