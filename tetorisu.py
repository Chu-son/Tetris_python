import sys
import time
import copy
import threading
import msvcrt

class block:
    def __init__(self,data):
        self.data = data
        self.height
    def __reset_height(self):
        self.height = 0
        for index, row in enumwrate(self.data):
            if '#' in row:
                self.height += 1
    def __reset_width(self):
        min = len(self.data[0])
        max = 0
        for row in self.data:
            pass

blk1 = [['_','#','#'],
        ['#','#','_']]
blk2 = [['#','_','_'],
        ['#','#','#']]
blk3 = [['#','#','#'],
        ['_','#','_']]
blk4 = [['#','#'],
        ['#','#']]
bkl5 = [['#','#','#','#']]


field_info = [8,10] #x,y
blocks = [
        [2,2,2],
        [1,4,1],
        [3,3,0],
        [3,5,3],
        [5,1,0],
        [6,3,5]
        ]#h,w,x

field = [['_' for _ in range(field_info[0])] for _ in range(field_info[1])]

class Tetris:
    def __init__(self, field_size = [10,10]):
        self.field = [ ['_' for _ in range(field_size[0])] for _ in range(field_size[1]) ]
        self.field_info = field_size
        
    def start(self):
        while True:
            pass


def isHit(block_pos):
    for l in block_pos:
        if '#' in l:
            return True
    else:
        return False

def show(field, isFirst = False):
    if not isFirst:
        sys.stdout.write("\033[{}A".format(field_info[1]))
    for row in field:
        print("".join(row))

def getNextField(row,field,block):
    #f = field[:][:]
    f = copy.deepcopy(field)
    for i in range(row,row+block[0]):
        for j in range(block[2],block[1]+block[2]):
            f[i][j] = '#'
    return f
    
def checkCompleteLine(field):
    ret = []
    deleteCount = 0
    for index, row in enumerate(field):
        if not '_' in row:
            deleteCount += 1
        else:
            ret.append(row)
    return [['_']*field_info[0] for _ in range(deleteCount)] + ret
def waitKey_thread():
    global x
    while True:
        s = str(msvcrt.getwch())
        #sys.stdout.write("\033[1A")
        if s in 'a':
            x -= 1
        if s in 'd':
            x += 1
x=0
def waitKey(waitTime, block):
    time.sleep(waitTime)
    global x
    if block[2]+x >=0 and block[2]+x <= field_info[0] - block[1]:
        block[2] += x
    x=0
    
thread = threading.Thread(target = waitKey_thread)
thread.daemon = True
thread.start()

show(field, True)
for block in blocks:
    for row in range(field_info[1] - block[0]):
        waitKey(0.5,block)
        block_pos = [l[block[2]:block[1]+block[2]] for l in field[row:row+block[0]+1]]
        if isHit(block_pos):
            field = getNextField(row,field,block)
            break
        else:
            show(getNextField(row,field,block))
    else:
        for i in range(field_info[1]-block[0],field_info[1]):
            for j in range(block[2],block[1]+block[2]):
                field[i][j] = '#'
    field = checkCompleteLine(field)
    show(field)
