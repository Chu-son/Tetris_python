#-*- coding:utf-8 -*- 
import sys
import time
import copy
import threading
import msvcrt
import random

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
blk5 = [['#','#','#','#']]


field_info = [10,8] #x,y
blocks = [
        [['_','#','#'],
         ['#','#','_']],

        [['#','_','_'],
         ['#','#','#']],

        [['#','#','#'],
         ['_','#','_']],
        
        [['#','#'],
         ['#','#']],
        
        [['#','#','#','#']],

        [['■']]
        ]

field = [['_' for _ in range(field_info[0])] for _ in range(field_info[1])]

class Tetris:
    def __init__(self, field_size = [10,10]):
        self.field = [ ['_' for _ in range(field_size[0])] for _ in range(field_size[1]) ]
        self.field_info = field_size
        
    def start(self):
        while True:
            pass

def isGameOver(field):
    if '#' in field[0]:
        show(drowGameOver(field))
        return True
    else: return False

def drowGameOver(field):
    f_rows = len(field)
    f_cols = len(field[0])
    str_size = len("GameOver")

    if f_cols > str_size:
        start_index = int((f_cols-str_size)/2)
        field[int(f_rows/2)][start_index:str_size+1] = "GameOver"
    return field
    

def isHit(block,block_pos):
    for b_row, f_row in zip(block,block_pos):
        for b, f in zip(b_row, f_row):
            if b == '#' and f == '#':
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
    for f_i, b_row in zip(range(row,row+len(block)), block):
        for f_j, b in zip(range(pos,len(block[0])+pos), b_row):
            if b == '#':
                f[f_i][f_j] = b
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
    global rotate_flag
    while True:
        s = str(msvcrt.getwch())
        #sys.stdout.write("\033[1A")
        if s in 'a':
            x -= 1
        if s in 'd':
            x += 1
        if s in 's':
            rotate_flag = 1
x = 0
pos = 0
rotate_flag = 0
def waitKey(waitTime, block):
    time.sleep(waitTime)
    global x
    global pos
    global rotate_flag
    
    block = rotateBlock(block, rotate_flag)
    rotate_flag = 0

    if pos + x >=0 and pos + x <= field_info[0] - len(block[0]):
        pos += x
    x=0

    return block
    
def rotateBlock(block, direction):
    ret = [ list(b) for b in zip(*block)]
    if direction == 0:
        return block
    elif direction == 1:#cw
        [row.reverse() for row in ret]
        return ret
    elif direction == 2:#ccw
        ret.reverse()
        return ret
    elif direction == 3:#flip
        block.reverse()
        return block


thread = threading.Thread(target = waitKey_thread)
thread.daemon = True
thread.start()

show(field, True)
for _ in range(10):
    #block = rotateBlock(blocks[random.randint(0,len(blocks)-1)],
    #                    random.randint(0,3))
    block = blocks[5]
    pos = 0
    row = 0
    while row < field_info[1] - len(block)+1:
        block = waitKey(0.5,block)
        block_pos = [l[pos:len(block[0])+pos] for l in field[row+1:row+len(block)+1]]
        if isHit(block,block_pos):
            field = getNextField(row,field,block)
            break
        else:
            show(getNextField(row,field,block))
        row += 1
    else:
        field = getNextField(field_info[1] - len(block),field,block)
    field = checkCompleteLine(field)
    show(field)
    if isGameOver(field):
        break
