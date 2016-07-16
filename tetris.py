#-*- coding:utf-8 -*- 
import sys
import time
import copy
import threading
import msvcrt
import random
import argparse


class Tetris:
    def __init__(self, field_size = [10,10]):
        self.field_info = field_size #x,y
        self.field = [ ['_' for _ in range(field_size[0])] for _ in range(field_size[1]) ]
        self.tmp_field = []

        self.blocks =  [
                        [['_','#','#'],
                         ['#','#','_']],

                        [['#','_','_'],
                         ['#','#','#']],

                        [['#','#','#'],
                         ['_','#','_']],
                        
                        [['#','#'],
                         ['#','#']],
                        
                        [['#','#','#','#']]]

        self.block = []

        self.movement = 0
        self.rotate_flag = 0
        self.end_flag = 0

    def isHit(self, block_pos):
        for b_row, f_row in zip(self.block,block_pos):
            for b, f in zip(b_row, f_row):
                if b == '#' and f == '#':
                    return True
        else:
            return False

    def draw_field(self, field = None, isFirst = False):
        field = self.field if field == None else field
        if not isFirst:
            sys.stdout.write("\033[{}A".format(self.field_info[1]))
        for row in field:
            print("".join(row))

    def drawGameOver(self):
        str_size = len("gameover")

        if self.field_info[0] > str_size:
            start_index = int((self.field_info[0]-str_size)/2)
            self.field[int(self.field_info[1]/2)][start_index:str_size+1] = "GameOver"
        return self.field
     

    def isGameOver(self):
        if '#' in self.field[0]:
            self.draw_field(self.drawGameOver())
            return True
        else: return False

    def getNextField(self, field = None, block = None):
        field = self.field if field == None else field
        block = self.block if block == None else block

        f = copy.deepcopy(field)
        for f_i, b_row in zip(range(self.row, self.row + len(block)), block):
            for f_j, b in zip(range(self.pos, len(block[0]) + self.pos), b_row):
                if b == '#':
                    f[f_i][f_j] = b
        return f
     
    def checkCompleteLine(self):
        ret = []
        deleteCount = 0
        for index, row in enumerate(self.field):
            if not '_' in row:
                deleteCount += 1
            else:
                ret.append(row)
        return [['_']*self.field_info[0] for _ in range(deleteCount)] + ret

    def waitKey_thread(self):
        while True:
            s = str(msvcrt.getwch())
            #sys.stdout.write("\033[1A")
            if s in 'j' or s in 'a':
                self.movement -= 1
            if s in 'l' or s in 'd':
                self.movement += 1
            if s in 'k' or s in 's':
                self.rotate_flag = 1
            if s in 'q':
                self.end_flag = 1

    def start_wait_key(self):
        self.thread = threading.Thread(target = self.waitKey_thread)
        self.thread.daemon = True
        self.thread.start()

    def waitKey(self, waitTime):
        time.sleep(waitTime)
        
        self.block = self.rotateBlock(self.block, self.rotate_flag)
        if self.isHit( [rows[self.pos : len(self.block[0])+self.pos] 
                        for rows in self.field[self.row:self.row+len(self.block)]] ) \
            or len(self.block) + self.row > len(self.field):
            self.block = self.rotateBlock(self.block, 
                    1 if self.rotate_flag == 2 else 2 if self.rotate_flag == 1 else 0)
        self.rotate_flag = 0

        if self.pos + self.movement >= 0 \
                and self.pos + self.movement <= self.field_info[0] - len(self.block[0]) \
                and not self.isHit(
                        [rows[self.pos+self.movement : len(self.block[0])+self.pos+self.movement] 
                            for rows in self.field[self.row:self.row+len(self.block)]] ):
            self.pos += self.movement
        self.movement = 0

    def rotateBlock(self, block, direction):
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

    def start(self):
        self.start_wait_key()
        self.draw_field(self.field, True)
        while True:
            self.block = self.rotateBlock(self.blocks[random.randint(0,len(self.blocks)-1)],
                        random.randint(0,3))
            self.pos = int((len(self.field[0]) - len(self.block[0]))/2)
            self.row = 0
            self.tmp_field = self.getNextField()
            while self.row < self.field_info[1] - len(self.block) + 1:
                self.waitKey(0.3)
                block_pos = [rows[self.pos:len(self.block[0])+self.pos] 
                        for rows in self.field[self.row:self.row+len(self.block)]]
                if self.isHit(block_pos) or self.end_flag:
                    self.field = self.tmp_field
                    #field = getNextField(row,field,block)
                    self.draw_field()
                    break
                else:
                    self.tmp_field = self.getNextField()
                    self.draw_field(self.tmp_field)
                self.row += 1
            else:
                self.row -= 1
                self.field = self.getNextField()
            self.field = self.checkCompleteLine()
            self.draw_field()
            if self.isGameOver() or self.end_flag:
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "TETRIS")

    tetris = Tetris([10,15])
    tetris.start()

