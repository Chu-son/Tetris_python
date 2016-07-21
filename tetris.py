#-*- coding:utf-8 -*- 
import time
import copy
import threading
import msvcrt
import random
import argparse

import math
import numpy as np
from chainer import cuda, optimizers, FunctionSet, Variable, Chain
import chainer.functions as F


def print_23(s = '', end = None):
    if end != None:
#       print(s,end = end)
        print s,
    else:
        print(s)

class SState(object):
    def __init__(self, field_size, frame_num):
        self.seq = np.ones((frame_num, field_size[0] * field_size[1]), dtype=np.float32)
        
    def push_s(self, state):
        self.seq[1:frame_num] = self.seq[0:frame_num-1]
        self.seq[0] = state
        
    def fill_s(self, state):
        for i in range(0, frame_num):
            self.seq[i] = state

class Q(Chain):
    def __init__(self, state_dim, action_num ):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 500),
            q_value=F.Linear(500, action_num)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
        h1 = F.leaky_relu(self.l1(x))
        y = F.leaky_relu(self.q_value(h1))
        return y


class Agent():
    def __init__(self, field, epsilon = 0.99):
        # 行動
        self.actions = range(-5, 7)
        self.prevActions = np.zeros_like(self.actions)
        
        self.STATE_DIM = len(field) * len(field[0]) + len(self.prevActions)
        
        # DQN Model
        self.model = Q(self.STATE_DIM, len(self.actions))
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model)
        
        self.epsilon = epsilon
        
        # 経験関連
        self.eMem = np.array([],dtype = np.float32)
        self.memPos = 0 
        self.memSize = 30000

        # 学習関連のパラメータ
        self.batch_num = 30
        self.gamma = 0.7
        self.loss = 0.0
        
        self.State = SState([len(field),len(field[0])], 2)
        self.prevState = np.ones((1,self.STATE_DIM))
    
    def UpdateState(self, field ):
        s = np.array(field, dtype = np.float32).reshape((1,-1))
        self.State.push_s(s)

    def get_action_value(self, state):
        x = Variable(state.reshape((1, -1)))
        return self.model.predict(x).data[0]
        
    def get_greedy_action(self, state):
        action_index = np.argmax(self.get_action_value(state))
        return action_index
        
    def reduce_epsilon(self):
        self.epsilon -= 1.0/2000
        self.epsilon = max(0.02, self.epsilon) 
        
    def get_action(self,state,train):
        action = 0
        if train==True and np.random.random() < self.epsilon:
            action_index = np.random.randint(len(self.actions))
        else:
            action_index = self.get_greedy_action(state)
        indices = np.zeros_like(self.actions)
        indices[action_index] = 1
        return self.actions[action_index], indices

    def experience(self,x):
        if self.eMem.shape[0] > self.memSize:
            self.eMem[int(self.memPos%self.memSize)] = x
            self.memPos+=1
        elif self.eMem.shape[0] == 0:
            self.eMem = x
        else:       
            self.eMem = np.vstack( (self.eMem, x) )

    def update_model(self):
        if len(self.eMem)<self.batch_num:
            return

        memsize     = self.eMem.shape[0]
        batch_index = np.random.permutation(memsize)[:self.batch_num]
        batch       = np.array(self.eMem[batch_index], dtype=np.float32).reshape(self.batch_num, -1)

        x = Variable(batch[:,0:self.STATE_DIM])
        targets = self.model.predict(x).data.copy()

        for i in range(self.batch_num):
            #[ state..., action, reward, seq_new]
            a = int(batch[i,self.STATE_DIM])
            r = batch[i, self.STATE_DIM+1]

            new_seq= batch[i,(self.STATE_DIM+2):(self.STATE_DIM*2+2)]

            targets[i,a]=( r + self.gamma * np.max(self.get_action_value(new_seq)))

        t = Variable(np.array(targets, dtype=np.float32).reshape((self.batch_num,-1))) 

        # ネットの更新
        self.model.zerograds()
        loss=self.model(x ,t)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()
        
#描画を管理するクラス
class Drawer():
    def __init__(self):
        self.rows = 0
        self.max_cols = []
        
    def draw_line(self, line):
        for c in line:
            if isinstance(c, list):
                print_23(c[1]+c[0], end = '')
            else:
                print_23( c , end = '')
        if len(self.max_cols)-1 >= self.rows:
            print_23 ("　" * self.max_cols[self.rows])
            self.max_cols[self.rows] = len(line)
        else:
            print_23()
            self.max_cols.append(len(line))
        self.rows += 1

    def draw_lines(self, lines):
        for c in lines.split("\n"):
            self.draw_line(c)

    def reset(self):
        print_23("\033[{}A".format(self.rows),end="")
        self.rows = 0

# テトリスクラス
class Tetris:
    def __init__(self, field_size = [10,15]):
        self.field_info = field_size #x,y
        self.field = [ [u'＿' for _ in range(field_size[0])] for _ in range(field_size[1] + 1) ]
        self.tmp_field = []

        self.blocks =  [
                        [['　','■','■'],
                         ['■','■','　']],

                        [['■','　','　'],
                         ['■','■','■']],

                        [['■','■','■'],
                         ['　','■','　']],
                        
                        [['■','■'],
                         ['■','■']],
                        
                        [['■','■','■','■']]]

        self.next_block, _ = self.get_new_block()

        self.movement = 0
        self.rotate_flag = 0
        self.end_flag = 0
        self.skip_flag = 0

        self.drawer = Drawer()

        self.score = 0
        self.pre_score = 0
        self.speed = 1.0

        self.agent = Agent(self.field)

    def ai_get_action(self):
        # Update States
        self.agent.UpdateState(self.tmp_field)
        state = np.hstack((self.agent.State.seq.reshape(1,-1), 
                    self.agent.prevActions.reshape(1,-1))).astype(np.float32)
        action, self.agent.prevActions = self.agent.get_action(state, True)
        return action

    def ai_learning(self):
        # 報酬計算(とりあえず点数の差分)
        reward = self.score - self.pre_score
        self.pre_score = self.score
        
        # Learning Step
        # 行動する前のフレームとその前のフレームを記憶してるけどいいの？
        self.agent.experience(np.hstack([
                    self.agent.prevState,
                    np.array([np.argmax(self.agent.prevActions)]).reshape(1,-1),
                    np.array([reward]).reshape(1,-1),
                    state
                ]))
        self.agent.prevState = state.copy()
        self.agent.update_model()
        self.agent.reduce_epsilon()

    def is_hit(self, block_pos):
        for b_row, f_row in zip(self.block,block_pos):
            for b, f in zip(b_row, f_row):
                if b == '■' and f == '■':
                    return True
        else:
            return False

    def draw_field(self, field = None ):
        field = self.field if field == None else field
        
        self.drawer.reset()
        
        #スコア表示
        self.drawer.draw_line("")
        self.drawer.draw_line("")
        self.drawer.draw_line("SCORE : {}".format(self.score))
        self.drawer.draw_line("")
        
        #次のブロック
        self.drawer.draw_line(" NEXT :")
        self.drawer.draw_line("")
        for row in self.next_block:
#           self.drawer.draw_line("  " + "".join(row))
            self.drawer.draw_line([" "] + row)
        #行数合わせ
        for _ in range(4-len(self.next_block)):
            self.drawer.draw_line("")
        
        #フィールド表示
        for row in field[:-1]:
#           self.drawer.draw_line(" " + "".join(row))
            self.drawer.draw_line([" "] + row)

    def draw_gameover(self):
        GO = list("ＧａｍｅＯｖｅｒ")
        GO[0] = [GO[0], "\033[35m\033[47m\033[1m"]
        
        str_size = len(GO)

        if self.field_info[0] > str_size:
            start_index = int((self.field_info[0]-str_size)/2)
            self.field[int(self.field_info[1]/2)][start_index:str_size+1] = GO

            self.field[int(self.field_info[1]/2)][str_size + 1] = [self.field[int(self.field_info[1]/2)][str_size + 1], "\033[39m\033[49m\033[0m"] 
        
        return self.field

    def is_gameover(self):
        if '■' in self.field[0]:
            self.draw_field(self.draw_gameover())
            return True
        else: return False

    def get_next_field(self, field = None, block = None):
        field = self.field if field == None else field
        block = self.block if block == None else block

        f = copy.deepcopy(field)
        for f_i, b_row in zip(range(self.row, self.row + len(block)), block):
            for f_j, b in zip(range(self.pos, len(block[0]) + self.pos), b_row):
                if b == '■':
                    f[f_i][f_j] = b
        return f
     
    def delete_complete_lines(self):
        ret = []
        deleteCount = 0
        for index, row in enumerate(self.field):
            if not '＿' in row:
                deleteCount += 1
            else:
                ret.append(row)
        self.score += deleteCount * 10
        return [['＿']*self.field_info[0] for _ in range(deleteCount)] + ret

    def wait_key_thread(self):
        while True:
            s = str(msvcrt.getwch())
            if s in 'j' or s in 'a':
                self.movement -= 1
            if s in 'l' or s in 'd':
                self.movement += 1
            if s in 'k' or s in 's':
                self.rotate_flag = 1
            if s in 'q':
                self.end_flag = 1
            if s in 'i' or s in 'w':
                self.skip_flag = 1

    def start_wait_key(self):
        self.thread = threading.Thread(target = self.wait_key_thread)
        self.thread.daemon = True
        self.thread.start()

    def wait_key(self):
        action = self.ai_get_action()
        if action == 5 or action == 6:
            self.rotate_flag = action - 4
        elif action == -5:
            self.skip_flag = 1
        else:
            self.movement = action

        self.block, self.block_size = self.get_block_size( self.rotate_block(self.block, self.rotate_flag))
        if self.is_hit( [rows[self.pos : self.block_size[0] + self.pos] 
                        for rows in self.field[self.row:self.row + self.block_size[1]]] ) \
                                or self.block_size[1] + self.row > self.field_info[1]: 
            self.block, self.block_size = self.get_block_size( self.rotate_block(self.block, 
                    1 if self.rotate_flag == 2 else 2 if self.rotate_flag == 1 else 0))

        if self.pos + self.movement >= 0 \
                and self.pos + self.movement <= self.field_info[0] - self.block_size[0] \
                and not self.is_hit(
                        [rows[self.pos + self.movement : self.block_size[0] + self.pos+self.movement] 
                            for rows in self.field[ self.row : self.row + self.block_size[1] ]] ):
            self.pos += self.movement

        if self.skip_flag:
            #while is_hit(get_block_size()):
            for index in range(self.row, self.field_info[1] - self.block_size[1] + 1):
                self.row = index
                if self.is_hit(self.get_block_pos()):
                    break
                self.tmp_field = self.get_next_field()

        self.rotate_flag = 0
        self.skip_flag = 0
        self.movement = 0

    def rotate_block(self, block, direction):
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
    
    def get_new_block(self):
        b = self.blocks[random.randint(0,len(self.blocks)-1)]
        b = self.rotate_block( b, random.randint( 0, 3 ))
        return self.get_block_size( b )

    def get_block_size(self, block):
        return block, [ len( block[0] ), len( block ) ]

    def get_block_pos(self, rows = 0, movement = 0):
        return [ rows[ self.pos + movement : self.block_size[0] + self.pos + movement ] 
                        for rows in self.field[ self.row + rows : self.row + rows + self.block_size[1] ]]

    def start(self):
#       print_23("\033[2", end = '')
        print_23("\033[0;0H", end = '')
        print_23("\033[2J", end = '')

        self.start_wait_key()
        self.draw_field(self.field)
        
        self.pretime = time.time()
        self.blockcount = 0
        self.hitflag = False

        # GameOverまでループ
        while True:
            # スピードの管理
            self.blockcount += 1
            if not int(self.blockcount % 5):self.speed -= 0.1
           
            # ブロック生成
            self.block, self.block_size = self.get_block_size(self.next_block)
            self.next_block, _ = self.get_new_block()
           
            # もろもろ初期化
            self.pos = int( ( self.field_info[0] - self.block_size[0] ) / 2 )
            self.row = 0
            self.tmp_field = self.get_next_field()
           
            # blockが何かにぶつかるまでループ
            while self.row <= self.field_info[1] - self.block_size[1]:
                self.nowtime = time.time()
                if self.nowtime - self.pretime > self.speed:
                    self.row += 1
                    self.pretime = self.nowtime
                self.wait_key()
                
                # ブロックの当たり判定
                block_pos = self.get_block_pos()
                if self.is_hit(block_pos) or self.end_flag:
                    self.field = self.tmp_field
                    self.draw_field()
                    break
                else:
                    self.tmp_field = self.get_next_field()
                    self.draw_field(self.tmp_field)

                # 学習
                self.ai_learning()
            else:
                self.row = self.field_info[1] - self.block_size[1]
                self.field = self.get_next_field()
            self.field = self.delete_complete_lines()
            self.draw_field()
            if self.is_gameover() or self.end_flag:
                break

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "TETRIS")

    tetris = Tetris([10,15])
    tetris.start()

