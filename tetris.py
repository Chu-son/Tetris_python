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
        print(s,end = end)
#       print s,
    else:
        print(s)

class SState(object):
    def __init__(self, field_size, frame_num):
        self.frame_num = frame_num
        self.seq = np.ones((frame_num, field_size[0] * field_size[1]), dtype=np.float32)
        
    def push_s(self, state):
        self.seq[1:self.frame_num] = self.seq[0:self.frame_num-1]
        self.seq[0] = state
        
    def fill_s(self, state):
        for i in range(0, self.frame_num):
            self.seq[i] = state

class Q(Chain):
    def __init__(self, state_dim, action_num ):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 3000),
            l2=F.Linear(3000, 3000),
            l3=F.Linear(3000, 512),
            q_value=F.Linear(512, action_num)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        y = self.q_value(h3)
        return y


class Agent():
    def __init__(self, field, epsilon = 1.0):
        self.FRAME_NUM =10 
        self.PREV_ACTIONS_NUM = 10
        self.STATE_DIM = len(field) * len(field[0]) * self.FRAME_NUM + self.PREV_ACTIONS_NUM
        
        # 行動
        self.actions = range(-3, 5) # 左右移動(最大2),回転,スキップ
        self.prevActions = np.zeros_like(range(self.PREV_ACTIONS_NUM))
        
        # DQN Model
        self.model = Q(self.STATE_DIM, len(self.actions))
        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025,alpha=0.95,momentum=0.95,eps=0.0001)
        self.optimizer.setup(self.model)
        
        self.epsilon = epsilon
        
        # 経験関連
        self.memPos = 0 
        self.memSize = 10**5
        self.eMem = [np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32)]

        # 学習関連のパラメータ
        self.batch_num = 32
        self.gamma = 0.99
        self.loss = 0.0
        self.initial_exploration = 10**4
        self.target_model_update_freq = 10**4
        
        self.State = SState([len(field),len(field[0])], self.FRAME_NUM)
        self.prevState = np.ones((1,self.STATE_DIM))
    
    def push_prev_actions(self, action):
        self.prevActions[1:self.PREV_ACTIONS_NUM] = self.prevActions[:-1]
        self.prevActions[0] = action

    def UpdateState(self, field ):
        f = []
        for rows in field:
            for state in rows:
                f.append(0 if state=='＿' else 1)
        s = np.array(f, dtype = np.float32).reshape((1,-1))
        self.State.push_s(s)

    def get_action_value(self, state):
        x = Variable(state.reshape((1, -1)))
        return self.model.predict(x).data[0]
        
    def get_greedy_action(self, state):
        action_index = np.argmax(self.get_action_value(state))
        return action_index
        
    def reduce_epsilon(self):
        self.epsilon -= 1.0/10**6
        self.epsilon = max(0.1, self.epsilon) 
        
    def get_action(self,state,train):
        action = 0
        if train==True and np.random.random() < self.epsilon:
            action_index = np.random.randint(len(self.actions))
        else:
            action_index = self.get_greedy_action(state)
        return self.actions[action_index]

    def experience(self, prev_state, action, reward, state):
        index = int(self.memPos%self.memSize)
        
        self.eMem[0][index] = prev_state
        self.eMem[1][index] = action
        self.eMem[2][index] = reward
        self.eMem[3][index] = state
        
        self.memPos+=1

    def update_model(self):
        batch_index = np.random.permutation(self.memSize)[:self.batch_num]
        prev_state  = np.array(self.eMem[0][batch_index], dtype=np.float32)
        action      = np.array(self.eMem[1][batch_index], dtype=np.float32)
        reward      = np.array(self.eMem[2][batch_index], dtype=np.float32)
        state       = np.array(self.eMem[3][batch_index], dtype=np.float32)
        
        s = Variable(prev_state)
        Q = self.model.predict(s)

        s_dash = Variable(state)
        tmp = self.model_target.predict(s_dash)
        tmp = list(map(np.max, tmp.data))
        max_Q_dash = np.asanyarray(tmp,dtype=np.float32)
        target = np.asanyarray(Q.data,dtype=np.float32)

        for i in range(self.batch_num):
            tmp_ = np.sign(reward[i]) + self.gamma * max_Q_dash[i]
            action_index = self.action_to_index(action[i])
            target[i,action_index] = tmp_

        td = Variable(target) - Q 
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)
        
        zero_val = Variable(np.zeros((self.batch_num, len(self.actions)),dtype=np.float32))

        # ネットの更新
        self.model.zerograds()
        loss = F.mean_squared_error(td_clip, zero_val)
        self.loss = loss.data
        loss.backward()
        self.optimizer.update()

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.actions[index_of_action]

    def action_to_index(self, action):
        return self.actions.index(action)
        
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
        self.field = self.new_field()
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
        self.action = 0
        self.reward = 0
        self.times = 0

    def ai_get_action(self):
        # Update States
        self.agent.UpdateState(self.tmp_field)
        self.state = np.hstack((self.agent.State.seq.reshape(1,-1), 
                    self.agent.prevActions.reshape(1,-1))).astype(np.float32)
        action = self.agent.get_action(self.state, True)
        self.agent.push_prev_actions(action)
        # 学習
        self.ai_learning()

        return action

    def ai_learning(self):
        # 報酬計算(とりあえず点数の差分)
        self.reward += (self.score - self.pre_score) * 10
        self.pre_score = self.score
        
        # Learning Step
        # 行動する前のフレームとその前のフレームを記憶してるけどいいの？
        self.agent.experience(
                    self.agent.prevState,
                    self.agent.prevActions[0],
                    self.reward,
                    self.state
                )
        self.agent.prevState = self.state.copy()
        self.agent.update_model()

        if self.agent.initial_exploration < self.times:
            self.agent.reduce_epsilon()
        if self.agent.initial_exploration < self.times and self.times % self.agent.target_model_update_freq == 0:
            self.agent.target_model_update()

        self.reward = -50
        self.times += 1

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
        self.drawer.draw_lines(str(self.agent.epsilon))
        
        #フィールド表示
        for row in field:
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
        if '■' in self.field[0][1:-1]:
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
        for index, row in enumerate(self.field[:-1]):
            if not '＿' in row:
                deleteCount += 1
            else:
                ret.append(row)
        ret.append(self.field[-1])
        self.score += deleteCount * 10
        return [self.new_line() for _ in range(deleteCount)] + ret

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
        action = int(self.ai_get_action())
        if action == 3 or action == 4:
            self.rotate_flag = action - 2
        elif action == -3:
            self.skip_flag = 1
        else:
            self.movement = action
        self.action = action

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

    def new_field(self):
#       return [ ['＿' for _ in range(self.field_info[0])] for _ in range(self.field_info[1] + 1) ]
        ret = [ self.new_line() for _ in range(self.field_info[1] ) ]
        ret.append( str('■ ' + '■ '*self.field_info[0] + '■').split(' '))
        return ret

    def new_line(self):
        return str('■ ' + '＿ '*self.field_info[0] + '■').split(' ')

    def start(self):
        print_23("\033[0;0H", end = '')
        print_23("\033[2J", end = '')

        self.start_wait_key()
        self.draw_field(self.field)
        
        self.pretime = time.time()
        self.blockcount = 0

        # GameOverまでループ
        while self.end_flag == 0:
            # スピードの管理
#           self.blockcount += 1
#           if not int(self.blockcount % 5):self.speed -= 0.1
           
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
            else:
                self.row = self.field_info[1] - self.block_size[1]
                self.field = self.get_next_field()
            self.field = self.delete_complete_lines()
            self.draw_field()
            if self.is_gameover():
#               break
                self.field = self.new_field()
                self.speed = 1
                self.reward -= 100

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description = "TETRIS")

    tetris = Tetris([10,15])
    tetris.start()
