#! /usr/bin/env python
#-*- coding:utf-8 -*- 
from __future__ import print_function
import sys
import os

import time
import copy
import threading
import random
import argparse
import pickle
import matplotlib
matplotlib.use(matplotlib.get_backend(),warn = False)
import matplotlib.pyplot as plt

import math
import numpy as np
import chainer
from chainer import cuda, optimizers, FunctionSet, Variable, Chain
import chainer.functions as F

import warnings
warnings.filterwarnings("ignore")

from myutils.myutils import _Getch
from myutils.myjson import JsonAdapter
from myutils.mychainerutils import ChainInfo
import json


_getchar = _Getch()

#class Q(ChainInfo):
#    def __init__(self, state_dim, action_num ):
#        super(Q, self).__init__(
#            l1=F.Linear(state_dim, 512),
#            l2=F.Linear(512, 2048),
#            l3=F.Linear(2048, 3072),
#            l4=F.Linear(3072, 2048),
#            l5=F.Linear(2048, 512),
#            q_value=F.Linear(512, action_num)
#        )
#    def __call__(self, x, t):
#        return F.mean_squared_error(self.predict(x, train=True), t)
#        
#    def predict(self, x, train = False):
#        h1 = F.dropout(F.leaky_relu(self.l1(x)), train = train)
#        h2 = F.dropout(F.leaky_relu(self.l2(h1)), train = train)
#        h3 = F.dropout(F.leaky_relu(self.l3(h2)), train = train)
#        h4 = F.dropout(F.leaky_relu(self.l4(h3)), train = train)
#        h5 = F.dropout(F.leaky_relu(self.l5(h4)), train = train)
#        y = self.q_value(h5)
#        return y
#

#class Q(ChainInfo):
#    def __init__(self, state_dim, action_num ):
#        super(Q, self).__init__(
#            l1=F.Linear(state_dim, 512),
#            l2=F.Linear(512, 2048),
#            l3=F.Linear(2048, 3072),
#            l4=F.Linear(3072, 2048),
#            l5=F.Linear(2048, 512),
#            q_value=F.Linear(512, action_num)
#        )
#    def __call__(self, x, t):
#        return F.mean_squared_error(self.predict(x, train=True), t)
#        
#    def predict(self, x, train = False):
#        h1 = F.leaky_relu(self.l1(x))
#        h2 = F.leaky_relu(self.l2(h1))
#        h3 = F.leaky_relu(self.l3(h2))
#        h4 = F.leaky_relu(self.l4(h3))
#        h5 = F.leaky_relu(self.l5(h4))
#        y = self.q_value(h5)
#        return y

class Q(ChainInfo):
    def __init__(self, state_dim, action_num ):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 512),
            l2=F.Linear(512, 2048),
            l3=F.Linear(2048, 2048),
            l4=F.Linear(2048, 512),
            q_value=F.Linear(512, action_num)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 = F.leaky_relu(self.l4(h3))
        y = self.q_value(h4)
        return y

class Agent(JsonAdapter):
    def __init__(self, field, agentdata = None):
        self.FRAME_NUM = 1
        self.PREV_ACTIONS_NUM = self.FRAME_NUM

        self.FIELD_SIZE = [len(field[0]), len(field)]
        self.STATE_DIM = self.FIELD_SIZE[0] * self.FIELD_SIZE[1] * self.FRAME_NUM + self.PREV_ACTIONS_NUM
        
        # 行動
        self.actions = list(range(-3, 5)) # 左右移動(最大2),回転,スキップ
        
        # DQN Model
        self.model = Q(self.STATE_DIM, len(self.actions)) if agentdata is None else agentdata.model
        if gpu_flag >= 0:
            self.model.to_gpu()

        self.model_target = copy.deepcopy(self.model)

#       self.optimizer = optimizers.RMSpropGraves(lr=0.00025,alpha=0.95,momentum=0.95,eps=0.0001)
        self.optimizer = optimizers.Adam() if agentdata is None else agentdata.optimizer
        self.optimizer.setup(self.model)

        self.epsilon = 1.0 
        
        # 経験関連
        self.memPos = 0 
        self.memSize = 10**6
        self.eMem = [np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32)]
        if not agentdata is None:self.eMem = agentdata.experience

        # 学習関連のパラメータ
        self.batch_num = 32
        self.gamma = 0.99
        self.initial_exploration = 10**4
        self.target_model_update_freq = 10**4
        self.epsilon_decrement = 1.0 / 10**5
        self.min_epsilon = 0.1

        self.loss_list = []

        self.is_draw_graph = False
        self.is_train = True

        if not agentdata is None:
            self.serialize(agentdata.params)

        print("Network")
#       print("In:{}".format(self.STATE_DIM))
#       print("Out:{}\n".format(len(self.actions)))
        print(self.model.get_chain_info_str())
        self.model.set_optimizer(self.optimizer)
        print("Optimizer")
        print(self.model.get_optimizer_name())
        print()
        
    class Container():
        def __init__(self, field_size, frame_num, prev_action_num):
            self.frame_num = frame_num
            self.prev_action_num = prev_action_num
            self.seq = np.ones((frame_num, field_size[0] * field_size[1]), dtype=np.float32)
            self.prevActions = np.zeros_like(range(prev_action_num))
            self.prevAction = 0
            self.prevState = np.ones((1,field_size[0] * field_size[1] * frame_num + prev_action_num))
            
        def push_s(self, state):
            if len(self.seq) == 0:return
            state = np.array(state, dtype = np.float32).reshape((1,-1))
            self.seq[1:self.frame_num] = self.seq[0:self.frame_num-1]
            self.seq[0] = state
            
        def push_prev_actions(self, action):
            if len(self.prevActions) == 0: return
            self.prevActions[1:self.prev_action_num] = self.prevActions[:-1]
            self.prevActions[0] = action

    class AgentData():
        def __init__(self, model, experience, params, optimizer):
            self.model = model
            self.experience = experience
            self.params = params
            self.optimizer = optimizer
    
    def get_container(self):
        return self.Container(self.FIELD_SIZE, self.FRAME_NUM, self.PREV_ACTIONS_NUM)

    def get_action_value(self, state):
        x = Variable(state.reshape((1, -1)))
        return self.model.predict(x,self.is_train).data[0]
        
    def get_greedy_action(self, state):
        action_index = np.argmax(self.get_action_value(state))
        return action_index
        
    def reduce_epsilon(self):
        if not self.is_train:return
        self.epsilon -= self.epsilon_decrement
        self.epsilon = max(self.min_epsilon, self.epsilon) 
        
    def get_action(self,state):
        action = 0
        is_random = False
        ep = self.epsilon if self.is_train else self.min_epsilon
#       ep = self.epsilon if self.is_train else 0

        if np.random.random() < ep:
            action_index = np.random.randint(len(self.actions))
            is_random = True
        else:
            action_index = cuda.to_cpu(self.get_greedy_action(state)) if gpu_flag >= 0 else self.get_greedy_action(state)
        return self.actions[action_index], is_random

    def experience(self, prev_state, action, reward, state):
        if not self.is_train:return

        if self.memPos < self.memSize:
            index = int(self.memPos%self.memSize)
            
            self.eMem[0][index] = prev_state
            self.eMem[1][index] = action
            self.eMem[2][index] = reward
            self.eMem[3][index] = state
            
            self.memPos+=1
        else:
            index = random.randint(0, self.memSize - 1)
            
            self.eMem[0][index] = prev_state
            self.eMem[1][index] = action
            self.eMem[2][index] = reward
            self.eMem[3][index] = state

    def update_model(self):
        if not self.is_train:return

        batch_index = np.random.permutation(self.memSize)[:self.batch_num]
        prev_state  = xp.array(self.eMem[0][batch_index], dtype=xp.float32)
        action      = xp.array(self.eMem[1][batch_index], dtype=xp.float32)
        reward      = xp.array(self.eMem[2][batch_index], dtype=xp.float32)
        state       = xp.array(self.eMem[3][batch_index], dtype=xp.float32)
        
        s = Variable(prev_state)
        Q = self.model.predict(s, self.is_train)

        s_dash = Variable(state)
        tmp = self.model_target.predict(s_dash, self.is_train)
        tmp = list(map(xp.max, tmp.data))
        max_Q_dash = xp.asanyarray(tmp,dtype=xp.float32)
        target = xp.asanyarray(copy.deepcopy(Q.data),dtype=xp.float32)

        for i in range(self.batch_num):
            tmp_ = xp.sign(reward[i]) + self.gamma * max_Q_dash[i]
            action_index = self.action_to_index(action[i])
            target[i,action_index] = tmp_

        td = Variable(target) - Q 
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        #td_clip = td
        
        zero_val = Variable(xp.zeros((self.batch_num, len(self.actions)),dtype=xp.float32))

        # ネットの更新
        self.model.zerograds()
        
        loss = F.mean_squared_error(td_clip, zero_val)
#       t = Variable(target)
#       loss = F.mean_squared_error(t, Q)
        loss.backward()
        self.optimizer.update()

        self.loss_list.append(loss.data.tolist())
        self.draw_graph(self.loss_list)

    def target_model_update(self):
        self.model_target = copy.deepcopy(self.model)

    def index_to_action(self, index_of_action):
        return self.actions[index_of_action]

    def action_to_index(self, action):
        return self.actions.index(action)

    def draw_graph(self, plotlist):
        if not self.is_draw_graph:
            plt.close()
            return
        plt.plot(plotlist, 'g')
        plt.pause(0.01)
        
#描画を管理するクラス
class Drawer():
    def __init__(self, is_half, is_draw):
        self.is_half = is_half
        self.is_draw = is_draw
        self.rows = 0
        self.max_cols = []

        self.draw_char = ['_','#',' '] if is_half else ['＿','■','　']
        self.delete_space = ' ' if is_half else '　'

    def print_(self, c):
        if isinstance(c, str):
            print(c, end = '')
        else:
            print(self.draw_char[c], end = '')
        
    def draw_line(self, line):
        if not self.is_draw:return

        for c in line:
            if isinstance(c, list):
                for c_ in c:self.print_(c_)
            else:
                self.print_(c)
        if len(self.max_cols)-1 >= self.rows:
            print (self.delete_space * (self.max_cols[self.rows] - len(line)))
            self.max_cols[self.rows] = len(line)
        else:
            print()
            self.max_cols.append(len(line))
        
        self.rows += 1

    def reset(self):
        if not self.is_draw:return
        print("\033[{}A".format(self.rows),end="")
        self.rows = 0

class RewardCalculator():
    def __init__(self):
        self.reset()

        max_point = 2 # 一回のアクションで獲得できる最大ポイント
        self.base_reward = 1.0 / max_point # 1ポイント分の報酬.最大ポイントで1.0になる

    def exception(self):
        self.exception_penalty_flag = True

    def gameover(self):
        self.gameover_penalty_flag = True

    def add_point(self, point = 1):
        self.total_point += point

    def get_reward(self):
        reward = -0.05
        if self.gameover_penalty_flag:
            reward = -1.0
#        elif self.exception_penalty_flag:
#            reward = -0.5
        else:
            reward = self.total_point * self.base_reward

        absreward = abs(reward)
        if absreward > 1.0:
            reward = max(1.0,absreward) * reward/absreward

        self.reset()
        return reward

    def reset(self):
        self.exception_penalty_flag = False
        self.gameover_penalty_flag = False
        
        self.total_point = 0


# テトリスクラス
class Tetris:
    # クラス変数
    agent = None
    experience_times = 0
    model_update_times = 0
    total_score = 0
    end_flag = 0

    # クラスメソッド
    @classmethod
    def cm_wait_key_thread(cls):
        stop_flag = False
        while not stop_flag:
            s = str(_getchar())
            if s in 'q':
                Tetris.end_flag = 1
                stop_flag = True

            if s in 'g':
                if Tetris.agent == None:return
                Tetris.agent.is_draw_graph = not Tetris.agent.is_draw_graph

            if s in 't':
                if Tetris.agent == None:return
                Tetris.agent.is_train = not Tetris.agent.is_train

    @classmethod
    def cm_start_wait_key(cls):
        Tetris.thread = threading.Thread(target = Tetris.cm_wait_key_thread)
        Tetris.thread.daemon = True
        Tetris.thread.start()

    @classmethod
    def cm_ai_learning(cls):
        Tetris.agent.update_model()
        Tetris.model_update_times += 1

        if Tetris.agent.initial_exploration < Tetris.experience_times:
            Tetris.agent.reduce_epsilon()
        if Tetris.agent.initial_exploration < Tetris.experience_times and Tetris.model_update_times % Tetris.agent.target_model_update_freq == 0:
            Tetris.agent.target_model_update()

    # インスタンスメソッド
    def __init__(self, field_size = [10,15], is_half = False, is_draw = True):
        self.field_info = field_size #x,y
        self.field = self.new_field()
        self.tmp_field = []

        self.blocks =  [
#                        [[2,1,1],
#                         [1,1,2]],
#
#                        [[1,2,2],
#                         [1,1,1]],
#
#                        [[1,1,1],
#                         [2,1,2]],
#                        
                        [[1,1],
                         [1,1]],
#                        
#                        [[1,1,1,1]]
                        ]

        self.next_block, _ = self.get_new_block()

        self.movement = 0
        self.rotate_flag = 0
        self.skip_flag = 0

        self.is_half = is_half
        self.drawer = Drawer(is_half, is_draw)

        self.score = 0
        self.pre_score = 0
        self.default_speed = 1.0
        self.speed = self.default_speed

        self.action = 0
        self.rewardCalclator = RewardCalculator()

        self.pretime = time.time()
        self.blockcount = 0

        self.is_random = False

    def set_draw(self, is_draw):
        self.drawer.is_draw = is_draw
        self.drawer.reset()

    def reset_speed(self, sp):
        self.speed = sp
        self.default_speed = sp

    def init_learning(self, agent = None):
        if Tetris.agent == None:
            Tetris.agent = Agent(self.field) if agent is None else Agent(self.field,agent)
        self.container = Tetris.agent.get_container()

    def ai_get_action(self):
        # Update States
        self.container.push_s(self.tmp_field)

        if len(self.container.prevActions) != 0:
            self.state = np.hstack((self.container.seq.reshape(1,-1), 
                                    self.container.prevActions.reshape(1,-1))).astype(np.float32)
        else:
            self.state = np.hstack(self.container.seq.reshape(1,-1)).astype(np.float32)

        s = cuda.to_gpu(self.state) if gpu_flag >= 0 else self.state
        action, self.is_random = Tetris.agent.get_action(s)
        self.container.push_prev_actions(action)

        # 前回の行動による報酬計算
        if Tetris.agent.is_train:
            reward = self.rewardCalclator.get_reward()
            Tetris.total_score += self.score - self.pre_score
            self.pre_score = self.score

            Tetris.experience_times += 1
        
            Tetris.agent.experience(
                        self.container.prevState,
                        self.container.prevAction,
                        reward,
                        self.state
                    )
        self.container.prevState = self.state.copy()
        self.container.prevAction = action

        return action

    def is_hit(self, block_pos):
        for b_row, f_row in zip(self.block,block_pos):
            for b, f in zip(b_row, f_row):
                if b == 1 and f == 1:
                    return True
        else:
            return False

    def draw_field(self, field = None ):
        field = self.field if field == None else field
        
        self.drawer.reset()
        
        #スコア表示
        self.drawer.draw_line("")
        self.drawer.draw_line("")
        self.drawer.draw_line("TOTAL SCORE : {}".format(Tetris.total_score))
        self.drawer.draw_line("")
        
        #次のブロック
        self.drawer.draw_line(" NEXT :")
        self.drawer.draw_line("")
        for row in self.next_block:
#           self.drawer.draw_line("  " + "".join(row))
            self.drawer.draw_line(row)
        #行数合わせ
        for _ in range(4-len(self.next_block)):
            self.drawer.draw_line("")
        mem = "Train:" + str(Tetris.agent.is_train) if Tetris.agent != None else ''
        self.drawer.draw_line(mem)
        self.drawer.draw_line("Is random action:" + str(self.is_random))
        ep = "ep:" + str(Tetris.agent.epsilon) if Tetris.agent != None else ''
        self.drawer.draw_line(ep)
        
        #フィールド表示
        for row in field:
#           self.drawer.draw_line(" " + "".join(row))
            self.drawer.draw_line(row)

        self.drawer.draw_line("")
        self.drawer.draw_line("Press 'q' key if you want to exit.")

    def draw_gameover(self):
        GO = list("GameOver") if self.is_half else list("ＧａｍｅＯｖｅｒ")
        GO[0] = ["\033[35m\033[47m\033[1m",GO[0]]
        
        str_size = len(GO)

        if self.field_info[0] > str_size:
            start_index = int((self.field_info[0]-str_size)/2)
            self.field[int(self.field_info[1]/2)][start_index:str_size+1] = GO

            self.field[int(self.field_info[1]/2)][str_size + 1] = [self.field[int(self.field_info[1]/2)][str_size + 1], "\033[39m\033[49m\033[0m"] 
        
        return self.field

    def is_gameover(self):
        if 1 in self.field[0][1:-1]:
            self.draw_field(self.draw_gameover())
            return True
        else: return False

    def get_next_field(self, field = None, block = None):
        field = self.field if field == None else field
        block = self.block if block == None else block

        f = copy.deepcopy(field)
        for f_i, b_row in zip(range(self.row, self.row + len(block)), block):
            for f_j, b in zip(range(self.pos, len(block[0]) + self.pos), b_row):
                if b == 1:
                    f[f_i][f_j] = b
        return f
     
    def delete_complete_lines(self):
        ret = []
        deleteCount = 0
        for index, row in enumerate(self.field[:-1]):
            if not 0 in row:
                deleteCount += 1
            else:
                ret.append(row)
        ret.append(self.field[-1])
        self.score += deleteCount * 10
        self.rewardCalclator.add_point(deleteCount)

        return [self.new_line() for _ in range(deleteCount)] + ret

    def wait_key_thread(self):
        while True:
            s = str(_getchar())
            if s in 'j' or s in 'a':
                self.movement -= 1
            if s in 'l' or s in 'd':
                self.movement += 1
            if s in 'k' or s in 's':
                self.rotate_flag = 1
            if s in 'q':
                Tetris.end_flag = 1
            if s in 'i' or s in 'w':
                self.skip_flag = 1

    def start_wait_key(self):
        self.thread = threading.Thread(target = self.wait_key_thread)
        self.thread.daemon = True
        self.thread.start()

    def wait_key(self):
        if Tetris.agent != None:
            action = int(self.ai_get_action())
            if action == 3 or action == 4:
                self.rotate_flag = action - 2
            elif action == -3:
                self.skip_flag = 1
            else:
                self.movement = action
            self.action = action

        # 回転できるか
        self.block, self.block_size = self.get_block_size( self.rotate_block(self.block, self.rotate_flag))
        if self.is_hit( [rows[self.pos : self.block_size[0] + self.pos] 
                        for rows in self.field[self.row:self.row + self.block_size[1]]] ) \
                                or self.block_size[1] + self.row > self.field_info[1]: 

            self.block, self.block_size = self.get_block_size( self.rotate_block(self.block, 
                    1 if self.rotate_flag == 2 else 2 if self.rotate_flag == 1 else 0))
            self.rewardCalclator.exception()

        # 移動できるか
        while self.movement != 0:
            if self.pos + self.movement >= 0 \
                    and self.pos + self.movement <= self.field_info[0] - self.block_size[0] + 1\
                    and not self.is_hit(
                            [rows[self.pos + self.movement : self.block_size[0] + self.pos+self.movement] 
                                for rows in self.field[ self.row : self.row + self.block_size[1] ]] ):
                self.pos += self.movement
                break
            else:
                # 可能な範囲で移動する
                absmovement = abs(self.movement)
                self.movement = (absmovement - 1) * self.movement//absmovement
#       else: self.rewardCalclator.exception()

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
        ret = [ self.new_line() for _ in range(self.field_info[1] ) ]
        ret.append(self.new_line(1))
        return ret

    def new_line(self, fill = 0):
        return [1] + [fill for _ in range(self.field_info[0])] + [1]

    def start(self):
        print("\033[0;0H", end = '')
        print("\033[2J", end = '')

        self.start_wait_key()

        # GameOverまでループ
        while Tetris.end_flag == 0:
            # スピードの管理
            self.blockcount += 1
            if not int(self.blockcount % 5):self.speed -= 0.1
            
            self.init_next_block()

            # blockが何かにぶつかるまでループ
            while self.row <= self.field_info[1] - self.block_size[1]:
                self.nowtime = time.time()
                if self.nowtime - self.pretime > self.speed:
                    self.row += 1
                    self.pretime = self.nowtime
                self.wait_key()
                
                # ブロックの当たり判定
                block_pos = self.get_block_pos()
                if self.is_hit(block_pos) or Tetris.end_flag:
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

    def init_next_block(self):
        # ブロック生成
        self.block, self.block_size = self.get_block_size(self.next_block)
        self.next_block, _ = self.get_new_block()
       
        # もろもろ初期化
        self.pos = int( ( self.field_info[0] - self.block_size[0] ) / 2 )
        self.row = 0
        self.tmp_field = self.get_next_field()

    def move_blocks(self):
        if not self.drawer.is_draw and Tetris.agent != None and not Tetris.agent.is_train:return
        self.nowtime = time.time()
        if self.nowtime - self.pretime > self.speed or Tetris.agent.is_train:
            self.row += 1
            self.pretime = self.nowtime
        self.wait_key() # AIの行動決定等もここでやる
        
        # ブロックの当たり判定
        hit_flag = False
        block_pos = self.get_block_pos()
        if self.is_hit(block_pos) or Tetris.end_flag:
            self.field = self.tmp_field
            self.draw_field()
            hit_flag = True
        else:
            self.tmp_field = self.get_next_field()
            self.draw_field(self.tmp_field)
        
        # ブロックに当たるか一番下に到達した場合
        if hit_flag or self.row >= self.field_info[1] - self.block_size[1]:
            if not hit_flag:
                self.row = self.field_info[1] - self.block_size[1]
                self.field = self.get_next_field()
        
            self.field = self.delete_complete_lines()
            self.draw_field()
            self.init_next_block()
            if self.is_gameover():
                self.field = self.new_field()
                self.speed = self.default_speed
                self.rewardCalclator.gameover()
                return False
        return True
    
import enum
class SaveLoadBase():
    formats_ = enum.Enum("Json","pickle")

    def __init__(self,dirname):
        self.dirname = dirname

    def save(self, data, dataname, format_):
        if format_ is formats_["Json"]:
            with open(self.dirname + '/' + dataname, "w") as f:
                f.write(data)
        elif format_ is formats_["pickle"]:
            with open(self.dirname + '/' + dataname, "wb") as f:
                pickle.dump(data, f)

    def load(self):
        pass


    
def start_learning(tetris_size, is_half, num_of_tetris, savedataname):
    print("\033[0;0H", end = '')
    print("\033[2J", end = '')

    # load data
    if savedataname != '':
        savedataname += '/'
        with open(savedataname + "model.model", "rb") as f:
            model = pickle.load(f)
        with open(savedataname + "optimizer.opt", "rb") as f:
            opt = pickle.load(f)
        with open(savedataname + "experience.exp", "rb") as f:
            exp = pickle.load(f)
        with open(savedataname + "agentparams.json", "r") as f:
            agentdata = json.load(f)
        with open(savedataname + "tetrisparams.json", "r") as f:
            tetrisdata = json.load(f)
        num_of_tetris = tetrisdata["num"]
        Tetris.total_score = tetrisdata["totalscore"]
        agent = Agent.AgentData(model,exp, agentdata, opt)

        print("Load {} data".format(savedataname))
    else:agent = None

    print(str(num_of_tetris) + " tetris\n")

    # prerare tetris list
    tetris_list = [ Tetris(tetris_size, is_half, False) for _ in range(num_of_tetris) ]
    tetris_list[0].set_draw(True)
    Tetris.cm_start_wait_key()

    # init
    for tetris in tetris_list:
        tetris.reset_speed(0)
        tetris.init_next_block()
        tetris.init_learning(agent)
    # learning loop
    while Tetris.end_flag == 0:
        for tetris in tetris_list:
            tetris.move_blocks()
        Tetris.cm_ai_learning()

    is_save_model = input("Do you want to save ?(y/n) => ")
    # save
    if 'y' in is_save_model or is_save_model == '':
        f_name = input("Directory name => ")
        f_name = "tetris" if f_name == "" else f_name
        if not os.path.exists(f_name):
            os.mkdir(f_name)
        f_name += "/"
        with open(f_name + "model.model", "wb") as f:
            pickle.dump(Tetris.agent.model.to_cpu(), f)
        with open(f_name + "optimizer.opt", "wb") as f:
            pickle.dump(Tetris.agent.optimizer, f)
        with open(f_name + "experience.exp", "wb") as f:
            pickle.dump(Tetris.agent.eMem, f)
        with open(f_name + "agentparams.json", "w") as f:
            f.write(Tetris.agent.to_json())
        with open(f_name + "tetrisparams.json", "w") as f:
            d = {"num":num_of_tetris,"totalscore":Tetris.total_score}
            json.dump(d,f,ensure_ascii=False,indent=4)

        print("Saved in " + f_name)

if __name__ == "__main__":
    plt.plot([0.0])
    plt.pause(0.01)
    plt.close()

    parser = argparse.ArgumentParser(add_help=False, description = "TETRIS")
    parser.add_argument("--help", action="help")
    parser.add_argument("-g","--gpu",type=int,default=-1)
    parser.add_argument("--half", type=bool, default=False)
    parser.add_argument("--mode", type=int, default=1, help="0:nomal tetris, 1:learning")
    parser.add_argument("-n","--num", type=int, default=1, help="Num of tetris")
    parser.add_argument("-w","--width" , type=int, default=10)
    parser.add_argument("-h","--height", type=int, default=15)
    parser.add_argument("-s","--savedata", type=str, default='')
    args = parser.parse_args()

    gpu_flag = args.gpu
    if gpu_flag >= 0:
        cuda.check_cuda_available()
        chainer.Function.type_check_enable = False
        cuda.get_device(gpu_flag).use()
        xp = cuda.cupy
    else:
        xp = np

    tetris_size = [args.width, args.height]
    if args.mode == 0:
        tetris = Tetris(tetris_size, args.half)
        tetris.start()
    else:
        start_learning(tetris_size, args.half, args.num, args.savedata)

