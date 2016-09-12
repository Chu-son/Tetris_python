#-*- coding:utf-8 -*- 
from __future__ import print_function
import sys

import time
import copy
import threading
import msvcrt
import random
import argparse
import pickle
import matplotlib.pyplot as plt

import math
import numpy as np
import chainer
from chainer import cuda, optimizers, FunctionSet, Variable, Chain
import chainer.functions as F

class Q(Chain):
    def __init__(self, state_dim, action_num ):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 1000),
            l2=F.Linear(1000, 512),
            q_value=F.Linear(512, action_num)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        y = self.q_value(h2)
        return y


class Agent():
    def __init__(self, field, epsilon = 1.0):
        self.FRAME_NUM = 1
        self.PREV_ACTIONS_NUM = 0
        self.FIELD_SIZE = [len(field[0]), len(field)]
        self.STATE_DIM = self.FIELD_SIZE[0] * self.FIELD_SIZE[1] * self.FRAME_NUM + self.PREV_ACTIONS_NUM
        
        # 行動
        self.actions = range(4) # 前後左右移動
        
        # DQN Model
        self.model = Q(self.STATE_DIM, len(self.actions))
        if gpu_flag >= 0:
            self.model.to_gpu()

        self.model_target = copy.deepcopy(self.model)

        self.optimizer = optimizers.RMSpropGraves(lr=0.00025,alpha=0.95,momentum=0.95,eps=0.0001)
        self.optimizer.setup(self.model)
        
        self.epsilon = epsilon
        
        # 経験関連
        self.memPos = 0 
        self.memSize = 10**4
        self.eMem = [np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,1), dtype=np.float32),
                     np.zeros((self.memSize,self.STATE_DIM), dtype=np.float32)]

        # 学習関連のパラメータ
        self.batch_num = 32
        self.gamma = 0.99
        self.initial_exploration = 10**2
        self.target_model_update_freq = 10**2

        self.loss_list = []

        self.is_draw_graph = False
        self.is_train = True
        
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
    
    def get_container(self):
        return self.Container(self.FIELD_SIZE, self.FRAME_NUM, self.PREV_ACTIONS_NUM)

    def get_action_value(self, state):
        x = Variable(state.reshape((1, -1)))
        return self.model.predict(x).data[0]
        
    def get_greedy_action(self, state):
        action_index = np.argmax(self.get_action_value(state))
        return action_index
        
    def reduce_epsilon(self):
        if not self.is_train:return
        self.epsilon -= 1.0/10**4
        self.epsilon = max(0.1, self.epsilon) 
        
    def get_action(self,state):
        action = 0
        if self.is_train == True and np.random.random() < self.epsilon:
            action_index = np.random.randint(len(self.actions))
        else:
            action_index = cuda.to_cpu(self.get_greedy_action(state)) if gpu_flag >= 0 else self.get_greedy_action(state)
        return self.actions[action_index]

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
        Q = self.model.predict(s)

        s_dash = Variable(state)
        tmp = self.model_target.predict(s_dash)
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
        

def start_learning():
    # create QLearning object
    QL = QLearning(Field())
    # Learning Phase
    while QL.agent.epsilon > 0.1:
        QL.learn() # Learning 1 episode
    # After Learning
    QL.learn(greedy_flg=True) # 学習結果をgreedy法で行動選択させてみる

# "S": Start地点, "#": 壁, "数値": 報酬
RAW_Field = [
[0,0,0,-10,0],
[0,-10,0,0,0],
[0,-10,0,-10,0],
[0,0,0,-10,0],
[0,-10,0,0,100],
]


# 定数
ALPHA = 0.2 # LEARNING RATIO
GAMMA  = 0.9 # DISCOUNT RATIO
E_GREEDY_RATIO = 0.2
LEARNING_COUNT = 1000


class Field(object):
    """ Fieldに関するクラス """
    def __init__(self, raw_field=RAW_Field):
            self.field = raw_field
            self.field_size = (len(self.field[0]),len(self.field))
            self.now_coord = self.get_start_point()

    def display(self, point=None):

            """ Fieldの情報を出力する. """
            field = copy.deepcopy(self.field)

            print ("----- Dump Field: {} -----".format( str(point)))
            for line in field:
                    print("\t",end="")
                    for val in line:
                        if val == 1: print('@' ,end = "") 
                        elif val < 0: print('#' ,end = "") 
                        elif val > 0: print('G' ,end = "") 
                        else: print('_' ,end = "") 
                    print("\n")

    def move(self, direction):
            """ 引数で指定した座標から移動できる座標リストを獲得する. """
            dx = 0
            dy = 0

            if direction == 0: dx -= 1
            elif direction == 1: dx += 1
            elif direction == 2: dy -= 1
            elif direction == 3: dy += 1

            if self.field_size[1] >= self.now_coord[1] + dy or self.now_coord[1] + dy < 0: dy = 0
            if self.field_size[0] >= self.now_coord[0] + dx or self.now_coord[0] + dx < 0: dx = 0

            self.now_coord = ( self.now_coord[0] + dx, self.now_coord[1] + dy ) 

    def get_reward(self):
            """ 指定した座標のfieldの値を返す. エピソード終了判定もする. """
            x, y = self.now_coord
            try:
                     return float(self.field[y][x])

            except ValueError:
                    if self.field_data[y][x] == "S": return 0.0, False # start地点の時
                    sys.exit("Field.get_val() ERROR: 壁を指定している.(x, y)=(%d, %d)" % (x, y))

    def get_state(self):
            x, y = self.now_coord
            self.field[y][x] = 1
            return self.field

    def get_start_point(self):
            """ Field中の Start地点:"S" の座標を返す """
            return 0,0

    def restart(self):
            self.now_coord = self.get_start_point()


class QLearning(object):
    """ class for Q Learning """

    # クラス変数
    agent = None
    times = 0
    end_flag = 0

    def __init__(self, map_obj):
        self.Qvalue = {}
        self.Field = map_obj
        self.ep_end_flag = False

        self.init_learning()

    def init_learning(self):
        if QLearning.agent == None:
            QLearning.agent = Agent(self.Field.get_state())
        self.container = QLearning.agent.get_container()

    def learn(self, greedy_flg=False):
        """ 1エピソードを消化する. """
        self.Field.restart()
        self.ep_end_flag = False
        if greedy_flg: QLearning.agent.is_train = False

        #print "----- Episode -----"
        while True:
            action = self.get_action()
            self.Field.move(action)

            if greedy_flg:
                self.Field.display(action)
                print( "\tstate: {} -> action:{}\n").format(state, action)

            if self.ep_end_flag:
                self.update_model()
                e = QLearning.agent.epsilon
                print(e)
                break # finish this episode

    def get_action(self):
        # Update States
        self.container.push_s(self.Field.get_state())

        if len(self.container.prevActions) != 0:
            self.state = np.hstack((self.container.seq.reshape(1,-1), 
                                    self.container.prevActions.reshape(1,-1))).astype(np.float32)
        else:
            self.state = np.hstack(self.container.seq.reshape(1,-1)).astype(np.float32)

        s = cuda.to_gpu(self.state) if gpu_flag >= 0 else self.state
        action = QLearning.agent.get_action(s)
        self.container.push_prev_actions(action)

        # 前回の行動による報酬計算
        reward = self.Field.get_reward()
        if reward > 0: self.ep_end_flag = True

        QLearning.times += 1
        
        # Learning Step
        QLearning.agent.experience(
                    self.container.prevState,
                    self.container.prevAction,
                    reward,
                    self.state
                )
        self.container.prevState = self.state.copy()
        self.container.prevActon = action
        
        return action

    def update_model(self):
        QLearning.agent.update_model()

        if QLearning.agent.initial_exploration < QLearning.times:
            QLearning.agent.reduce_epsilon()
        if QLearning.agent.initial_exploration < QLearning.times and QLearning.times % QLearning.agent.target_model_update_freq == 0:
            QLearning.agent.target_model_update()


if __name__ == "__main__":
    plt.plot([0.0])
    plt.pause(0.01)
    plt.close()
    
    parser = argparse.ArgumentParser(add_help=False, description = "TETRIS")
    parser.add_argument("--help", action="help")
    parser.add_argument("-g","--gpu",type=int,default=-1)
    parser.add_argument("--half", type=bool, default=False)
    parser.add_argument("--mode", type=int, default=0, help="0:nomal tetris, 1:single learning, 2~:multiple learning")
    parser.add_argument("-w","--width" , type=int, default=10)
    parser.add_argument("-h","--height", type=int, default=15)
    parser.add_argument("--model", type=str, default='')
    args = parser.parse_args()

    gpu_flag = args.gpu
    if gpu_flag >= 0:
        cuda.check_cuda_available()
        chainer.Function.type_check_enable = False
        cuda.get_device(gpu_flag).use()
        xp = cuda.cupy
    else:
        xp = np

    start_learning()

