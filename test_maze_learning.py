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

import re
import math
import numpy as np
import chainer
from chainer import cuda, optimizers, FunctionSet, Variable, Chain, serializers
import chainer.functions as F

class Drawer(object):
    def __init__(self):
        self.rows = 0
        self.max_cols = []
        self.delete_space = ' '

    def draw_char(self, char):
        if not isinstance(char, str):
            char = str(char)
        if char in '\n': self.rows += 1

        print(char, end = '')

    def draw_line(self, line):
        if not isinstance(line, str):
            line = str(line)
        for c in line:
            self.draw_char(c)

        if len(self.max_cols) - 1 >= self.rows:
            print(self.delete_space * self.max_cols[self.rows], end = '')
            self.max_cols[self.rows] = len(line)
        else:
            self.max_cols.append(len(line))

        print('')
        self.rows += 1

    def reset(self):
        print("\033[{}A".format(self.rows), end = '')
        self.rows = 0

class ChainInfo(Chain):
    def __init__(self, **links):
        super().__init__()

        self.l = links

        for name, link in self.l.items():
            self.add_link(name, link)

    def get_chain_info(self):
        links = self._sort_links()
        ret = ""
        for name, link in links:
            ret += "{}:({},{})\n".format(name,len(link.W.data[0]),len(link.W.data))
        return ret

    def _sort_links(self):
        links = [[name, link] for name, link in self.l.items()]
        sort_list = [[re.search("[a-z A-Z]*", name).group(), (re.search("[0-9]+", name)), name, link] for name, link in links]
        sort_list.sort(key = lambda x:(x[0],int(x[1].group())) if x[1] != None else (x[0],0))

        ret_list = [[name, link] for _,_, name, link in sort_list]

        return ret_list

class Q(ChainInfo):
    def __init__(self, state_dim, action_num ):
        super(Q, self).__init__(
            l1=F.Linear(state_dim, 256),
            l2=F.Linear(256, 512),
            l3=F.Linear(512, 1024),
            l4=F.Linear(1024, 2048),
            q_value=F.Linear(2048, action_num)
        )
    def __call__(self, x, t):
        return F.mean_squared_error(self.predict(x, train=True), t)
        
    def predict(self, x, train = False):
#       h2 = F.dropout(F.relu(self.l2(h1)),train=train)
        h1 = F.leaky_relu(self.l1(x))
        h2 = F.leaky_relu(self.l2(h1))
        h3 = F.leaky_relu(self.l3(h2))
        h4 = F.leaky_relu(self.l4(h3))
        y = self.q_value(h4)
        return y


class Agent(object):
    def __init__(self, field, epsilon = 1.0):

        self.FRAME_NUM = 1 #過去何フレームの状態を記憶するか
        self.PREV_ACTIONS_NUM = self.FRAME_NUM # 過去何回の行動を記憶するか

        self.FIELD_SIZE = [len(field[0]), len(field)]
        self.STATE_DIM = self.FIELD_SIZE[0] * self.FIELD_SIZE[1] * self.FRAME_NUM + self.PREV_ACTIONS_NUM
        
        # 行動
        self.actions = range(4) # 前後左右移動
        
        # DQN Model
        print("Network")
        print("In:{}".format(self.STATE_DIM))
        print("Out:{}\n".format(len(self.actions)))
        self.model = Q(self.STATE_DIM, len(self.actions))
        if gpu_flag >= 0:
            self.model.to_gpu()

        print(self.model.get_chain_info())

        self.model_target = copy.deepcopy(self.model)

#       self.optimizer = optimizers.RMSpropGraves(lr=0.00025,alpha=0.95,momentum=0.95,eps=0.0001)
        self.optimizer = optimizers.Adam()
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
        self.initial_exploration = 10**3
        self.target_model_update_freq = 10**3
        self.epsilon_decrement = 1.0/10**5

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
        return self.model.predict(x, self.is_train).data[0]
        
    def get_greedy_action(self, state):
        action_index = np.argmax(self.get_action_value(state))
        return action_index
        
    def reduce_epsilon(self):
        if not self.is_train:return
        self.epsilon -= self.epsilon_decrement
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
#       loss = F.mean_squared_error(Q, t)
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
        

# "S": Start地点, "#": 壁, "数値": 報酬
RAW_Field = [
[0,0,0,-0.5,0],
[0,-0.5,0,0,0],
[0,-0.5,0,-0.5,0],
[0,0,0,-0.5,0],
[0,-0.5,0,0,1.0],
]

class Field(object):
    """ Fieldに関するクラス """
    def __init__(self, raw_field=RAW_Field):
            self.field = raw_field
            self.field_size = (len(self.field[0]),len(self.field))
            self.now_coord = self.get_start_point()
            self.player_val = 0.01

    def display(self, point=None, drawer = Drawer()):

            """ Fieldの情報を出力する. """
            field = copy.deepcopy(self.get_state())

            drawer.draw_line ("----- Dump Field: {} -----".format( str(self.now_coord)))
            for line in field:
                    drawer.draw_char("\t")
                    for val in line:
                        if val == self.player_val: drawer.draw_char('@') 
                        elif val < 0: drawer.draw_char('#') 
                        elif val > 0: drawer.draw_char('G') 
                        else: drawer.draw_char('_') 
                    drawer.draw_line("")

    def move(self, direction):
            dx = 0
            dy = 0
            exception_penalty = 0.0

            if direction == 0: dx -= 1
            elif direction == 1: dx += 1
            elif direction == 2: dy -= 1
            elif direction == 3: dy += 1

            if self.field_size[1] <= self.now_coord[1] + dy or self.now_coord[1] + dy < 0: 
                exception_penalty = -1.0
                dy = 0
            if self.field_size[0] <= self.now_coord[0] + dx or self.now_coord[0] + dx < 0:
                exception_penalty = -1.0
                dx = 0

            self.now_coord = ( self.now_coord[0] + dx, self.now_coord[1] + dy ) 

            return self.get_reward() if exception_penalty == 0.0 else exception_penalty


    def get_reward(self):
            x, y = self.now_coord
            try:
                     return float(self.field[y][x])

            except ValueError:
                    if self.field_data[y][x] == "S": return 0.0, False # start地点の時
                    sys.exit("Field.get_val() ERROR: 壁を指定している.(x, y)=(%d, %d)" % (x, y))

    def get_state(self):
            x, y = self.now_coord
            ret_state = copy.deepcopy(self.field)
            ret_state[y][x] = self.player_val
            return ret_state

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
        self.reward = 0.0

        self.drawer = Drawer()

        self.init_learning()

    def init_learning(self):
        if QLearning.agent == None:
            QLearning.agent = Agent(self.Field.get_state())
#           QLearning.agent = Agent([self.Field.now_coord])
        self.container = QLearning.agent.get_container()

    def learn(self, greedy_flg=False):
        """ 1エピソードを消化する. """
        self.Field.restart()
        self.ep_end_flag = False
        if greedy_flg: QLearning.agent.is_train = False

        count = 0
        total_reward = 0
        while True:
            self.drawer.reset()
            count += 1
            action = self.get_action()
            self.reward = self.Field.move(action)
            total_reward += self.reward

            #if greedy_flg or int(QLearning.agent.epsilon * 1.0 / QLearning.agent.epsilon_decrement) % (1.0/QLearning.agent.epsilon_decrement / 10.0) == 0.0:
            self.drawer.draw_line("action:{}".format(action))
            self.drawer.draw_line("reward:{}".format(self.reward))
            self.drawer.draw_line("action count:{}".format(count))
            self.Field.display(action, self.drawer)

            if self.ep_end_flag or count % 10**3 == 0 :
                self.update_model()
                e = QLearning.agent.epsilon
                self.drawer.draw_line(e)
                self.drawer.draw_line(total_reward)
                self.drawer.draw_line(count)
                break # finish this episode

    def get_action(self):
        # Update States
        self.container.push_s(self.Field.get_state())
#       self.container.push_s([self.Field.now_coord])

        if len(self.container.prevActions) != 0:
            self.state = np.hstack((self.container.seq.reshape(1,-1), 
                                    self.container.prevActions.reshape(1,-1))).astype(np.float32)
        else:
            self.state = np.hstack(self.container.seq.reshape(1,-1)).astype(np.float32)

        s = cuda.to_gpu(self.state) if gpu_flag >= 0 else self.state
        action = QLearning.agent.get_action(s)
        self.container.push_prev_actions(action)

        # 前回の行動による報酬計算
        reward = self.reward
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
        self.container.prevAction = action
        
        return action

    def update_model(self):
        QLearning.agent.update_model()

        if QLearning.agent.initial_exploration < QLearning.times:
            QLearning.agent.reduce_epsilon()
        if QLearning.agent.initial_exploration < QLearning.times and QLearning.times % QLearning.agent.target_model_update_freq == 0:
            QLearning.agent.target_model_update()

def start_learning():
    print("\n\n")
    # create QLearning object
    QL = QLearning(Field())
    # Learning Phase
#   while QL.agent.epsilon > 0.1:
    print("\n")
    while True:
        QL.learn() # Learning 1 episode
    # After Learning
    QL.learn(greedy_flg=True) # 学習結果をgreedy法で行動選択させてみる

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(add_help=False, description = "MAZE Q-Learning")
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

