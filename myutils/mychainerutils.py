#! /usr/bin/env python
#-*- coding:utf-8 -*-

from chainer import Chain
import re

class ChainInfo(Chain):
    def __init__(self, **links):
        super().__init__()

        self.l = links
        self.opt = ''

        for name, link in self.l.items():
            self.add_link(name, link)

    def get_chain_info_str(self):
        links = self._sort_links()
        ret = ""
        for name, link in links:
            ret += "{}:{}({},{})\n".format(name,link.__class__.__name__, len(link.W.data[0]),len(link.W.data))
        return ret

    def _sort_links(self):
        links = [[name, link] for name, link in self.l.items()]
        sort_list = [[re.search("[a-z A-Z]*", name).group(), (re.search("[0-9]+", name)), name, link] for name, link in links]
        sort_list.sort(key = lambda x:(x[0],int(x[1].group())) if x[1] != None else (x[0],0))

        ret_list = [[name, link] for _,_, name, link in sort_list]

        return ret_list

    def set_optimizer(self, opt):
        self.opt = opt

    def get_optimizer_name(self):
        return self.opt.__class__.__name__

