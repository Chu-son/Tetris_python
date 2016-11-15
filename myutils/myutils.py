#! /usr/bin/env python
#-*- coding:utf-8 -*-

class _Getch:
    """Gets a single character from standard input.  Does not echo to the
screen."""
    def __init__(self):
        try:
            self.impl = self._GetchWindows()
        except ImportError:
            self.impl = self._GetchUnix()

    def __call__(self): return self.impl()


    class _GetchUnix:
        def __init__(self):
            import tty, sys, termios

        def __call__(self):
            import sys, tty, termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                #tty.setraw(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                ch = sys.stdin.read(1)
            finally:
                #termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                termios.tcsetattr(fd, termios.TCSANOW, old_settings)
            return ch


    class _GetchWindows:
        def __init__(self):
            import msvcrt

        def __call__(self):
            import msvcrt
            return msvcrt.getwchreak
