import alive_progress
import time
import os

#=============== Metrics functions ===============#
def perf_decorate(func, log=0):
    time_1 = time.time()
    func()
    time_2 = time.time()
    time_diff = time_2 - time_1
    print("Time elapsed:", time_diff, "seconds")
    if log:
        write_log()

#=============== Logging functions ===============#
def create_log():
    code_dir = os.path.dirname(__file__.replace('"\"metrics.py', ''))


def write_log(message, type):
    pass

#=============== Console functions ===============#
def print_clear(message):
    os.system('cls')
    print(message)

