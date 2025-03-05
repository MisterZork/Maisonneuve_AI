# This is where we keep a log of the AI's program. It's useful for debugging and for keeping track of the program's
# progress. We can also use it to compare the execution of the old and new versions of the code (or even different
# models of the neural network).
#----------------------------------------------------------------------------------------------------------------------#

#==================== Importation des modules ====================#
import time
import os

#==================== Initialisation des variables ====================#
log_file = "../logs.txt"
log = open(log_file, "w")
log.write(f"Log file created at {time.ctime()}\n")
log.close()

#==================== Fonctions ====================#
def log_message(message):
    log = open(log_file, "a")
    log.write(f"{time.ctime()} - {message}\n")
    log.close()

def log_error(message):
    log = open(log_file, "a")
    log.write(f"{time.ctime()} - ERROR: {message}\n")
    log.close()

def log_warning(message):
    log = open(log_file, "a")
    log.write(f"{time.ctime()} - WARNING: {message}\n")
    log.close()

def log_separator():
    log = open(log_file, "a")
    log.write("------------------------------------------------------------------\n")
    log.close()

def time_decorator(func):
    time_1 = time.time()
    func()
    time_2 = time.time()
    log_message(f"Function {func.__name__} took {time_2 - time_1:.2f} seconds)")