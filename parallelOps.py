import sysOps
import time
import alignOps
import clustOps
import libOps
import numpy
import os
import itertools
import subprocess
    
def get_next_open_task(task_log_file,task_input_file,first_task_arg = None):
    #form of task input file: 
    #Lines:     input_str1_task1;input_str2_task1;...\n
    #           input_str1_task2;input_str2_task2;...\n
    #form of task log file:
    #Lines:     input_str1_task1;input_str2_task1;...,time_start,time_end\n
    #           input_str1_task2;input_str2_task2;...,time_start,time_end\n
    #returns list of strings with task-input
    
    sysOps.delay_with_alertfile('_taskchange_inprog')
    my_TaskLog = sysOps.TaskLog()
    my_TaskLog.read_log_from_file(task_log_file)
        
    my_task = None
    time_start = None
    with open(sysOps.globaldatapath + task_input_file,'rU') as task_input_handle:
        for task_line in task_input_handle:
            my_task = task_line.strip('\n')
            if (first_task_arg == None or my_task.split(';')[0] == first_task_arg) and (my_task not in my_TaskLog.my_tasks):
                my_TaskLog.my_tasks.append(my_task)
                time_start = time.time()
                my_TaskLog.my_start_times.append(time_start)
                my_TaskLog.my_end_times.append(-1)
                my_TaskLog.my_queue.append('priority')
                my_TaskLog.write_log_to_file(task_log_file)
                my_task = my_task.split(';')
                break
            else:
                my_task = None
    
    sysOps.remove_alertfile('_taskchange_inprog')
    
    return [my_task,time_start]
            
def close_task(task_log_file, this_task, this_start):
    sysOps.delay_with_alertfile('_taskchange_inprog')
    
    my_TaskLog = sysOps.TaskLog()
    found_task = my_TaskLog.enter_end_time(task_log_file, this_task, this_start, time.time())
    
    sysOps.remove_alertfile('_taskchange_inprog')
            
    return found_task

