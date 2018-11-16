import csv
import sys
import time
import os
import os.path
import fileOps
import parseOps
import itertools
import masterProcesses

statuslogfilename = 'DEFAULTSTATUSLOGFILENAME'
globaldatapath = ''
maxruntimes = dict()

class ErrorLog:
    def __init__(self):
        self.my_times = []
        self.my_exceptions = []
    def append(self, time_exception):
        #appends list of lists with time and exception as first and second elements, respectively
        self.my_times = [time_exception[i][0] for i in range(len(time_exception))]
        self.my_exceptions = [time_exception[i][1] for i in range(len(time_exception))]
        
class TaskLog:
    def __init__(self):
        self.my_tasks = list()
        self.my_start_times = list()
        self.my_end_times = list()
        self.my_queue = list()
    def printme(self):
        print "PRINTING TASK"
        print str(self.my_tasks)
        print str(self.my_start_times)
        print str(self.my_end_times)
        print str(self.my_queue)
    def read_log_from_file(self,task_log_file):
        self.my_tasks = list()
        self.my_start_times = list()
        self.my_end_times = list()
        self.my_queue = list()
        num_open_tasks = 0
        try:
            with open(globaldatapath + task_log_file,'rU') as task_log_handle:
                for task_line in task_log_handle:
                    if len(task_line.strip('\n').split(',')) == 4:
                        [my_task,time_start,time_end,queue] = task_line.strip('\n').split(',')
                    else:
                        [my_task,time_start,time_end] = task_line.strip('\n').split(',')
                        throw_exception('Missing queue. Assigned default = priority')
                        queue = 'priority'
                        
                    self.my_tasks.append(my_task)
                    self.my_start_times.append(float(time_start))
                    self.my_end_times.append(float(time_end))
                    self.my_queue.append(queue)
                    if float(time_end)<0:
                        num_open_tasks += 1
                        
            return num_open_tasks
        except:
            throw_status('Log file ' + task_log_file + ' does not exist. Creating.')
            with open(globaldatapath + task_log_file,'w') as task_log_handle:
                return 0
            
    def pop_overdue(self,maxruntimes):
        current_time = time.time()
        overdue = [((current_time > (maxruntimes[this_queue]*1.2 + start_time)) and (end_time<0)) for [start_time,end_time,this_queue] in itertools.izip(self.my_start_times,self.my_end_times,self.my_queue)]
        
        rename_filename = None
        
        if True in overdue:
            [jobid, rename_filename] = self.my_tasks[overdue.index(True)].split(';')
            throw_status('Overdue job found: jobid = ' + str(jobid) + ' ... cancelling and returning to queue to restart')
            del self.my_tasks[overdue.index(True)]
            del self.my_start_times[overdue.index(True)]
            del self.my_end_times[overdue.index(True)]
            del self.my_queue[overdue.index(True)]
            
        return rename_filename
            
    def write_log_to_file(self,task_log_file, alertfile = None):
        if alertfile != None:
            delay_with_alertfile(alertfile)
        
        if len(self.my_tasks)>0 and len(self.my_queue) == 0:
            throw_status('Assigning priority-queue default to ' + globaldatapath + task_log_file)
            self.my_queue = ['priority']*len(self.my_tasks)
        
        with open(globaldatapath + task_log_file,'w') as task_log_handle:
            for task,start,end,queue in itertools.izip(self.my_tasks,self.my_start_times,self.my_end_times,self.my_queue):
                task_log_handle.write(','.join([task,str(start),str(end),queue]) + '\n')
                
        if alertfile != None:
            remove_alertfile(alertfile)
                
    def enter_end_time(self,task_log_file,this_task,this_start,new_end_time):
        
        self.read_log_from_file(task_log_file)
        found_task = False
                
        with open(globaldatapath + task_log_file,'w') as task_log_handle:
            for task,start,end,queue in itertools.izip(self.my_tasks,self.my_start_times,self.my_end_times,self.my_queue):
                if task==this_task and abs(start-this_start)<=1 and end<0:
                    task_log_handle.write(','.join([task,str(start),str(new_end_time),queue]) + '\n')
                    found_task = True
                else:
                    task_log_handle.write(','.join([task,str(start),str(end),queue]) + '\n')
                    
        if not found_task:
            
            print 'enter_end_time() returning false'
            print 'this_task = ' + str(this_task)
            print 'this_start = ' + str(this_start)
            print 'new_end_time = ' + str(new_end_time)
            print 'self.my_tasks = ' + str(self.my_tasks)
            print 'self.my_start_times = ' + str(self.my_start_times)
            print 'self.my_end_times = ' + str(self.my_end_times)
            print 'self.my_queue = ' + str(self.my_queue)
        
        if new_end_time<this_start:
            throw_exception('task ' + str(this_task) + ' written with start-time after end-time: ' + str(this_start) + '>' + str(new_end_time))
            
        return found_task
        
def organize_lib_settings(settings_template_filename, bc_list_filename, runfilename):
    #takes list of barcode (in numerical form) and generates new directories (when none exist) containing relevant libsettings.txt file and runfile.txt

    bcnames = []
    
    with open(globaldatapath + bc_list_filename,'rU') as infile:
        for myline in infile:
            bcnames.append(myline.strip('\n'))    
        
def find_missing_uxi_files(settingsfilename, output_prefix):
    settings_dict = fileOps.read_settingsfile_to_dictionary(settingsfilename)
    for_seqform = parseOps.parse_seqform(settings_dict['-seqform_for'][0])
    rev_seqform = parseOps.parse_seqform(settings_dict['-seqform_rev'][0])
    
    incomplete_consolidation = []
    
    uxi_index = 0
    for el in for_seqform:
        if el == 'U':
            my_uxi_filename = output_prefix + 'for_uxi' + str(uxi_index) + '.fasta'
            if not check_file_exists(my_uxi_filename):
                incomplete_consolidation.append(my_uxi_filename)
            uxi_index += 1
    
    uxi_index = 0     
    for el in rev_seqform:
        if el == 'U':
            my_uxi_filename = output_prefix + 'rev_uxi' + str(uxi_index) + '.fasta'
            if not check_file_exists(my_uxi_filename):
                incomplete_consolidation.append(my_uxi_filename)
            uxi_index += 1
    
    return incomplete_consolidation

def exitProgram():
    #add_nodes_running(-1,0,True)
    sys.exit()
    
def print_master_time():
    with open(globaldatapath + 'univ_time' , 'w') as outfile:
        outfile.write(str(time.time()))
        
def get_master_time():
    with open(globaldatapath + 'univ_time' , 'rU') as infile:
        for line in infile:
            return float(line)
    
def check_file_exists(filename):
    return os.path.isfile(globaldatapath + filename) 
            
def throw_exception(this_input):
    #throws exception this_input[0] to file-name this_input[1], if this_input[1] exists, or errorlog.csv otherwise

    if(type(this_input)==list and len(this_input)==2):
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if(type(this_input)==list):
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = globaldatapath + "errorlog.csv"        

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename,'a+') as csvfile:
        csvfile.write(my_datetime + '|' + statusphrase + '\n')

    print my_datetime + "|" + statusphrase

def throw_status(this_input):
    #throws status this_input[0] to file-name this_input[1], if this_input[1] exists, or statuslog.csv otherwise
    #if this_input[1] is global variable statuslogfilename, globaldatapath will already be incorporated to beginning of string, and therefore it is not included in call to file-open function

    if(type(this_input)==list and len(this_input)==2):
        statusphrase = this_input[0]
        statuslog_filename = this_input[1]
    else:
        if(type(this_input)==list):
            statusphrase = this_input[0]
        else:
            statusphrase = this_input
        statuslog_filename = globaldatapath + "statuslog.csv"        

    my_datetime = time.strftime("%Y/%m/%d %H:%M:%S")
    with open(statuslog_filename,'a+') as csvfile:
        csvfile.write(my_datetime + '|' + statusphrase + '\n')

    print my_datetime + "|" + statusphrase
    
def get_directory_and_file_list(myrunpath = ''):
    fullrunpath = globaldatapath + myrunpath
    if len(fullrunpath) == 0:
        fullrunpath = '.'
    while True:
        try:
            for dirname, dirnames, filenames in os.walk(fullrunpath):
                return [dirnames,filenames] #first level of directory hierarchy only
        except:
            throw_exception('Error during file/directory-readout. Re-trying.')

def initiate_statusfilename(prefix = '',make_file = False):
    #globaldatarunpath added directly to statuslogfilename
    global statuslogfilename
    fullprefix = prefix + 'statuslog'
    max_statuslog_index = 0
    [dirnames, filenames] = get_directory_and_file_list()
    for filename in filenames:
        if filename.startswith(fullprefix) and filename.endswith('.csv'):
            try:
                max_statuslog_index = max(max_statuslog_index, int(filename[len(fullprefix):(len(filename)-4)]))
            except: #no integer-form index substring
                pass
            
    statuslogfilename = globaldatapath + fullprefix + str(max_statuslog_index + 1) + ".csv"
    
    if make_file:
        status_outfile = open(statuslogfilename,'w')
        status_outfile.close()
    return
    
def initiate_runpath(mydatapath, autoinitialize_statusfilename=True):
    global globaldatapath
    globaldatapath = mydatapath
    
    if autoinitialize_statusfilename:
        initiate_statusfilename()
        
    return
    
def delay_with_alertfile(alertfile):
    #delays until alertfile is removed from directory, at which point the alertfile is replaced and process continues
    while True:
        try:
            alertfile_handle = open(globaldatapath + alertfile,'rU')
            alertfile_handle.close()
            time.sleep(1)
        except:
            with open(globaldatapath + alertfile,'w') as alertfile_handle:
                alertfile_handle.write('1')
            break
    return
    
def remove_alertfile(alertfile):
    os.remove(globaldatapath + alertfile)
    return