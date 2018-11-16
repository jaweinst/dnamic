import sysOps
import csv
import numpy
from Bio import SeqIO

def read_settingsfile_to_dictionary(mysettingsfile):
    mydictionary = dict()
    with open(sysOps.globaldatapath +mysettingsfile,'rU') as settingsfile:
        for myline in settingsfile:
            myline = str(myline).strip('\n').split(" ")
            if(len(myline)>1):
                this_entry = " ".join(myline[1:len(myline)])
                if('\r' in this_entry):
                    this_entry = this_entry[:this_entry.find('\r')]  
                    
                if myline[0] in mydictionary:
                    mydictionary[myline[0]].append(this_entry) #append multiple options for amplicon
                else:
                    mydictionary[myline[0]] = [this_entry]

    print mydictionary
    return mydictionary

def load_cluster_file_to_dictionary(uxi_cluster_file):
    #takes in file with line-items 
    #-cluster-index_-sequence_read-number
    #outputs dictionary with entries {-sequence: [-cluster-index, read-number]}
    uxi_handle = open(sysOps.globaldatapath +uxi_cluster_file,'rU')
    clust_dict = dict()
    for uxi_line in uxi_handle:
        split_line = uxi_line.strip("\n").split("_")
        clust_dict[str(split_line[1])] = [int(split_line[0]), int(split_line[2])]
    uxi_handle.close()
    return clust_dict

def load_uxi_dict(uxi_list_file):
    #takes in file listing of form
    #1_0_read-number
    #Read 1 ID
    #Read 2 ID
    #2_1_read-number
    #Read 3 ID
    #...
    #Outputs dictionary of form {1: [Read 1 ID, Read 2 ID], 2: [Read 3 ID]}
    uxi_dict = dict()
    uxi_list_handle = open(sysOps.globaldatapath +uxi_list_file,'rU')
    my_uxi = ""
    for uxi_line in uxi_list_handle:
        split_line = uxi_line.strip('\n').split("_")
        if(len(split_line)==3):
            my_uxi = split_line[0]
            uxi_dict[my_uxi] = []
        else:
            uxi_dict[my_uxi].append(split_line[0].strip('\n')) #append with read ID
            
    uxi_list_handle.close()
    return uxi_dict

def group_uxi_reads(uxi_clust_file, uxi_list_file):
    #takes in clustered -file and identically-matched  file, generates look-up of reads based on identically-matched file
    uxi_dict = load_uxi_dict(uxi_list_file)
    uxi_clust_handle = open(sysOps.globaldatapath +uxi_clust_file,'rU')
    read_id_grouping = []
    uxis_and_readnums = [] #list of -sequences and corresponding read-numbers, indexed as a list of lists with one-to-one correspondence to clusters
    for uxi_clust_line in uxi_clust_handle:
        [clust_index, my_uxi, read_num] = uxi_clust_line.strip('\n').split("_")
        clust_index = int(clust_index)
        if clust_index >= len(read_id_grouping):
            read_id_grouping.append([])
            uxis_and_readnums.append([])
        
        if not (my_uxi in uxi_dict):
            print "Error: could not find  " + my_uxi
            sysOps.throw_exception("Could not find  " + my_uxi)
            
        read_id_grouping[clust_index].extend(uxi_dict[my_uxi]) 
        
    uxi_clust_handle.close()
    
    return read_id_grouping

def print_array2csv(my_array, csvfilename):
    [rows,cols] = my_array.shape
    print "Printing CSV file " + csvfilename + " with dimensions (" + str(rows) + "," + str(cols) + ")"
    with open(sysOps.globaldatapath +csvfilename,'w') as csvfile:
        mywriter = csv.writer(csvfile, delimiter=',')
        mywriter.writerows([[my_array[i,j] for j in range(cols)] for i in range(rows)])
        
def consolidateCSVdata(myfiles,mycols):
    myconsolidated = dict()
    lenfiles = len(myfiles)
    for i in range(lenfiles):
        thisfile = myfiles[i]
        thisfile = thisfile[0:(len(thisfile)-1)]
        print i, "/", lenfiles
        with open(sysOps.globaldatapath +'data//' + thisfile, 'rb') as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',')
            for row in csvreader:
                myconsolidated[row[0]] = [row[i] for i in range(mycols)]
                
    return myconsolidated #contains ace information
                
def get_contig_mapping(acefilename):
    mymapping = dict()
    f = open(sysOps.globaldatapath +acefilename, 'r')
    contigtag = 'CO'
    lencontigtag = len(contigtag)
    for myline in f.readlines():
        myline = myline.strip('\n').split(' ')
        if myline[0]==contigtag: #contig number
            contignum = myline[1]
            mymapping[contignum] = []
        elif len(myline)>1 and myline[0] =='AF':
            mymapping[contignum].append(myline[1:(len(myline))])
    return mymapping

def convert_identicalmatchfile2clusterfile(identical_uxi0_file):
    #generates thresh0 cluster-file from identically matched file
    uxi_dict = load_uxi_dict(identical_uxi0_file)
    outfile = open(sysOps.globaldatapath +"thresh0_" + identical_uxi0_file,'w')
    my_index = 0
    for el in uxi_dict:
        num_reads = len(uxi_dict[el])
        outfile.write(str(my_index) + "_" + el + "_" + str(num_reads) + "\n")
        my_index += 1
        
    outfile.close()

def gather_raw_read_stats(mypath,raw_uxi_file='_for_uxi0.fasta',amp_match_file = '_amp_match.txt'):
    
    discarded = 0
    with open(sysOps.globaldatapath +mypath + '_for_discarded_sequences.fasta') as for_discarded_handle: 
        for for_record in SeqIO.parse(for_discarded_handle, "fasta"):
            discarded+=1
        
    amp_discarded = 0
    with open(sysOps.globaldatapath +mypath + '_for_amp_discarded_sequences.fasta') as for_amp_discarded_handle:
        for for_amp_record in SeqIO.parse(for_amp_discarded_handle, "fasta"):
            amp_discarded+=1
    
    accepted = 0
    with open(sysOps.globaldatapath +mypath + raw_uxi_file) as for_acc_handle:
        for for_acc_record in SeqIO.parse(for_acc_handle, "fasta"):
            accepted+=1
    
    all_amp_matches = [0, 0, 0]
    with open(sysOps.globaldatapath +mypath + amp_match_file,'rU') as amp_file_handle:
        for my_line in amp_file_handle:
            all_amp_matches[int(my_line)] += 1
            
    return [accepted,discarded,amp_discarded,all_amp_matches]
    
def gather_lib_stats(mypath,raw_uxi_files=['_for_uxi0.fasta','_for_uxi1.fasta','_rev_uxi0.fasta'],amp_match_file = '_amp_match.txt'):
    #gather all library stats in particular subdirectory
    
    nt_bins = 'ACGT'
    all_reads = []
    plur_nt_counts = []
    all_identical = []
    for raw_uxi_file in raw_uxi_files:
        uxi_handle = open(sysOps.globaldatapath +mypath + raw_uxi_file,'rU')
        my_nt_counts = [0,0,0,0]
        my_num_reads = 0
        for uxi_record in SeqIO.parse(uxi_handle, "fasta"):   
            my_num_reads += 1
            for i in range(4):
                my_nt_counts[i] += uxi_record.seq.count(nt_bins[i])
                
        uxi_handle.close()
        my_num_identical = 0
        identical_uxi_handle = open(sysOps.globaldatapath +mypath + 'identical_' + raw_uxi_file, 'rU')
        for my_line in identical_uxi_handle:
            if my_line.count('_') == 2:
                my_num_identical += 1
                
        identical_uxi_handle.close()
        
        all_reads.append(my_num_reads)
        plur_nt_counts.append([nt_bins[my_nt_counts.index(max(my_nt_counts))], float(max(my_nt_counts))/sum(my_nt_counts)]) #displays base that has plurality of counts, and the fraction plurality
        all_identical.append(my_num_identical)
    
    all_amp_matches = [0, 0, 0]
    with open(sysOps.globaldatapath +mypath + amp_match_file,'rU') as amp_file_handle:
        for my_line in amp_file_handle:
            all_amp_matches[int(my_line)] += 1
    
    return [all_reads, plur_nt_counts, all_identical, all_amp_matches]
    
    
    
    