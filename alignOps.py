import sysOps
import fileOps
import numpy
import random
import os
import itertools
import time
from Bio import pairwise2
from Bio import SeqIO
from Bio import Seq

def count_mismatches(str1, str2):
    minlen = min(len(str1),len(str2))
    tot_mismatches = 0
    for i in range(minlen):
        tot_mismatches += int(str1[i] != str2[i])
    return tot_mismatches, minlen

def has_ambiguity(c1,c2):
    #assumes c2 is not ambiguous nucleotide (if it is, function will return False)
    #returns True if c2 is subset of c1, False otherwise
    c1 = c1.upper()
    c2 = c2.upper()
    if (c2=='T' and c1 in ['T','N','W','K','Y','B','D','H']) or (c2=='A' and c1 in ['A','N','W','M','R','D','H','V']) or (c2=='C' and c1 in ['C','N','S','M','Y','B','H','V']) or (c2=='G' and c1 in ['G','N','S','K','R','B','D','V']):
        return True
    return False

def get_next_uxi_file_entry(handle):
    
    header = handle.readline()
    if len(header) == 0:
        return [[],[]]
    
    header = header.strip('\n').split('_')
    if len(header) != 3:
        sysOps.throw_exception('Error in get_next_uxi_file_entry(): new line = ' + '_'.join(header))
    id_list = list()
    
    for i in range(int(header[2])):
        id_list.append(handle.readline().strip('\n'))
    
    return [header, id_list]

def consolidate_uxi(uxi_file, start_index = 0, prefix = '', include_inv_amp = False):
    # Function generates file ("identical_" + uxi_file) with list of identical unique uxi's (perfectly matched) with indices and and the number of corresponding reads
    
    #aux_info_file, if provided, contains line-by-line auxiliary assignments (stagger + amplicon-identity, if either exist)
    uxi_lib = dict()
    #build dictionary directly in memory
    
    with open(sysOps.globaldatapath +uxi_file,'rU') as fasta_handle:
        sysOps.throw_status('Proceeding with consolidation ...')
        for my_record in SeqIO.parse(fasta_handle, "fasta"):
            my_seq = str(my_record.seq)
            if my_seq in uxi_lib:
                uxi_lib[my_seq].append(str(my_record.id))
            else:
                uxi_lib[my_seq] = [my_record.id]
    uxi_len = len(my_seq) #final sequence used
    
    if include_inv_amp:
        with open(sysOps.globaldatapath + uxi_file[:uxi_file.find('.')] 
                  + '_amp_inv' + uxi_file[uxi_file.find('.'):],'rU') as fasta_handle:
            sysOps.throw_status('Proceeding with consolidation, including invalid amplicons ...')
            for my_record in SeqIO.parse(fasta_handle, "fasta"):
                my_seq = str(my_record.seq)
                if my_seq in uxi_lib:
                    uxi_lib[my_seq].append(str(my_record.id))
                else:
                    uxi_lib[my_seq] = [my_record.id]
    
    uxi_list_handle = open(sysOps.globaldatapath + prefix + "identical_" + uxi_file, 'w')
    uxi_index = int(start_index)
    for my_uxi_key, my_uxi_record_ids in sorted(uxi_lib.items()): #alphabetize by uxi sequence
        uxi_list_handle.write(str(my_uxi_key) + '_' + str(uxi_index) + '_' + str(len(my_uxi_record_ids)) + '\n') #output line includes uxi index and number of reads
        for my_record_id in my_uxi_record_ids:
            uxi_list_handle.write(my_record_id + '\n')
        uxi_index += 1
        
    uxi_list_handle.close()
    
    del uxi_lib
    
    return [uxi_index,uxi_len] #returns total number of unique entries and length of uxi itself
    
def define_nuc_degeneracy(c1):
    c1 = c1.upper()
    if(c1 in 'ACGTU'):
        return [c1]
    elif(c1 == 'N'):
        return ['A','C','G','T']
    elif(c1 == 'W'):
        return ['A','T']
    elif(c1 == 'S'):
        return ['C','G']
    elif(c1 == 'M'):
        return ['A','C']
    elif(c1 == 'K'):
        return ['G','T']
    elif(c1 =='R'):
        return ['A','G']
    elif(c1 == 'Y'):
        return ['C','T']
    elif(c1 == 'B'):
        return ['C','G','T']
    elif(c1 == 'D'):
        return ['A','G','T']
    elif(c1 == 'H'):
        return ['A','C','T']
    elif(c1 == 'V'):
        return ['A','C','G']
    else:
        sysOps.throw_exception(['Error: ' + c1 + 'does not code for a single- or degenerate-nucleotide'])
        sysOps.exitProgram()
        
def degen_nuc_comp(c1, c2):
    #Returns true if char c1's degenerate set is completely subsumed by c2's degenerate set
    my_bool = True
    c1_set = define_nuc_degeneracy(c1)
    c2_set = define_nuc_degeneracy(c2)
    
    for ci1 in c1_set:
        my_bool = (my_bool and (ci1 in c2_set))
    
    return my_bool

def load_linkage_file_to_list(linkage_file):
    linkage_list = []
    with open(sysOps.globaldatapath +linkage_file,'rU') as linkage_handle:
        for uxi_line in linkage_handle:
            [line_part1, line_part2] = uxi_line.strip('\n').split(":")
            line_part1 = line_part1.split(',')
            linkage_list.append([line_part1[0],int(line_part1[1]),int(line_part1[2]),[int(s) for s in line_part2.split(',')]])
            #elements of linkage_list are:
            #uxi-sequence, self-read-number, RND: list of linked-to indices with self-index as first in line
            
    return linkage_list

def compare(clustfile1,clustfile2,comparison_file_name,rev_comp,read_thresh = 2,filter_substr_list=[],filter_val = 0.75):
    #rev_comp = True/False depending on need of reverse-complement being taken
    #filter_val = maximum fraction of bases in uxi allowed to be the same
    
    #all filtering of legitimate comparison occurs here, at the front end
    print "Beginning comparison between " + clustfile1 + " and " + clustfile2
    
    #Stage 1 of comparison: determine total read-abundance of clusters in clustfile1 and clustfile2, 
    #assign to abund_dict1 and abund_dict2
    
    abund_dict1 = dict()
    with open(sysOps.globaldatapath +clustfile1,'rU') as clust1_handle:
        for clust_line in clust1_handle:
            my_el = clust_line.strip('\n').split('_')
            if(len(my_el)==3):
                uxi_index = my_el[0]
                my_numreads = int(my_el[2])
                if uxi_index not in abund_dict1:
                    abund_dict1[uxi_index] = {'reads': my_numreads, 'is_shared': False}
                else:
                    abund_dict1[uxi_index]['reads'] += my_numreads
                
    abund_dict2 = dict()
    with open(sysOps.globaldatapath +clustfile2,'rU') as clust2_handle:
        for clust_line in clust2_handle:
            my_el = clust_line.strip('\n').split('_')
            if(len(my_el)==3):
                uxi_index = my_el[0]
                my_numreads = int(my_el[2])
                if uxi_index not in abund_dict2:
                    abund_dict2[uxi_index] = {'reads': my_numreads, 'is_shared': False}
                else:
                    abund_dict2[uxi_index]['reads'] += my_numreads
    
    #Stage 2 of comparison: enter actual uxi sequences into dict_clust1 and dict_clust2, 
    #enter their respective cluster-indices into dict_uxi_indices1 and dict_uxi_indices2
    
    dict_clust1 = dict()
    with open(sysOps.globaldatapath +clustfile1,'rU') as clust1_handle:
        for clust_line in clust1_handle:
            my_el = clust_line.strip('\n').split('_')
            if(len(my_el)==3):
                uxi_index = int(my_el[0])
                this_uxi = str(my_el[1])
                my_numreads = int(my_el[2])
                has_disallowed_substr = [my_substr in this_uxi for my_substr in filter_substr_list]
                if abund_dict1[my_el[0]]['reads']>=read_thresh and (True not in has_disallowed_substr) and max(numpy.bincount([('ACGT').index(s) for s in this_uxi]))<=filter_val*len(this_uxi):
                    dict_clust1[this_uxi] = [uxi_index, my_numreads, False] #final entry corresponds to being shared
    
    print "Completed first cluster-file input. Second cluster-file being read, output to cross_comparisons//" + comparison_file_name
    
    comparison_handle = open(sysOps.globaldatapath +'cross_comparisons//' + comparison_file_name,'w')
    
    dict_clust2 = dict()
    with open(sysOps.globaldatapath +clustfile2,'rU') as clust2_handle:
        for clust_line in clust2_handle:
            my_el = clust_line.strip('\n').split('_')
            if(len(my_el)==3):
                uxi_index = int(my_el[0]) #references clustfile2
                #my_uxi references clustfile2 uxi sequences
                #this_uxi references clustfile1 uxi sequences
                my_uxi = str(my_el[1])
                my_numreads = int(my_el[2])
                this_uxi = str(my_uxi)
                if(rev_comp):
                    this_uxi = str(Seq.Seq(this_uxi).reverse_complement())
                has_disallowed_substr = [my_substr in this_uxi for my_substr in filter_substr_list]
                
                if abund_dict2[my_el[0]]['reads']>=read_thresh and (True not in has_disallowed_substr) and max(numpy.bincount([('ACGT').index(s) for s in this_uxi]))<=filter_val*len(this_uxi):
                    dict_clust2[my_uxi] = [uxi_index, my_numreads, False]
                    if this_uxi in dict_clust1:
                        dict_clust1[this_uxi][2] = True
                        dict_clust2[my_uxi][2] = True
                        if str(dict_clust1[this_uxi][0]) not in abund_dict1:
                            sysOps.throw_exception('A: ' + str(dict_clust1[this_uxi][0]) + ' not in dict_uxi_indices1')
                            sysOps.exitProgram()
                        if str(uxi_index) not in abund_dict2:
                            sysOps.throw_exception('B: ' + str(uxi_index) + ' not in dict_uxi_indices2')
                            sysOps.exitProgram()
                            
                        abund_dict1[str(dict_clust1[this_uxi][0])]['is_shared'] = True
                        abund_dict2[str(uxi_index)]['is_shared'] = True
                        
                        comparison_handle.write(str(this_uxi) + "," + str(dict_clust1[this_uxi][0]) + "," + str(dict_clust1[this_uxi][1])  + "," + str(abund_dict1[str(dict_clust1[this_uxi][0])]['reads']) + "," + str(dict_clust2[my_uxi][0]) + "," + str(dict_clust2[my_uxi][1]) + "," + str(abund_dict2[str(dict_clust2[my_uxi][0])]['reads']) + "\n")
                    
    comparison_handle.close()
    
    #count number unique shared and unique unshared
    num_unique_shared = [0,0]
    num_unique_unshared = [0,0]
    read_abundance_shared = [0,0]
    read_abundance_unshared = [0,0]
    
    for uxi_index1 in abund_dict1:
        if abund_dict1[uxi_index1]['is_shared']:
            num_unique_shared[0] += 1
            read_abundance_shared[0] += abund_dict1[uxi_index1]['reads']
        else:
            num_unique_unshared[0] += 1
            read_abundance_unshared[0] += abund_dict1[uxi_index1]['reads']
    
    for uxi_index2 in abund_dict2:
        if abund_dict2[uxi_index2]['is_shared']:
            num_unique_shared[1] += 1
            read_abundance_shared[1] += abund_dict2[uxi_index2]['reads']
        else:
            num_unique_unshared[1] += 1
            read_abundance_unshared[1] += abund_dict2[uxi_index2]['reads']
    
    return [num_unique_shared,num_unique_unshared,read_abundance_shared,read_abundance_unshared]

def compare_identical(idfile1,idfile2,comparison_file_name,rev_comp):
    #rev_comp = True/False depending on need of reverse-complement being taken
    print "Beginning comparison between " + idfile1 + " and " + idfile2
    
    uxi_handle1 = open(sysOps.globaldatapath +idfile1,'rU')
    uxi_dict1 = dict()
    len_uxi1 = -1
    uxi_index = 0    
    for uxi_line1 in uxi_handle1:
        split_str = uxi_line1.strip('\n').split('_')
        if(len(split_str)==3):
            
            my_uxi = split_str[0]
            if len_uxi1<0:
                len_uxi1 = len(my_uxi)
            elif len_uxi1 != len(my_uxi):
                print 'Error: uxi length-mismatch'
                sysOps.exitProgram()
                
            my_numreads = int(split_str[2])
            uxi_dict1[my_uxi] = [uxi_index, my_numreads, False] #final entry corresponds to being shared
            uxi_index += 1 
    
    uxi_handle1.close()
    
    uxi_handle2 = open(sysOps.globaldatapath +idfile2,'rU')
    uxi_dict2 = dict()
    len_uxi2 = -1
    uxi_index = 0    
    comparison_handle = open(sysOps.globaldatapath +comparison_file_name,'w')
    for uxi_line2 in uxi_handle2:
        split_str = uxi_line2.strip('\n').split('_')
        if(len(split_str)==3):
            
            my_uxi = split_str[0]
            if len_uxi2<0:
                len_uxi2 = len(my_uxi)
                if len_uxi1 != len_uxi2:
                    print 'Error: uxi1/uxi2 length-mismatch'
                    sysOps.exitProgram()
                
            my_numreads = int(split_str[2])
            uxi_dict2[my_uxi] = [uxi_index, my_numreads, False]
            this_uxi = str(my_uxi)
            if rev_comp:
                this_uxi = str(Seq.Seq(this_uxi).reverse_complement())
                
            if this_uxi in uxi_dict1:
                print "Found match " + this_uxi
                uxi_dict1[this_uxi][2] = True
                uxi_dict2[my_uxi][2] = True
                comparison_handle.write(this_uxi + "," + str(uxi_dict1[this_uxi][0]) + "," + str(uxi_dict1[this_uxi][1]) + "," + str(uxi_index) + "," + str(my_numreads) + "\n")
            
            uxi_index += 1
            
    comparison_handle.close()
    
    unshared_handle = open(sysOps.globaldatapath +"unshared_" + comparison_file_name,'w')
    for dict_el in uxi_dict1:
        if not uxi_dict1[dict_el][2]:
            unshared_handle.write(dict_el + ",0,"+ str(uxi_dict1[dict_el][0]) + "," + str(uxi_dict1[dict_el][1]) + "\n")

    for dict_el in uxi_dict2:
        if not uxi_dict2[dict_el][2]:
            unshared_handle.write(dict_el + ",1," + str(uxi_dict2[dict_el][0]) + "," + str(uxi_dict2[dict_el][1]) + "\n")
    unshared_handle.close()
    
    return True


def compare_partial_overlap(clustfile,range1,range2):
    #analyze mixing of different parts of analyzed uxi in single, clustered data-set
    #range1 and range2 are 2-element lists
    #output: range-index (0 or 1), number of uxi's having specified sub-uxi, top two read-counts
    clust_handle = open(sysOps.globaldatapath +clustfile,'rU')
    range1_dict = dict()
    range2_dict = dict()
    readcount_dict = dict()
    
    for clust_line in clust_handle:
        my_el = clust_line.strip('\n').split('_')
        if(len(my_el)==3):
            if my_el[0] in readcount_dict:
                readcount_dict[my_el[0]] += int(my_el[2])
            else:
                readcount_dict[my_el[0]] = int(my_el[2])
                
            range1_str = my_el[1][range1[0]:range1[1]]
            range2_str = my_el[1][range2[0]:range2[1]]
            
            if range1_str in range1_dict:
                range1_dict[range1_str].append(int(my_el[0]))
            else:
                range1_dict[range1_str] = [int(my_el[0])]
                
            if range2_str in range2_dict:
                range2_dict[range2_str].append(int(my_el[0]))
            else:
                range2_dict[range2_str] = [int(my_el[0])]
            
    clust_handle.close()
     
    output_handle = open(sysOps.globaldatapath +"compare_partial_" + clustfile,'w')   
    for el in range1_dict:
        unique_els = list(set(range1_dict[el]))
        if(len(unique_els)>1): #more than 1 uxi associated with specified sub-set of uxi
            numreads_list = [readcount_dict[str(x)] for x in unique_els]
            numreads_list = sorted(numreads_list)
            numreads_list.reverse()
            output_handle.write("0," + str(len(unique_els)) + "," + str(numreads_list[0]) + "," + str(numreads_list[1]) + "\n")
    
    for el in range2_dict:
        unique_els = list(set(range2_dict[el]))
        if(len(unique_els)>1): #more than 1 uxi associated with specified sub-set of uxi
            numreads_list = [readcount_dict[str(x)] for x in unique_els]
            numreads_list = sorted(numreads_list)
            numreads_list.reverse()
            output_handle.write("1," + str(len(unique_els)) + "," + str(numreads_list[0]) + "," + str(numreads_list[1]) + "\n")
             
    output_handle.close()         
            
    return True

    
