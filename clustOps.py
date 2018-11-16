import sysOps
import clustOps
import hashAlignments
import alignOps
import os
import itertools
from Bio import SeqIO
from Bio import Seq

def threshold_cluster_uxi_prelinked(uxi_list,identical_uxi_filename,threshold,P=0,subsample = -1, prefix = ''):
    
    # Function will be called while loading linkage_file into uxi_list through load_linkage_file_to_list(linkage_file) in hashAlignments.py
    # Format of linkage file:
    #    uxi-sequence, self-read-number, RND: list of linked-to indices with self-index as first in line
    # linkage_list elements: [uxi-sequence,self-read-number,RND,[list of linked-to indices with self-index as first in line]])
            
    #sort uxi_list by decreasing RND
    num_uxi = len(uxi_list)
    sysOps.throw_status('Starting uxi list sort. List size = ' + str(num_uxi))
    sorted_uxi_list = sorted(uxi_list, key=lambda row: -row[2]) #note: sorted_uxi_list _REMAINS_ a pointer to uxi_list
    index_vals = [-1 for i in range(num_uxi)]
    sysOps.throw_status('Completed uxi list sort. Assigning EASL-clusters ...')
        
    for sorted_uxi_el in sorted_uxi_list: 
        #index_vals, with indices corresponding to _original_ positions in pre-sorted uxi_list, are initiated at -1 (stored in list at row[3])
        #uxi's accepted into cluster with seed of index i, will be given value i in index_vals
        #uxi's rejected from all classification are given index
        if index_vals[sorted_uxi_el[3][0]] < 0: #if this seed has index -1 (has not been assigned to any seed itself)
            index_vals[sorted_uxi_el[3][0]] = int(sorted_uxi_el[3][0]) # set cluster seed to itself
            
        my_index_val = int(index_vals[sorted_uxi_el[3][0]])
        
        for i in range(1,len(sorted_uxi_el[3])):
            if index_vals[sorted_uxi_el[3][i]] < 0: #connected read is unassigned -- assign to current cluster seed
                index_vals[sorted_uxi_el[3][i]] = my_index_val

    sysOps.throw_status('Consolidating clustered uxis ...')
    #consolidate clustered uxi's
    
    if -1 in index_vals:
        sysOps.throw_exception('Error: UNASSIGNED/UNCLUSTERED uxis. Exiting program')
        sysOps.exitProgram()
        
    index_str_vals = [str(int(x)) for x in index_vals]
    new_uxi_dict= dict()
    
    for i in range(num_uxi):
        my_index_str = index_str_vals[i] 
        if my_index_str in new_uxi_dict:
            new_uxi_dict[my_index_str].append(uxi_list[i][0] + "_" + str(uxi_list[i][1]))
        else:
            new_uxi_dict[my_index_str] = [(uxi_list[i][0] + "_" + str(uxi_list[i][1]))]
            
    if(subsample<=0):
        new_uxi_handle = open(sysOps.globaldatapath + prefix + "thresh" + str(threshold) + "_" + identical_uxi_filename,'w')
    else:
        new_uxi_handle = open(sysOps.globaldatapath + prefix + "thresh" + str(threshold) + "_sub" + str(subsample) + identical_uxi_filename,'w')
    
    i = 0
    for dict_el in new_uxi_dict:
        for el in new_uxi_dict[dict_el]:
            new_uxi_handle.write(str(i) + "_" + el + "\n")     
        i += 1   
        
    new_uxi_handle.close()
    
    print "Completed clustering."
    
    return True
