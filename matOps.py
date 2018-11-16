import scipy
import sysOps
import itertools
import scipy
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix
from scipy import *
from importlib import import_module
import numpy as np
import os

global global_csc_op
            
def filter_mats(bcn_dict, trg_dict, bcn_div_dict, trg_div_dict, min_uei_count):
    
    # prune UEI data to exclude UMIs with UEI counts < min_uei_count
    
    if len(bcn_dict) == 0:
        return [bcn_dict, trg_dict, bcn_div_dict, trg_div_dict]
    
    deletion_iteration = 0
    is_list = None
    
    sysOps.throw_status('Filtering matrices with ' + str(len(bcn_div_dict)) + '+' + str(len(trg_div_dict)) + ' UMIs.')
    
    while True:
        
        bcn_retained = 0
        trg_retained = 0
        bcn_deleted = list()
        trg_deleted = list()
        
        for bcn_el in bcn_div_dict:
            if bcn_div_dict[bcn_el]<min_uei_count:
                bcn_deleted.append(bcn_el)
            else:
                bcn_retained += 1
                
        for trg_el in trg_div_dict:
            if trg_div_dict[trg_el]<min_uei_count:
                trg_deleted.append(trg_el)
            else:
                trg_retained += 1
        
        #check if bcn_dict and trg_dict are still list or already converted to values
        if is_list == None:
            for bcn_el in bcn_dict:
                for trg_el in bcn_dict[bcn_el]:
                    is_list = (type(bcn_dict[bcn_el][trg_el]) is list)
                    break
                break
            
        if len(bcn_deleted)==0 and len(trg_deleted)==0:
            sysOps.throw_status('On deletion-iteration ' + str(deletion_iteration) + ', all retained.')
            break
            
        sysOps.throw_status('On deletion-iteration ' + str(deletion_iteration) + ' deleting ' + str(len(bcn_deleted)) + '+' + str(len(trg_deleted)) + ', retained ' + str(bcn_retained) + '+' + str(trg_retained) + '. is_list=' + str(is_list))
        
        if is_list == None:
            sysOps.throw_exception('Error, could not find any elements: len(bcn_dict) = ' + str(len(bcn_dict)))
            sysOps.exitProgram()
            
        for bcn_el in bcn_deleted:
            for trg_el in bcn_dict[bcn_el]:
                if is_list:
                    trg_div_dict[trg_el] -= len(trg_dict[trg_el][bcn_el])
                else:
                    trg_div_dict[trg_el] -= trg_dict[trg_el][bcn_el]
                del trg_dict[trg_el][bcn_el]
                
            del bcn_dict[bcn_el]
            del bcn_div_dict[bcn_el]
            
        for trg_el in trg_deleted:
            for bcn_el in trg_dict[trg_el]:
                if bcn_el in bcn_div_dict: #if not already deleted above
                    if is_list:
                        bcn_div_dict[bcn_el] -= len(bcn_dict[bcn_el][trg_el])
                    else:
                        bcn_div_dict[bcn_el] -= bcn_dict[bcn_el][trg_el]
                    del bcn_dict[bcn_el][trg_el]
                
            del trg_dict[trg_el]
            del trg_div_dict[trg_el]
                        
        deletion_iteration += 1
    
    #check for consistency
    for bcn_el in bcn_dict:
        for trg_el in bcn_dict[bcn_el]:
            if is_list and len(bcn_dict[bcn_el][trg_el])!=len(trg_dict[trg_el][bcn_el]):
                sysOps.throw_exception('ERROR: bcn_dict and trg_dict contain different elements')
                sysOps.exitProgram()
                
    for trg_el in trg_dict:
        for bcn_el in trg_dict[trg_el]:
            if is_list and len(bcn_dict[bcn_el][trg_el])!=len(trg_dict[trg_el][bcn_el]):
                sysOps.throw_exception('ERROR: bcn_dict and trg_dict contain different elements')
                sysOps.exitProgram()
               
    
    return [bcn_dict, trg_dict, bcn_div_dict, trg_div_dict]

def print_features(my_dict,matrix_outfilename,trg_feature_dict_list):
    #NOTE: in current implementation, my_dict must always be trg_dict, such that el1_list becomes list of target cluster-indices, el2_list becomes list of beacon cluster-indices
    
    with open(sysOps.globaldatapath +'features_' + matrix_outfilename,'w') as features_outfile:
        for trg_el in my_dict:
            features_outfile.write(trg_el + ',' + ','.join([str(x) for x in trg_feature_dict_list[trg_el]]) + '\n')

    return

def get_features_from_dict(trg_dict):
    #feature list, in order:  tot-abundance, max-abundance, neighbors connected to directly by 1 degree, neighbors connected to by 2 degrees
    feature_dict_list = dict()

    for el1 in trg_dict:
        
        my_abundance = sum([sum(trg_dict[el1][el2]) for el2 in trg_dict[el1]])
        max_abundance = max([sum(trg_dict[el1][el2]) for el2 in trg_dict[el1]])
        onedeg_connect = len(trg_dict[el1])

        feature_dict_list[el1] = [my_abundance, max_abundance, onedeg_connect]
    
    return feature_dict_list

def print_trg_features(trg_list,trg_feature_dict_list,outfilename):
    features_outfile = open(sysOps.globaldatapath +'features_' + outfilename,'w')
    for trg in trg_list:
        features_outfile.write(','.join([str(x) for x in trg_feature_dict_list[trg]]) + '\n')
    features_outfile.close()

def print_imagemodule_input(trg_dict, imagemodule_input_filename,key_filename,output_dim):
    
    # At this point, trg_dict indexes everything according to original clusters, which will often be non-consecutive due to earlier filtering step
    # The first task of this function is to generate consecutive indices for both beacons and targets
    # The index keys are written to the dictionaries bcn_el_dict and trg_el_dict

    bcn_el_dict = dict()
    trg_el_dict = dict()
    
    uei_val_list = list()
    read_val_list = list()
    bcn_count = 0
    trg_count = 0
    Nuei = 0
    
    for trg_el in trg_dict:
        trg_el_dict[trg_el] = trg_count
        trg_count += 1
        
        for bcn_el in trg_dict[trg_el]:
            if bcn_el not in bcn_el_dict:
                bcn_el_dict[bcn_el] = bcn_count
                bcn_count += 1
    
    Nbcn = len(bcn_el_dict)
    Ntrg = len(trg_el_dict)
    for trg_el in trg_dict:
        trg_el_dict[trg_el] += Nbcn # make all target indices follow beacon indices
            
    for trg_el in trg_dict:
        for bcn_el in trg_dict[trg_el]:
            uei_val_list.append([bcn_el_dict[bcn_el], trg_el_dict[trg_el], 
                                 len(trg_dict[trg_el][bcn_el]), max(trg_dict[trg_el][bcn_el])])    
            
            for read_val in trg_dict[trg_el][bcn_el]:
                read_val_list.append([bcn_el_dict[bcn_el], trg_el_dict[trg_el], read_val])
                
            Nuei += len(trg_dict[trg_el][bcn_el])
    
    with open(sysOps.globaldatapath + key_filename,'w') as key_file:
        for bcn_el in bcn_el_dict:
            key_file.write('0,' + bcn_el + ',' + str(bcn_el_dict[bcn_el]) + '\n')
        for trg_el in trg_el_dict:
            key_file.write('1,' + trg_el + ',' + str(trg_el_dict[trg_el]) + '\n')
            
    uei_val_list.sort(key = lambda row: (row[0], row[1]))

    with open(sysOps.globaldatapath + imagemodule_input_filename,'w') as imagemodule_input_file:    
        for line in uei_val_list:        
            imagemodule_input_file.write(','.join([str(x) for x in line]) + '\n')
            #bcn-index, trg-index, value
            
    read_val_list.sort(key = lambda row: (row[0], row[1]))
            
    with open(sysOps.globaldatapath + 'read_' + imagemodule_input_filename,'w') as imagemodule_input_file_readvals:    
        for line in read_val_list:        
            imagemodule_input_file_readvals.write(','.join([str(x) for x in line]) + '\n')
            
            
    with open(sysOps.globaldatapath + 'seq_params_' + imagemodule_input_filename,'w') as params_file:
        params_file.write('-Nbcn ' + str(int(Nbcn)) + '\n') #total number of (analyzeable) bcn UMI's
        params_file.write('-Ntrg ' + str(int(Ntrg)) + '\n') #total number of (analyzeable) trg UMI's
        params_file.write('-Nuei ' + str(int(Nuei)) + '\n')             #total number of UEI's
        params_file.write('-Nassoc ' + str(int(len(uei_val_list))) + '\n')    #total number of unique associations
        params_file.write('-spat_dims ' + str(int(output_dim)) + '\n')       #output dimensionality
        params_file.write('-err_bound ' + str(0.3) + '\n')              #maximum error
        params_file.write('-max_nontriv_eigenvec_to_calculate 5')
        
def get_umi_uei_matrices(consensus_pairing_csv_file, minreadcount):
    # Will return:
    # 1. bcn_dict: a dictionary of dictionaries (ordered bcn-index-elements, then trg-index-elements) with values being list of UEI read-abundances
    # 2. trg_dict: a dictionary of dictionaries (ordered trg-index-elements, then bcn-index-elements) with values being list of UEI read-abundances
    # 3. bcn_abund_dict: a dictionary of ints with values being total UEI read-abundances
    # 4. trg_abund_dict: a dictionary of ints with values being total UEI read-abundances
    # 5. bcn_div_dict: a dictionary of ints with values being total UEI counts
    # 6. trg_div_dict: a dictionary of ints with values being total UEI counts
    
    csv_handle = open(sysOps.globaldatapath + consensus_pairing_csv_file,'rU')
    
    bcn_dict = dict()
    trg_dict = dict()
    bcn_abund_dict = dict()
    trg_abund_dict = dict()
    bcn_div_dict = dict()
    trg_div_dict = dict()
    
    #consensus_pairing_csv_file has elements: 
    #uei index, beacon-umi index, target-umi index, read-count
    for my_line in csv_handle:
        el = my_line.strip('\n').split(',')
        if int(el[3]) >= minreadcount: #requiring minimum uei abundance
                
            if el[1] in bcn_dict:
                bcn_div_dict[el[1]] += 1
                bcn_abund_dict[el[1]] += int(el[3])
                if el[2] in bcn_dict[el[1]]:
                    bcn_dict[el[1]][el[2]].append(int(el[3]))
                else:
                    bcn_dict[el[1]][el[2]] = [int(el[3])]
            else:
                bcn_dict[el[1]] = dict()
                bcn_dict[el[1]][el[2]] = [int(el[3])]
                bcn_div_dict[el[1]] = 1
                bcn_abund_dict[el[1]] = int(el[3])
            
            if el[2] in trg_dict:
                trg_div_dict[el[2]] += 1
                trg_abund_dict[el[2]] += int(el[3])
                if el[1] in trg_dict[el[2]]:
                    trg_dict[el[2]][el[1]].append(int(el[3]))
                else:
                    trg_dict[el[2]][el[1]] = [int(el[3])]
            else:
                trg_dict[el[2]] = dict()
                trg_dict[el[2]][el[1]] = [int(el[3])]
                trg_div_dict[el[2]] = 1
                trg_abund_dict[el[2]] = int(el[3])
    
    csv_handle.close()
    return [bcn_dict,trg_dict,bcn_abund_dict,trg_abund_dict,bcn_div_dict,trg_div_dict]

def generate_wmat(consensus_pairing_csv_file, minreadcount, min_uei_count, outfilename = 'wmat.csv'):
    #consensus_pairing_csv_file has elements: 
    #uei index, beacon-umi index, target-umi index, read-count
    #if outfilename == None, does not print data to new files
    
    [bcn_dict,trg_dict,
     bcn_abund_dict,trg_abund_dict,
     bcn_div_dict,trg_div_dict] = get_umi_uei_matrices(consensus_pairing_csv_file, minreadcount)       
    if len(trg_dict)==0 or len(bcn_dict)==0:
        sysOps.throw_exception(consensus_pairing_csv_file + ' generated an empty UEI matrix.')
        sysOps.exitProgram()
    
    sysOps.throw_status(['Generating feature list.',sysOps.statuslogfilename])
    trg_feature_dict_list = get_features_from_dict(trg_dict) #collects salient pieces of information on targets for printing in file later
    [bcn_dict, trg_dict, bcn_div_dict, trg_div_dict] = filter_mats(bcn_dict, trg_dict,
                                                                   bcn_div_dict, trg_div_dict, min_uei_count)

    sysOps.throw_status(['Replacing matrix elements with UEI numbers (scalars).',sysOps.statuslogfilename])
    del bcn_dict
    sysOps.throw_status(['Generating weight matrix.',sysOps.statuslogfilename])
    
    if len(trg_dict)==0:
        sysOps.throw_exception('After filtering, ' + consensus_pairing_csv_file + ' generated an empty UEI matrix.')
        sysOps.exitProgram()
    
    if outfilename != None:
        print_features(trg_dict, 'trg_' + outfilename, trg_feature_dict_list)
    
    return trg_dict

    
def get_transpose_dict(my_dict):
    #transpose dictionary of dictionaries
    transpose_dict = dict()
    for el1 in my_dict:
        for el2 in my_dict[el1]:
            if el2 not in transpose_dict:
                transpose_dict[el2] = dict()
            
            transpose_dict[el2][el1] = my_dict[el1][el2]
    
    return transpose_dict
    
