import sysOps
import os
import itertools
import time
import subprocess
import random
import os
import numpy
import libOps
import alignOps
from Bio import SeqIO
from Bio import Seq

def makefile():
    generate_test_data('NNNNNNNWWSNNNWWWNNNSWWNNNNNNN', 'NNNNNNNWWSNNNWWWNNNSWWNNNNNNN', 'NNNNNNNNNNNNNNNNNNNN', 100, 100, 100, 1000, [102, 57], libsettings_filename = 'libsettings.txt', rev_amp_window = 25, error_rate_per_bp = 0.01)

def rand_from_ambig_base(nt):
    nt_isupper = nt.isupper()
    
    if nt.upper() == 'N':
        chars = 'ACGT'
        return_char = chars[int(numpy.floor(float(numpy.random.random()*4)))]
    elif nt.upper() == 'W':
        chars = 'AT'
        return_char = chars[int(numpy.floor(float(numpy.random.random()*2)))]
    elif nt.upper() == 'S':
        chars = 'CG'
        return_char = chars[int(numpy.floor(float(numpy.random.random()*2)))]
    else:
        print 'Error with base ' + nt
        sysOps.exitProgram()
    
    if not nt_isupper:
        return return_char.lower()
    
    return return_char    

def generate_test_data(bumi_form, tumi_form, uei_form, num_true_bumi,num_true_tumi,num_true_uei, numreads, readlens, libsettings_filename = 'libsettings.txt', rev_amp_window = 25, error_rate_per_bp = 0.01):
    #Header format:
    #R(read-number)-C(class-number)-A(amplicon-type)-(mean qual score)-(adapter-sequence mismatches)-(amplicon-sequence-mismatches):(true-sequence group-number)-(read-index within true-sequence group)-(total class-0 reads in true-sequence group)
    
    my_lib = libOps.libObj(settingsfilename = 'libsettings.txt', output_prefix = '_', do_partition_fastq=False) #read libsettings.txt without constructing library
    
    #Classes:
    #0: all ok
    #1: <30 mean qual score
    #2: >1 mismatches on one or more adapter sequence
    
    for_handle = open('test_for_source.fastq','w')
    rev_handle = open('test_rev_source.fastq','w')
    
    bases = 'ACGT'
    bumi_seq_list = list()
    tumi_seq_list = list()
    uei_seq_list = list()
    amp_i_list = list()
    amp_seq_list = list()
    
    for i in range(num_true_bumi):
        bumi_seq_list.append(''.join([rand_from_ambig_base(c) for c in bumi_form]))
    
    for i in range(num_true_tumi):
        tumi_seq_list.append(''.join([rand_from_ambig_base(c) for c in tumi_form]))
        i_amplicon = int(numpy.floor(float(numpy.random.random()*len(my_lib.mySettings["-amplicon"]))))
        amp_i_list.append(i_amplicon)
        amp_seq_list.append(''.join(my_lib.mySettings["-amplicon"][i_amplicon].split(',')))
        
    for i in range(num_true_uei):
        uei_seq_list.append(''.join([rand_from_ambig_base(c) for c in uei_form]))
    
    for i_amplicon in range(len(my_lib.mySettings["-amplicon"])):
        my_lib.mySettings["-amplicon"][i_amplicon] = ''.join([c for c in my_lib.mySettings["-amplicon"][i_amplicon] if c!=','])
        my_lib.mySettings["-amplicon"][i_amplicon] = my_lib.mySettings["-amplicon"][i_amplicon][:rev_amp_window]
    
    for i_read in range(numreads):
        
        i_bumi = int(numpy.floor(float(numpy.random.random()*len(bumi_seq_list))))
        i_tumi = int(numpy.floor(float(numpy.random.random()*len(tumi_seq_list))))
        i_uei = int(numpy.floor(float(numpy.random.random()*len(uei_seq_list))))
        
        newread_for = 'N'*readlens[0]
        newread_rev = 'N'*readlens[1]
        newphred_for = list(numpy.random.poisson(34, readlens[0]))
        newphred_rev = list(numpy.random.poisson(34, readlens[1]))
        i_for_seqform = int(numpy.floor(float(numpy.random.random()*len(my_lib.seqform_for_params))))
        i_rev_seqform = int(numpy.floor(float(numpy.random.random()*len(my_lib.seqform_for_params))))
        myclass = 0
        if numpy.mean(newphred_for) < 30 or numpy.mean(newphred_rev) < 30:
            myclass = 1
    
        error_summary = list()
        
        #beacon UMI
        bumi_el = my_lib.seqform_for_params[i_for_seqform]['U'][0]
        on_umi_pos = 0
        uppercase_err = 0
        lowercase_err = 0
        uxi_err = 0
        
        for j in range(len(bumi_el[1])):
            if bumi_el[1][j].upper() not in 'ACGT':
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newbase = bases[int(numpy.floor(numpy.random.random()*4))]
                    newread_for = newread_for[:(bumi_el[0][0]+j)] + newbase + newread_for[(bumi_el[0][0]+j + 1):]
                    if not alignOps.has_ambiguity(bumi_el[1][j], newread_for[bumi_el[0][0]+j]):
                        if bumi_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                    elif newbase != bumi_seq_list[i_bumi][on_umi_pos]:
                        uxi_err += 1
                else:
                    newread_for = newread_for[:(bumi_el[0][0]+j)] + bumi_seq_list[i_bumi][on_umi_pos] + newread_for[(bumi_el[0][0]+j + 1):]
                
                on_umi_pos += 1
            else:
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newread_for = newread_for[:(bumi_el[0][0]+j)] + bases[int(numpy.floor(numpy.random.random()*4))] + newread_for[(bumi_el[0][0]+j + 1):]
                    if not alignOps.has_ambiguity(bumi_el[1][j], newread_for[bumi_el[0][0]+j]):
                        if bumi_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                else:
                    newread_for = newread_for[:(bumi_el[0][0]+j)] + bumi_el[1][j] + newread_for[(bumi_el[0][0]+j + 1):]

        if uppercase_err>0 or lowercase_err>1:
            myclass = 2
        error_summary.append('U' + str(uppercase_err) + '.' + str(lowercase_err) + '.' + str(uxi_err))
        
        #UEI
        uei_el = my_lib.seqform_for_params[i_for_seqform]['U'][1]
        on_uei_pos = 0
        uppercase_err = 0
        lowercase_err = 0
        uxi_err = 0
        for j in range(len(uei_el[1])):
            if uei_el[1][j].upper() not in 'ACGT':
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newbase = bases[int(numpy.floor(numpy.random.random()*4))] 
                    newread_for = newread_for[:(uei_el[0][0]+j)] + newbase + newread_for[(uei_el[0][0]+j + 1):]
                    if not alignOps.has_ambiguity(uei_el[1][j], newread_for[uei_el[0][0]+j]):
                        if uei_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                    elif newbase != uei_seq_list[i_uei][on_uei_pos]:
                        uxi_err += 1
                else:
                    newread_for = newread_for[:(uei_el[0][0]+j)] + uei_seq_list[i_uei][on_uei_pos] + newread_for[(uei_el[0][0]+j + 1):]
                
                on_uei_pos += 1
            else:
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newread_for = newread_for[:(uei_el[0][0]+j)] + bases[int(numpy.floor(numpy.random.random()*4))] + newread_for[(uei_el[0][0]+j+1):]
                    if not alignOps.has_ambiguity(uei_el[1][j], newread_for[uei_el[0][0]+j]):
                        if uei_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                else:
                    newread_for = newread_for[:(uei_el[0][0]+j)] + uei_el[1][j] + newread_for[(uei_el[0][0]+j + 1):]
    
        if uppercase_err>0 or lowercase_err>1:
            myclass = 2
        error_summary.append('U' + str(uppercase_err) + '.' + str(lowercase_err) + '.' + str(uxi_err))
            
        #target UMI
        tumi_el = my_lib.seqform_rev_params[i_rev_seqform]['U'][0]
        on_umi_pos = 0
        uppercase_err = 0
        lowercase_err = 0
        uxi_err = 0
        
        for j in range(len(tumi_el[1])):
            if tumi_el[1][j].upper() not in 'ACGT':
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newbase = bases[int(numpy.floor(numpy.random.random()*4))]
                    newread_rev = newread_rev[:(tumi_el[0][0]+j)] + newbase + newread_rev[(tumi_el[0][0]+j+1):]
                    if not alignOps.has_ambiguity(tumi_el[1][j], newread_rev[tumi_el[0][0]+j]):
                        if tumi_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                    elif newbase != tumi_seq_list[i_tumi][on_umi_pos]:
                        uxi_err += 1
                else:
                    newread_rev = newread_rev[:(tumi_el[0][0]+j)] + tumi_seq_list[i_tumi][on_umi_pos] + newread_rev[(tumi_el[0][0]+j+1):]
                
                on_umi_pos += 1
            else:
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newread_rev = newread_rev[:(tumi_el[0][0]+j)] + bases[int(numpy.floor(numpy.random.random()*4))] + newread_rev[(tumi_el[0][0]+j+1):]
                    if not alignOps.has_ambiguity(tumi_el[1][j], newread_rev[tumi_el[0][0]+j]):
                        if tumi_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                else:
                    newread_rev = newread_rev[:(tumi_el[0][0]+j)] + tumi_el[1][j] + newread_rev[(tumi_el[0][0]+j+1):]

        if uppercase_err>0 or lowercase_err>1:
            myclass = 2
        error_summary.append('U' + str(uppercase_err) + '.' + str(lowercase_err) + '.' + str(uxi_err))
        
        for prim_el in my_lib.seqform_for_params[i_for_seqform]['P']:
            uppercase_err = 0
            lowercase_err = 0
            
            for j in range(len(prim_el[1])):
                if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                    newread_for = newread_for[:(prim_el[0][0]+j)] + bases[int(numpy.floor(numpy.random.random()*4))] + newread_for[(prim_el[0][0]+j + 1):]
                    if not alignOps.has_ambiguity(prim_el[1][j], newread_for[prim_el[0][0]+j]):
                        if prim_el[1][j].isupper():
                            uppercase_err += 1
                        else:
                            lowercase_err += 1
                else:
                    newread_for = newread_for[:(prim_el[0][0]+j)] + prim_el[1][j] + newread_for[(prim_el[0][0]+j + 1):]
    
            if uppercase_err>0 or lowercase_err>1:
                myclass = 2    
            error_summary.append('P' + str(uppercase_err) + '.' + str(lowercase_err))
        
        if 'P' in my_lib.seqform_rev_params[i_rev_seqform]:
            for prim_el in my_lib.seqform_rev_params[i_rev_seqform]['P']:
                uppercase_err = 0
                lowercase_err = 0
                
                for j in range(len(prim_el[1])):
                    if numpy.random.random()<((4.0/3.0)*error_rate_per_bp): # introduce error
                        newread_rev = newread_rev[:(prim_el[0][0]+j)] + bases[int(numpy.floor(numpy.random.random()*4))] + newread_rev[(prim_el[0][0]+j+1):]
                        if not alignOps.has_ambiguity(prim_el[1][j], newread_rev[prim_el[0][0]+j]):
                            if prim_el[1][j].isupper():
                                uppercase_err += 1
                            else:
                                lowercase_err += 1
                    else:
                        newread_rev = newread_rev[:(prim_el[0][0]+j)] + prim_el[1][j] + newread_rev[(prim_el[0][0]+j+1):]
        
                if uppercase_err>0 or lowercase_err>1:
                    myclass = 2       
                error_summary.append('P' + str(uppercase_err) + '.' + str(lowercase_err)) 
          
        amp_el = my_lib.seqform_rev_params[i_for_seqform]['A'][0] #add amplicon sequence verbatim
        for j in range(amp_el[0][0],readlens[1]):
            newread_rev = newread_rev[:j] + (amp_seq_list[i_tumi][j-amp_el[0][0]]) + newread_rev[(j+1):]
                    
        header = 'R' + str(i_read) + '-C' + str(myclass) + '-A' + str(amp_i_list[i_tumi]) + '-' + '-'.join(error_summary) + ':' + str(i_bumi) + '-' + str(i_uei) + '-' + str(i_tumi)
        for_record = SeqIO.SeqRecord(id = header, name = '', description = '', seq = newread_for.upper(), letter_annotations = {'phred_quality': newphred_for})
        rev_record = SeqIO.SeqRecord(id = header, name = '', description = '', seq = newread_rev.upper(), letter_annotations = {'phred_quality': newphred_rev})
        for_handle.write(for_record.format('fastq'))
        rev_handle.write(rev_record.format('fastq'))
    
    for_handle.close()
    rev_handle.close()
    
def chart_system_times(num_trials,outfilename):
    
    for i in range(num_trials):
        
        with open('time_chart_script' + str(i) +'.py','w') as pythonscript:
            pythonscript.write('import time\n')
            pythonscript.write('with open("time_chart' + str(i) + '.csv","w") as outfile:\n')
            pythonscript.write('    outfile.write(str(time.time()))')
        subprocess.call('bsub -q hour -o out python2.7 time_chart_script' + str(i) +'.py',shell=True)
        terminate = False
        
        while not terminate:
            try:
                with open('time_chart' + str(i) + '.csv','rU') as infile:
                    for line in infile:
                        with open(outfilename,'a') as outfile:
                            outfile.write(line + ',' + str(time.time()) + '\n')
                        break
                os.remove('time_chart_script' + str(i) +'.py')
                os.remove('time_chart' + str(i) + '.csv')
                terminate = True
                time.sleep(1)
            except:
                time.sleep(1)
    
    return