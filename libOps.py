from Bio import SeqIO
from Bio import Seq
import fileOps
import sysOps
import parseOps
import os
import shutil
import numpy as np
import sys
import itertools
import random

'''
This code performs sequence analysis on raw DNA microscopy reaction output 
'''

def get_subrecord(seq_np_array, my_element = [[],None,None,None], max_frac_mismatch = 0.0, tolerate_ambig_newseq = True, trunc_len = None):
            
    # seqforms are dicts with elements comprising lists of the following sub-list elements: 
    #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)] -- seq_bool_vec and capital_bool_vec have 4xbase number elements (others only have base number elements)
    # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
    
    if(len(my_element)>1 and type(my_element[0][1])==int and (seq_np_array.shape[0]/4 < my_element[0][1] or my_element[1].shape[0]/4 != my_element[0][1]-my_element[0][0])):
        sysOps.throw_exception("Error matching sequence dimensions to sequence filter")
        sysOps.exitProgram()
    
    if len(my_element[0]) == 0:
        print 'len(my_element[0]) == 0'
        print str(my_element)
        sysOps.exitProgram()
    else:
        start_element = int(my_element[0][0])
        if(len(my_element[0]) == 1 or my_element[0][1] is None):
            if type(trunc_len) == int:
                end_element = min(seq_np_array.shape[0]/4,trunc_len)
            else:
                end_element = seq_np_array.shape[0]/4 # full length

            end_element = min(end_element, start_element + my_element[1].shape[0]/4) # if input length does not cover entire input sequence, don't require it to
        else:
            end_element = int(my_element[0][1])
    
    if start_element>end_element or start_element < 0 or end_element < 0:
        print '[start_element,end_element] = ' + str([start_element,end_element])
        print str(seq_np_array)
        sysOps.exitProgram()
              
    if(type(my_element[1])!=np.ndarray or my_element[1].shape[0]==0 or start_element==end_element): #nothing to compare record to, return true on question of filter-match (will be the case for amplicons)
        return True

    len_record = end_element-start_element
    
    if np.sum(seq_np_array[(4*start_element):(4*end_element)]) != len_record and (not tolerate_ambig_newseq):
        return False
    
    counted_bp_uppercase = np.sum(my_element[2][:4*len_record])/4
    counted_bp_lowercase = np.sum(~my_element[2][:4*len_record])/4

    counted_mismatches_uppercase = counted_bp_uppercase - np.sum(np.multiply(my_element[2][:4*len_record],
                                                                             np.multiply(my_element[1][:4*len_record],
                                                                                         seq_np_array[(4*start_element):(4*end_element)])))
    counted_mismatches_lowercase = counted_bp_lowercase - np.sum(np.multiply(~my_element[2][:4*len_record],
                                                                             np.multiply(my_element[1][:4*len_record],
                                                                                         seq_np_array[(4*start_element):(4*end_element)])))
        
    #allow specified frequency of mismatches on lower-case bases of my_element[1], disallow mismatches on upper-case bases of my_element[1]
    return ((counted_bp_lowercase==0
             or float(counted_mismatches_lowercase)/counted_bp_lowercase <=max_frac_mismatch)
             and 
             (counted_bp_uppercase==0
              or counted_mismatches_uppercase == 0))

class baseTally:
    def __init__(self):
        self.tot_base = np.zeros([]) #rows ordered ACGTN
        self.my_len = None
        self.num_tallied = 0
        
    def get_str_consensus(self):
        consensus = self.get_consensus()
        cons_str = ''
        for el in consensus:
            if len(el)>1: #ignore details of ambiguities -- will all be treated the same
                cons_str = cons_str + 'N'
            else:
                cons_str = cons_str + el
        
        return str(cons_str)
        
    def get_consensus(self):
        bases = 'ACGTN'
        consensus = list()
        for i in range(self.my_len):
            maxcount = max(self.tot_base[:,i])
            
            if 2*maxcount < self.num_tallied: #if majority fail to vote on extending consensus further, truncate accordingly
                break

            maxcount_bases = ''
            for j in range(5):
                if self.tot_base[j,i] == maxcount:
                    maxcount_bases = maxcount_bases + bases[j]
            consensus.append(maxcount_bases)
        
        return consensus

    def add_record(self, seq_str, inp_tally = 1, trunc_len = None):
        
        if not (trunc_len is None):
            seq_str = seq_str[:trunc_len]
    
        my_len = len(seq_str)
            
        if my_len == 0:
            return
            
        if(self.num_tallied == 0 or self.my_len == None):
            self.my_len = my_len
            self.tot_base = np.zeros([5,self.my_len])
            
        if self.my_len < my_len:
            self.tot_base = np.concatenate((self.tot_base,np.zeros([5,my_len-self.my_len])),1)
            self.my_len = int(my_len)
        
        for i in range(min(self.my_len,my_len)):
            self.tot_base['ACGTN'.find(seq_str[i])][i] += inp_tally
            
        self.num_tallied+=inp_tally
        
    def to_str(self):
        a_line =  ",".join([str(int(x)) for x in list(self.tot_base[0])])
        c_line =  ",".join([str(int(x)) for x in list(self.tot_base[1])])
        g_line =  ",".join([str(int(x)) for x in list(self.tot_base[2])])
        t_line =  ",".join([str(int(x)) for x in list(self.tot_base[3])])
        n_line =  ",".join([str(int(x)) for x in list(self.tot_base[4])])
            
        return "num_tallied:" + str(self.num_tallied) + "\nA:" + a_line + "\nC:" + c_line + "\nG:" + g_line + "\nT:" + t_line + "\nN:"  + n_line + "\n"
    
    def read_from_file(self,basetally_filename,key):
        found_key = False
        self.tot_base = [list(),list(),list(),list(),list()]
        with open(sysOps.globaldatapath + basetally_filename,'rU') as infile:
            for line in infile:
                if found_key:
                    if line.startswith("num_tallied:"):
                        self.num_tallied = int(line.strip('\n').split(':')[1])
                    elif line.startswith("A:"):
                        self.tot_base[0] = [int(x) for x in (line.strip('\n').split(':')[1]).split(',')]
                        self.my_len = len(self.tot_base[0])
                    elif line.startswith("C:"):
                        self.tot_base[1] = [int(x) for x in (line.strip('\n').split(':')[1]).split(',')]
                    elif line.startswith("G:"):
                        self.tot_base[2] = [int(x) for x in (line.strip('\n').split(':')[1]).split(',')]
                    elif line.startswith("T:"):
                        self.tot_base[3] = [int(x) for x in (line.strip('\n').split(':')[1]).split(',')]
                    elif line.startswith("N:"):
                        self.tot_base[4] = [int(x) for x in (line.strip('\n').split(':')[1]).split(',')]
                    else:
                        break
                    
                elif line.startswith(key):
                    found_key = True
        
        if not found_key:
            sysOps.throw_exception('Did not find key ' + key + ' in base-stats file.')
            sysOps.exitProgram()
            
        sysOps.throw_status('Read base-statistics from file ' + basetally_filename)
    
    def calc_ncrit(self,max_ncrit = 10000000):
        #calculate probability that 2 random uxi's drawn from current base-distribution would be within 1 bp
        #as well as maximum diversity of uxi's for which there is less than a 50% chance that any 2 are within 1 bp
        
        print str(self.my_len)
        print str(self.tot_base)
        base_freqs = np.zeros([4,self.my_len])
        for j in range(self.my_len):
            sum_j = float(sum([self.tot_base[i][j] for i in range(4)]))
            base_freqs[0,j] = float(self.tot_base[0][j])/sum_j
            base_freqs[1,j] = float(self.tot_base[1][j])/sum_j
            base_freqs[2,j] = float(self.tot_base[2][j])/sum_j
            base_freqs[3,j] = float(self.tot_base[3][j])/sum_j
       
        my_prod = 1.0
        for j in range(self.my_len):
            my_prod *= sum([base_freqs[i,j]*base_freqs[i,j] for i in range(4)])
        
        my_sum = 0.0
        for j in range(self.my_len):
            my_sum += (sum([base_freqs[i,j]*(1.0-base_freqs[i,j]) for i in range(4)]))*(np.prod([sum([base_freqs[k,ell]*base_freqs[k,ell] for k in range(4)]) for ell in range(self.my_len) if ell!=j]))
        
        p1 = my_prod + my_sum
        
        sysOps.throw_status('Calculated p1=' + str(p1))
        
        p0overlap = 1.0
        n = 0
        while(p0overlap > 0.5 and n<=max_ncrit):
            n += 1
            p0overlap *= (1.0 - (n*p1))
        
        if n>max_ncrit:
            sysOps.throw_exception('Reached max_ncrit=' + str(max_ncrit))
        return [p1,n-1]
        
class libObj:
    '''
    libObj, or library object, stores consolidated sequence data, labeled by specific template uxi's
    Member variables:
    uxi_lib
    for_fastqsource
    rev_fastqsource
    '''
    def __init__(self, settingsfilename = "libsettings.txt", output_prefix = "", do_partition_fastq = True, output_subsampling = True):
        '''
        Constructor calls fastq-loader
        Default file-names take global run-path and use run.fastq and libsettings.txt
        libsettings.txt must contain -seqform tag
        
        Typical file:
        -source_for forfile.fastq
        -source_rev revfile.fastq
        -max_mismatch 0.06
        -max_mismatch_amplicon 0.0
        -min_mean_qual 30
        -seqform_for ...
        -seqform_rev ...
        -amplicon ...
        '''
        
        self.output_prefix = output_prefix
        self.lenrequirement_discarded_reads = None
        self.num_discarded = None
        self.num_amplicon_invalid = None
        self.num_retained = None
        
        try:
            self.load_lib_settings(settingsfilename)
        except:
            print "Error opening settings-file " + settingsfilename
            sysOps.throw_exception(["Error opening settings-file " + settingsfilename])
            sysOps.exitProgram()
        
        if("-source_for" in self.mySettings):
            self.for_fastqsource = ','.join(self.mySettings["-source_for"])
        else:
            self.for_fastqsource = "run_for.fastq"
        
        if("-source_rev" in self.mySettings):
            self.rev_fastqsource = ','.join(self.mySettings["-source_rev"])
        else:            
            self.rev_fastqsource = "run_rev.fastq"
            
        if("-max_mismatch" in self.mySettings):
            self.max_mismatch_template = float(self.mySettings["-max_mismatch"][0])
        else:
            self.max_mismatch_template = 0.0
            
        if("-max_mismatch_amplicon" in self.mySettings):
            self.max_mismatch_amplicon = float(self.mySettings["-max_mismatch_amplicon"][0])
        else:
            self.max_mismatch_amplicon = 0.0
            
        if("-min_mean_qual" in self.mySettings):
            self.min_mean_qual = int(self.mySettings["-min_mean_qual"][0])
        else:
            self.min_mean_qual = int(30)
            
        if "-filter_amplicon_window" in self.mySettings:
            self.filter_amplicon_window = int(self.mySettings["-filter_amplicon_window"][0])
        else:
            self.filter_amplicon_window = 25 # default
            
        sysOps.throw_status('-source_for: ' + str(self.for_fastqsource))
        sysOps.throw_status('-source_rev: ' + str(self.rev_fastqsource))
        sysOps.throw_status('-max_mismatch: ' + str(self.max_mismatch_template))
        sysOps.throw_status('-max_mismatch_amplicon: ' + str(self.max_mismatch_amplicon))
        sysOps.throw_status('-min_mean_qual: ' + str(self.min_mean_qual))
        sysOps.throw_status('-filter_amplicon_window: ' + str(self.filter_amplicon_window))
        sysOps.throw_status(["Constructing libObj: for_fastqsource=" + self.for_fastqsource + ", rev_fastqsource=" + self.rev_fastqsource + ", settingsfilename=" + settingsfilename])
            
        if("-amplicon" in self.mySettings):
            sysOps.throw_status('-amplicon: ' + str(self.mySettings["-amplicon"]))
            new_amplicon_lists = list()
            min_amplicon_len = None
            for i in range(len(self.mySettings["-amplicon"])):
                self.mySettings["-amplicon"][i] = self.mySettings["-amplicon"][i].split('|')
                if len(self.mySettings["-amplicon"][i]) > 1:
                    sysOps.throw_status('Removing amplicon template name ' + str(self.mySettings["-amplicon"][i][0]))
                    self.mySettings["-amplicon"][i] = str(self.mySettings["-amplicon"][i][1])
                else:
                    self.mySettings["-amplicon"][i] = str(self.mySettings["-amplicon"][i][0])
                this_amplicon_len = 0
                my_new_amplicon_list = list()
                for sub_amplicon in reversed(self.mySettings["-amplicon"][i].split(',')):
                    my_new_amplicon_list.append(self.ambig_reverse_complement(sub_amplicon))
                    this_amplicon_len += len(sub_amplicon)
                    #amplicons entered in "sense" configuration -- here, the reverse complement is taken, and accounting for the fact that the amplicon may be in ','-separated sequence-blocks
                new_amplicon_lists.append(','.join(my_new_amplicon_list))
                if min_amplicon_len==None or min_amplicon_len>this_amplicon_len:
                    min_amplicon_len = int(this_amplicon_len)
        
            self.mySettings["-amplicon"] = list(new_amplicon_lists) #string-entry converted to list with first element being forward-amplicon and second being reverse-amplicon. Biopython's reverse_complement() allows for ambiguous nucleotides.
            if min_amplicon_len < self.filter_amplicon_window:
                self.filter_amplicon_window = min_amplicon_len
                sysOps.throw_status('-filter_amplicon_window re-set to minimum entered amplicon length: ' + str(self.filter_amplicon_window))
    
        # seqforms are dicts with elements comprising lists of the following sub-list elements: 
        #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)]
        # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
        try:
            self.seqform_rev_params = list()
            if("-amplicon" in self.mySettings):
                for this_seqform in self.mySettings["-seqform_rev"]:
                    seqform_rev_params_sublist = list()
                    for amplicon_option in self.mySettings["-amplicon"]:
                        seqform_rev_params_sublist.append(parseOps.parse_seqform(this_seqform,amplicon_option))
                    self.seqform_rev_params.append(seqform_rev_params_sublist)
            else:
                for this_seqform in self.mySettings["-seqform_rev"]:
                    self.seqform_rev_params.append([parseOps.parse_seqform(this_seqform,None)])
            
            self.seqform_for_params = list()
            for this_seqform in self.mySettings["-seqform_for"]:
                self.seqform_for_params.append([parseOps.parse_seqform(this_seqform,None)])
                
        except:
            print "No seqform parameter in settings-file " + settingsfilename
            sysOps.throw_exception(["No seqform parameter in " + settingsfilename])
            sysOps.exitProgram()
            
        if("-amplicon_terminate" in self.mySettings):
            sysOps.throw_status('-amplicon_terminate: ' + str(self.mySettings["-amplicon_terminate"]))
            len_search_term = None
            my_new_amplicon_term_list = list()
            for i in range(len(self.mySettings["-amplicon_terminate"])):
                for sub_amplicon_term in self.mySettings["-amplicon_terminate"][i].split(','):
                    if not(len_search_term == None or len(sub_amplicon_term)==len_search_term):
                        sysOps.throw_exception('Error: termination sequences have different lengths.')
                        sysOps.exitProgram()
                        
                    len_search_term = len(sub_amplicon_term)
                    my_new_amplicon_term_list.append(self.ambig_reverse_complement(sub_amplicon_term))
                
            term_dict = dict()
            for new_amplicon_term in my_new_amplicon_term_list:
                term_dict[new_amplicon_term] = None
            #assemble dictionary of possible error-sequences, with number of errors <= self.mySettings['max_mismatch_amplicon']*len_search_term
            #for err_num in range(1,int(self.max_mismatch_amplicon*float(len_search_term))):
            for err_num in range(1,2):
                new_dict = dict()
                for el in term_dict:
                    for i in range(len_search_term):
                        for nt in 'ACGT':
                            new_seq = el[:i] + nt + el[(i+1):]
                            if (new_seq not in term_dict) and (new_seq not in new_dict):
                                new_dict[new_seq] = None
                for new_el in new_dict:
                    term_dict[new_el] = None                
            
            self.mySettings["-amplicon_terminate"] = [len_search_term, term_dict]
            
        self.uxi_lib = dict() #initiate empty library for later uxi-lookup
        
    def ambig_reverse_complement(self,seq_str):
        seq_revcomp = ''
        lookup_str =  'ATCGSWNYRMK'
        revcomp_str = 'TAGCWSNRYKM'
        for i in range(len(seq_str)):
            c = str(seq_str[i])
            is_lower = c.islower()
            c = c.upper()
            ind = lookup_str.index(c)
            if ind < 0:
                sysOps.throw_exception('ERROR: could not recognize nucleotide ' + c)
            else:
                c = str(revcomp_str[ind])
            if is_lower:
                c = c.lower()
            seq_revcomp = c + seq_revcomp
        return seq_revcomp
           
    def load_lib_settings(self,mysettingsfilename):
        self.mySettings = fileOps.read_settingsfile_to_dictionary(mysettingsfilename)
        print "Loaded library-settings file " + mysettingsfilename + ":"
        for s in self.mySettings:
            print str(s) + " ... " + str(self.mySettings[s])
    
    def truncate_amplicon(self, mystr):
        mystrlen = len(mystr)
        trunc_len = int(mystrlen)
        if('-amplicon_terminate' in self.mySettings):
            len_search_term = int(self.mySettings['-amplicon_terminate'][0])
            for i in range(mystrlen-len_search_term+1):
                if mystr[i:(i+len_search_term)] in self.mySettings['-amplicon_terminate'][1]: #search dictionary object
                    trunc_len = int(i)
                    break #found earliest occurring string in termination dictionary
            
        return str(mystr[:trunc_len])
        
    def get_min_allowed_readlens(self,filter_amplicon_window):
        
        # seqforms are dicts with elements comprising lists of the following sub-list elements: 
        #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)]
        # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
        
        min_allowed_forlen = 0
        min_allowed_revlen = 0
        
        print "Determining minimum allowable read lengths ..."
        
        for outer_list_el in self.seqform_for_params:
            for inner_list_el in outer_list_el:
                for dict_el in inner_list_el:
                    for my_sub_el in inner_list_el[dict_el]:
                        if (min_allowed_forlen < my_sub_el[0][0] or (len(my_sub_el[0])>1 and min_allowed_forlen < my_sub_el[0][1])):
                            min_allowed_forlen = int(my_sub_el[0][1])
                            
        sysOps.throw_status('Minimum allowable FORWARD-read length found to be ' + str(min_allowed_forlen))
        
        for outer_list_el in self.seqform_rev_params:
            for inner_list_el in outer_list_el:
                for dict_el in inner_list_el:
                    if dict_el == 'A':
                        filter_amplicon_window_remaining = int(filter_amplicon_window)
                        for my_sub_el in inner_list_el[dict_el]:
                            if filter_amplicon_window >= 0 and min_allowed_revlen < my_sub_el[0][0]+filter_amplicon_window_remaining:
                                min_allowed_revlen = int(my_sub_el[0][0]+filter_amplicon_window_remaining)
                            elif filter_amplicon_window < 0 and len(my_sub_el[0])>1 and filter_amplicon_window_remaining < my_sub_el[0][1]:
                                min_allowed_revlen = int(my_sub_el[0][1])
                            if len(my_sub_el[0])>1 and type(my_sub_el[0][1])==int:
                                filter_amplicon_window_remaining -= (my_sub_el[0][1] - my_sub_el[0][0])
                    else:
                        for my_sub_el in inner_list_el[dict_el]:
                            if (min_allowed_revlen < my_sub_el[0][0] or (len(my_sub_el[0])>1 and min_allowed_revlen < my_sub_el[0][1])):
                                min_allowed_revlen = int(my_sub_el[0][1])
                            
        sysOps.throw_status('Minimum allowable REVERSE-read length found to be ' + str(min_allowed_revlen))
                                        
        return [min_allowed_forlen,min_allowed_revlen]
    
    def output_lib_stats(self, outfilename, base_stats_file):
        with open(sysOps.globaldatapath + outfilename,'w') as outfile:
            outfile.write("num_retained:" + str(self.num_retained) + "\n")
            outfile.write("num_discarded:" + str(self.num_discarded) + "\n")
            outfile.write("num_amplicon_invalid:" + str(self.num_amplicon_invalid) + "\n")
            outfile.write("lenrequirement_discarded_reads:" + str(self.lenrequirement_discarded_reads) + "\n")
            for_uxi0_baseTally = baseTally()
            for_uxi1_baseTally = baseTally()
            rev_uxi0_baseTally = baseTally()
            for_uxi0_baseTally.read_from_file(base_stats_file,'for_uxi0')
            for_uxi1_baseTally.read_from_file(base_stats_file,'for_uxi1')
            rev_uxi0_baseTally.read_from_file(base_stats_file,'rev_uxi0')
            outfile.write("for_uxi0_capacity:" + ','.join([str(x) for x in for_uxi0_baseTally.calc_ncrit()]) + "\n")
            outfile.write("for_uxi1_capacity:" + ','.join([str(x) for x in for_uxi1_baseTally.calc_ncrit()]) + "\n")
            outfile.write("rev_uxi0_capacity:" + ','.join([str(x) for x in rev_uxi0_baseTally.calc_ncrit()]) + "\n")
        
    def partition_fastq_library(self, discarded_sequence_path = "discarded_sequences.fasta", mean_phred_score_path = "mean_phred_scores.txt"):
        '''
        generate 4 file-types from 2 fastq files (forward and reverse)
        one file per uxi per end, e.g. for_uxi0.fastq, for_uxi1.fastq, rev_uxi0.fastq -- containing only ambiguous base-positions
        one file per amplicon per end, e.g. for_amp0.fastq, rev_amp0.fastq
        self.seqform_for_params and self.seqform_rev_params are already stored in current object's memory
        Form of these variables is a list of the following:
            Element 1: [start_pos,end_pos]
            Element 2: np.ndarray(seq_bool_vec, dtype=np.bool_)
            Element 3: np.ndarray(capital_bool_vec, dtype=np.bool_)
            Element 4: np.ndarray(ambig_vec, dtype=np.bool_)
        '''
        
        num_seqform_for = len(self.seqform_for_params) 
        num_seqform_rev = len(self.seqform_rev_params) 
              
        sysOps.throw_status("Loading forward fastq file/s " + self.for_fastqsource + ", printing discarded sequences to for_" + discarded_sequence_path)
        
        for_source_filenames = list()

        if ',' in self.for_fastqsource: #more than 1 fastq source file to read from
            for this_fastqsource in self.for_fastqsource.split(','):
                if this_fastqsource.startswith('.'):
                    for_source_filenames.append(sysOps.globaldatapath +this_fastqsource)
                else:
                    for_source_filenames.append(str(this_fastqsource))
        else:
            if self.for_fastqsource.startswith('.'):
                for_source_filenames.append(sysOps.globaldatapath +self.for_fastqsource)
            else:
                for_source_filenames.append(str(self.for_fastqsource))
            
        sysOps.throw_status("Loading reverse fastq file/s " + self.rev_fastqsource + ", printing discarded sequences to rev_" + discarded_sequence_path)
        
        rev_source_filenames = list()

        if ',' in self.rev_fastqsource: #more than 1 fastq source file to read from
            for this_fastqsource in self.rev_fastqsource.split(','):
                if this_fastqsource.startswith('.'):
                    rev_source_filenames.append(sysOps.globaldatapath +this_fastqsource)
                else:
                    rev_source_filenames.append(str(this_fastqsource))
        else:
            if self.rev_fastqsource.startswith('.'):
                rev_source_filenames.append(sysOps.globaldatapath +self.rev_fastqsource)
            else:
                rev_source_filenames.append(str(self.rev_fastqsource))
                    
        for_uxi_filehandles = list()
        rev_uxi_filehandles = list()
        for_uxi_amp_invalid_filehandles = list()
        rev_uxi_amp_invalid_filehandles = list()
        for_uxi_baseTally_list = list()
        rev_uxi_baseTally_list = list()
        
        if 'U' in self.seqform_for_params[0][0]: 
            for i in range(len(self.seqform_for_params[0][0]['U'])): # assumes all sequence forms have the same set of UMI's
                for_uxi_filehandles.append(open(sysOps.globaldatapath +self.output_prefix + 'for_uxi' + str(i) + '.fasta','w'))
                for_uxi_amp_invalid_filehandles.append(open(sysOps.globaldatapath +self.output_prefix + 'for_uxi' + str(i) + '_amp_inv.fasta','w'))
                for_uxi_baseTally_list.append(baseTally())
        else:
            sysOps.throw_exception('Error: no UXI found in forward read.')
            sysOps.exitProgram()
            
        if 'U' in self.seqform_rev_params[0][0]: 
            for i in range(len(self.seqform_rev_params[0][0]['U'])):
                rev_uxi_filehandles.append(open(sysOps.globaldatapath +self.output_prefix + 'rev_uxi' + str(i) + '.fasta','w'))
                rev_uxi_amp_invalid_filehandles.append(open(sysOps.globaldatapath +self.output_prefix + 'rev_uxi' + str(i) + '_amp_inv.fasta','w'))
                rev_uxi_baseTally_list.append(baseTally())
        else:
            sysOps.throw_exception('Error: no UXI found in reverse read.')
            sysOps.exitProgram()
        
        if 'A' in self.seqform_rev_params[0][0]:
            rev_amp_filehandle = open(sysOps.globaldatapath +self.output_prefix + 'rev_amp0.fasta','w')
            rev_amp_invalid_fastq = open(sysOps.globaldatapath + self.output_prefix + 'rev_amp_inv.fasta','w')
            rev_amp_baseTally = baseTally()
        else:
            sysOps.throw_exception('Error: no amplicon found in reverse read.')
            sysOps.exitProgram()
        
        # fill in "primer" fields with empty lists in case empty
        if 'P' not in self.seqform_for_params[0][0]: 
            for i in range(len(self.seqform_for_params)):
                for j in range(len(self.seqform_for_params[i])):
                    self.seqform_for_params[i][j]['P'] = list()
            
        if 'P' not in self.seqform_rev_params[0][0]: 
            for i in range(len(self.seqform_rev_params)):
                for j in range(len(self.seqform_rev_params[i])):
                    self.seqform_rev_params[i][j]['P'] = list()
            
        amp_match_handle = open(sysOps.globaldatapath +self.output_prefix + 'amp_match.txt','w') 
        for_auxassign_handle = open(sysOps.globaldatapath +'auxassign_' + self.output_prefix + 'for_uxi0.txt','w') 
        rev_auxassign_handle = open(sysOps.globaldatapath + 'auxassign_' + self.output_prefix + 'rev_uxi0.txt','w')
        
        for_discarded_fasta = open(sysOps.globaldatapath +self.output_prefix + 'for_' + discarded_sequence_path,'w')
        rev_discarded_fasta = open(sysOps.globaldatapath +self.output_prefix + 'rev_' + discarded_sequence_path,'w')
        phred_file = open(sysOps.globaldatapath +self.output_prefix + mean_phred_score_path, 'w')
                
        self.lenrequirement_discarded_reads = 0
        self.num_discarded = 0
        self.num_amplicon_invalid = 0
        self.num_retained = 0
        num_discarded_qscore = 0
        
        [for_read_len,rev_read_len] = self.get_min_allowed_readlens(self.filter_amplicon_window)
        for_uxi_binary_list = list()
        rev_uxi_binary_list = list()
        rev_amp_binary_list = list()
        
        for seqform_index in range(num_seqform_for):
            sub_list = list()
            for amplicon_index in range(len(self.seqform_for_params[seqform_index])):
                sub_sub_list = list()
                for my_element in self.seqform_for_params[seqform_index][amplicon_index]['U']:
                    new_binary_array = np.zeros(for_read_len,dtype=np.bool_)
                    new_binary_array[my_element[0][0]:my_element[0][1]][my_element[3]] = np.True_
                    sub_sub_list.append(new_binary_array)
                sub_list.append(sub_sub_list)
            for_uxi_binary_list.append(sub_list)
            
        for seqform_index in range(num_seqform_rev):
            uxi_sub_list = list()
            amp_sub_list = list()
            for amplicon_index in range(len(self.seqform_rev_params[seqform_index])):
                sub_sub_list = list()
                for my_element in self.seqform_rev_params[seqform_index][amplicon_index]['U']:
                    new_binary_array = np.zeros(rev_read_len,dtype=np.bool_)
                    new_binary_array[my_element[0][0]:my_element[0][1]][my_element[3]] = np.True_
                    sub_sub_list.append(new_binary_array) # make list of distinct UXI instantiations in this sequence-form
                uxi_sub_list.append(sub_sub_list)
                sub_sub_list = list()
                for my_element in self.seqform_rev_params[seqform_index][amplicon_index]['A']:
                    new_binary_array = np.zeros(rev_read_len,dtype=np.bool_)
                    if type(my_element[0][1]) != int:
                        end_element = rev_read_len
                    else:
                        end_element = my_element[0][1]
                    new_binary_array[my_element[0][0]:end_element] = np.True_
                    sub_sub_list.append(new_binary_array) # make list of distinct sub-amplicon instantiations in this sequence-form
                amp_sub_list.append(sub_sub_list)
            rev_uxi_binary_list.append(uxi_sub_list)
            rev_amp_binary_list.append(amp_sub_list)
             
        # seqforms are dicts with elements comprising lists of the following sub-list elements: 
        #    [boundaries, seq_bool_vec, capital_bool_vec (is a capital base), ambig_vec (is an ambiguous base)]
        # seqform_*_params are lists of lists of seqforms (outer indices are frame-sequence index, inner indices are amplicon index)
        for_seq_buff = np.zeros(4*for_read_len,dtype=np.bool_)
        rev_seq_buff = np.zeros(4*rev_read_len,dtype=np.bool_)
        for_list_buff = ['' for i in range(for_read_len)]
        rev_list_buff = ['' for i in range(rev_read_len)]
        for_qual_buff = np.zeros(for_read_len,dtype=np.int32)
        rev_qual_buff = np.zeros(rev_read_len,dtype=np.int32)
        for_record = None
        rev_record = None
        
        for for_source_filename,rev_source_filename in zip(for_source_filenames,rev_source_filenames): #loop through list of file handles
            for_handle = SeqIO.parse(open(for_source_filename),"fastq")
            rev_handle = SeqIO.parse(open(rev_source_filename),"fastq")
            
            count = 0
            while True:
                try:
                    for_record = next(for_handle)
                    rev_record = next(rev_handle)
                    count += 1
                except StopIteration:
                    break # break condition for end of either forward or reverse read file
                
                if for_read_len > len(for_record.seq) or rev_read_len > len(rev_record.seq):
                    self.lenrequirement_discarded_reads += 1
                else:
                    #Truncate record lengths according to first entry. Fastq entries may therefore go up in length, but they cannot go down
                    for_seq_buff[:] = np.False_
                    rev_seq_buff[:] = np.False_
                    
                    parseOps.seq_to_np(for_record.seq,for_seq_buff,for_read_len)
                    parseOps.seq_to_np(rev_record.seq,rev_seq_buff,rev_read_len)
                    for_qual_buff[:] = for_record.letter_annotations['phred_quality'][:for_read_len]
                    rev_qual_buff[:] = rev_record.letter_annotations['phred_quality'][:rev_read_len]

                    for_params_index = None
                    rev_params_index = None
                    
                    if np.mean(for_qual_buff) >= self.min_mean_qual and np.mean(rev_qual_buff) >= self.min_mean_qual:
                                                
                        pass_filter_for = False
                        for seqform_index in range(num_seqform_for):
                            for amplicon_index in range(len(self.seqform_for_params[seqform_index])):
                                pass_filter_for = check_seqform_match(for_seq_buff,  
                                                                      self.seqform_for_params[seqform_index][amplicon_index]['P'],
                                                                      self.seqform_for_params[seqform_index][amplicon_index]['U'],
                                                                      self.max_mismatch_template)
                                if pass_filter_for:
                                    break
                            
                            if pass_filter_for:
                                for_params_index = int(seqform_index)
                                break
                        
                        pass_filter_rev = False
                        some_pass_filter_rev = False # kept in memory in case an amplicon variant requires looking for other rev seqforms
                        pass_filter_rev_amp = False  
                        all_sub_amplicons_pass = False 
                        rev_amp_binary = None
                        curr_rev_amp_binary = None

                        if pass_filter_for:
                            for seqform_index in range(num_seqform_rev):
                                all_sub_amplicons_pass = False
                                for amplicon_index in range(len(self.seqform_rev_params[seqform_index])):
                                    pass_filter_rev = check_seqform_match(rev_seq_buff,  
                                                                          self.seqform_rev_params[seqform_index][amplicon_index]['P'],
                                                                          self.seqform_rev_params[seqform_index][amplicon_index]['U'],
                                                                          self.max_mismatch_template)
                                    some_pass_filter_rev = some_pass_filter_rev or pass_filter_rev # flag that at least one sequence form succeeded
                                    all_sub_amplicons_pass = False
                                    
                                    
                                    if pass_filter_rev:
                                        rev_params_index = int(seqform_index)
                                        all_sub_amplicons_pass = True
                                        tot_rev_amp_bases = 0
                                        for j in range(len(self.seqform_rev_params[seqform_index][amplicon_index]['A'])): # loop through sub-amplicon-sequences
                                            pass_filter_rev_amp = get_subrecord(rev_seq_buff, 
                                                                                self.seqform_rev_params[seqform_index][amplicon_index]['A'][j], 
                                                                                self.max_mismatch_amplicon,True,rev_read_len)
                                            curr_rev_amp_binary = rev_amp_binary_list[seqform_index][amplicon_index][j]
                                            if not pass_filter_rev_amp:
                                                all_sub_amplicons_pass = False
                                                # not breaking out of loop at this stage, since un-recognized amplicons will continue 
                                                # to have information stored in curr_rev_amp_binary for "invalid" amplicon output
                                            else:
                                                tot_rev_amp_bases += np.sum(curr_rev_amp_binary) # edit 2/2019, again 7/12/19
                                        
                                    if all_sub_amplicons_pass:
                                        rev_amp_binary = curr_rev_amp_binary
                                        match_amplicon_index = int(amplicon_index)
                                        break
                                    
                                if all_sub_amplicons_pass: # note: it must be insufficient to break out of trying other seqform_rev options if the amplicon filter has not been passed
                                    break
                                              
                        if pass_filter_for and pass_filter_rev and all_sub_amplicons_pass: # passed all tests
                            
                            if not (rev_amp_filehandle is None):
                                mystr = parseOps.np_to_seq(rev_seq_buff,rev_amp_binary,rev_list_buff) # Edit 7/14/19 str()
                                
                                if len(rev_record.seq) > rev_read_len:
                                    mystr = str(mystr + str(rev_record.seq)[rev_read_len:])
                                    tot_rev_amp_bases += (len(rev_record.seq) - rev_read_len)

                                # will truncate final sub-amplicon-sequence
                                if '-amplicon_terminate' in self.mySettings:
                                    old_len = len(mystr)
                                    mystr = self.truncate_amplicon(mystr)  # Edit 7/14/19 str(return)
                                    change_in_len = len(mystr)-old_len
                                    if tot_rev_amp_bases + change_in_len < self.filter_amplicon_window:
                                        all_sub_amplicons_pass = False

                                if all_sub_amplicons_pass: # re-check this after the above conditional
                                    rev_amp_filehandle.write('>' + rev_record.id + '\n')
                                    rev_amp_filehandle.write(mystr + '\n')
                                    rev_amp_baseTally.add_record(mystr,1,6) #record base-statistics for only first 6 bases of gene

                            if all_sub_amplicons_pass: # re-check this after the above conditional
                                self.num_retained += 1
                                for for_uxi_binary,for_uxi_handle,for_uxi_baseTally in itertools.izip(for_uxi_binary_list[for_params_index][0],for_uxi_filehandles,for_uxi_baseTally_list):
                                    for_uxi_handle.write('>' + for_record.id + '\n')
                                    mystr = parseOps.np_to_seq(for_seq_buff,for_uxi_binary,for_list_buff)
                                    for_uxi_handle.write(mystr + '\n')
                                    for_uxi_baseTally.add_record(mystr,1)
                                for rev_uxi_binary,rev_uxi_handle,rev_uxi_baseTally in itertools.izip(rev_uxi_binary_list[rev_params_index][match_amplicon_index],rev_uxi_filehandles,rev_uxi_baseTally_list):
                                    rev_uxi_handle.write('>' + rev_record.id + '\n')
                                    mystr = parseOps.np_to_seq(rev_seq_buff,rev_uxi_binary,rev_list_buff)
                                    rev_uxi_handle.write(mystr + '\n')
                                    rev_uxi_baseTally.add_record(mystr,1)
                                    
                                amp_match_handle.write(str(match_amplicon_index) + '\n')
                                phred_file.write(str(float(sum(for_record.letter_annotations['phred_quality']))/len(for_record.seq)) + "," + str(float(sum(rev_record.letter_annotations['phred_quality']))/len(rev_record.seq)) + "\n")
                                #print index of forward/reverse auxiliary-assignments that match current read
                                for_auxassign_handle.write(str(for_params_index) + '\n') 
                                rev_auxassign_handle.write(str(rev_params_index) + '\n')
                            
                        if not all_sub_amplicons_pass:
                            if pass_filter_for and some_pass_filter_rev: # amplicon is a problem, but everything else makes sense
                                self.num_amplicon_invalid += 1
                                mystr = parseOps.np_to_seq(rev_seq_buff,curr_rev_amp_binary, rev_list_buff)
                                rev_amp_invalid_fastq.write('>' + rev_record.id + '\n')
                                if len(rev_record.seq) > rev_read_len:
                                    rev_amp_invalid_fastq.write(mystr + str(rev_record.seq)[rev_read_len:] + '\n')
                                else:
                                    rev_amp_invalid_fastq.write(mystr + '\n')
                                for for_uxi_binary,for_uxi_handle in itertools.izip(for_uxi_binary_list[for_params_index][0],for_uxi_amp_invalid_filehandles):
                                    mystr = parseOps.np_to_seq(for_seq_buff, for_uxi_binary, for_list_buff)
                                    for_uxi_handle.write('>' + for_record.id + '\n')
                                    for_uxi_handle.write(mystr + '\n')
                                    
                                for rev_uxi_binary,rev_uxi_handle in itertools.izip(rev_uxi_binary_list[rev_params_index][0],rev_uxi_amp_invalid_filehandles):
                                    mystr = parseOps.np_to_seq(rev_seq_buff,rev_uxi_binary, rev_list_buff)
                                    rev_uxi_handle.write('>' + rev_record.id + '\n')
                                    rev_uxi_handle.write(mystr + '\n')
                                    
                            else:
                                for_discarded_fasta.write('>' + for_record.id + '\n')
                                for_discarded_fasta.write(str(for_record.seq) + '\n')
                                rev_discarded_fasta.write('>' + rev_record.id + '\n')
                                rev_discarded_fasta.write(str(rev_record.seq) + '\n')
                                self.num_discarded += 1
                    else:
                        num_discarded_qscore += 1
                        
                if (self.num_discarded+self.num_retained+self.num_amplicon_invalid+num_discarded_qscore)%100000 == 0:
                    sysOps.throw_status('Analyzed ' + str(self.num_discarded+self.num_retained+self.num_amplicon_invalid+num_discarded_qscore) + ' reads. Discarded ' 
                                        + str(self.num_discarded) + ', retained ' 
                                        + str(self.num_retained) + ' reads, discarded ' 
                                        + str(num_discarded_qscore) + ' low q-score, set aside ' 
                                        + str(self.num_amplicon_invalid) + " invalid-amplicon reads.")
            
            
            for_handle.close()
            rev_handle.close()
                    
        sysOps.throw_status('Analyzed ' + str(self.num_discarded+self.num_retained+self.num_amplicon_invalid+num_discarded_qscore) + ' reads. Discarded ' + str(self.num_discarded) + ', retained ' + str(self.num_retained) + ' reads, discarded ' + str(self.lenrequirement_discarded_reads) + ' invalid-length reads, set aside ' + str(self.num_amplicon_invalid) + " invalid-amplicon reads.")
        sysOps.throw_status('Discarded ' + str(num_discarded_qscore) + ' due to quality scores.')
        for this_handle in for_uxi_filehandles:
            this_handle.close()
        for this_handle in rev_uxi_filehandles:
            this_handle.close()
        for this_handle in for_uxi_amp_invalid_filehandles:
            this_handle.close()
        for this_handle in rev_uxi_amp_invalid_filehandles:
            this_handle.close()
        
        if not (rev_amp_filehandle is None):
            rev_amp_filehandle.close()
        
        for_discarded_fasta.close()
        rev_discarded_fasta.close()
        rev_amp_invalid_fastq.close()
        amp_match_handle.close()
        for_auxassign_handle.close()
        rev_auxassign_handle.close()
        phred_file.close()
        
        sysOps.throw_status("Printing summary-statistic output for sample ...")
        with open(sysOps.globaldatapath + self.output_prefix + 'summary_stats.txt','w') as summary_stat_file:
            summary_stat_file.write(str(self.num_retained))
        
        sysOps.throw_status("Printing bases-statistics")
        with open(sysOps.globaldatapath +self.output_prefix + 'base_stats.txt','w') as base_stats_file: #write all averaged statistics concerning bases and quality-scores to base_stats.txt
            
            for i in range(len(for_uxi_baseTally_list)):
                base_stats_file.write('for_uxi' + str(i) + ":\n")
                base_stats_file.write(for_uxi_baseTally_list[i].to_str())
            for i in range(len(rev_uxi_baseTally_list)):
                base_stats_file.write('rev_uxi' + str(i) + ":\n")
                base_stats_file.write(rev_uxi_baseTally_list[i].to_str())
            if not (rev_amp_filehandle is None):
                base_stats_file.write('rev_amp' + str(i) + ":\n")
                base_stats_file.write(rev_amp_baseTally.to_str())
        
        sysOps.throw_status("Files closed. Writing library stats and primer capacities ...")
        
        self.output_lib_stats(self.output_prefix + 'lib_stats.txt', self.output_prefix + 'base_stats.txt')
                        
        sysOps.throw_status("Completed partitioning fastq, retained " + str(self.num_retained) + " out of " + str(self.num_discarded+self.num_retained) + " correct-length reads and " + str(self.lenrequirement_discarded_reads+self.num_discarded+self.num_retained) + " total reads.")

def subsample(seqform_for_params,seqform_rev_params,output_prefix):
    # Function for data sub-sampling to perform rarefaction analysis
    # Creates directory structure that duplicates those for individual libraries
    
    read_counts = list()
    this_read_count = 100000 # smallest read count to subsample, will proceed upward by factors of 2 
    all_for_uxi_filehandles = list()
    all_rev_uxi_filehandles = list()
    all_rev_amp_filehandles = list()
    all_amp_match_handles = list()
    
    try:
        with open(sysOps.globaldatapath + output_prefix + 'lib_stats.txt','rU') as retained_read_file:
            for line in retained_read_file:
                my_line = line.strip('\n').split(':')
                if my_line[0] == 'num_retained':
                    tot_retained_reads = int(my_line[1]) # number after colon is total retained reads
            sysOps.throw_status('According to ' + sysOps.globaldatapath + output_prefix + 'lib_stats.txt, ' + str(tot_retained_reads) + ' reads were retained.')
    except:
        sysOps.throw_exception('Error: unable to access num_retained in file ' +sysOps.globaldatapath + output_prefix + 'lib_stats.txt')
        sysOps.exitProgram()
        
    [subdirnames, filenames] = sysOps.get_directory_and_file_list()
    subsampled_readindices = list()
    readcounts = list()
    amp_match_found = sysOps.check_file_exists(output_prefix + 'amp_match.txt')
    readcount_index = 0
    num_for_uxis = len(seqform_for_params[0][0]['U'])
    num_rev_uxis = len(seqform_rev_params[0][0]['U'])
    while this_read_count < tot_retained_reads:
        if ('sub' + str(this_read_count)) not in subdirnames:
            sysOps.throw_status('Sub-sampling ' + str(this_read_count) + ' reads ...')
            read_counts.append(int(this_read_count))
            dirname = sysOps.globaldatapath + 'sub' + str(this_read_count) + '//'
            os.mkdir(dirname)
            shutil.copyfile(sysOps.globaldatapath + 'libsettings.txt', dirname + 'libsettings.txt')
            if sysOps.check_file_exists('amplicon_refs.txt'):
                shutil.copyfile(sysOps.globaldatapath + 'amplicon_refs.txt', dirname + 'amplicon_refs.txt')
            all_for_uxi_filehandles.append(list())
            for i in range(num_for_uxis): # assumes all sequence forms have the same set of UMI's
                all_for_uxi_filehandles[readcount_index].append(open(dirname + output_prefix 
                                                                     + 'for_uxi' + str(i) + '.fasta','w'))
            all_rev_uxi_filehandles.append(list())
            for i in range(num_rev_uxis): # assumes all sequence forms have the same set of UMI's
                all_rev_uxi_filehandles[readcount_index].append(open(dirname + output_prefix 
                                                                     + 'rev_uxi' + str(i) + '.fasta','w'))
            all_rev_amp_filehandles.append(open(dirname + output_prefix + 'rev_amp0.fasta','w'))
            if amp_match_found:
                all_amp_match_handles.append(open(dirname + output_prefix + 'amp_match.txt','w'))
            subsampled_readindices.append(list(sorted(random.sample(range(tot_retained_reads),this_read_count))))
            readcounts.append(int(this_read_count))
            readcount_index += 1
        else:
            sysOps.throw_status('Skipping sub-sampling ' + str(this_read_count) + ' due to pre-existing directory ...')
        this_read_count *= 2
        
    tot_readcount_indices = int(readcount_index)
    
    if tot_readcount_indices == 0:
        sysOps.throw_status('All sub-sampled data found pre-written.')
        return
    
    per_subsample_indices = np.zeros(tot_readcount_indices,dtype=np.int64)
    
    for_uxi_infiles = list()
    rev_uxi_infiles = list()
    for i in range(num_for_uxis): # assumes all sequence forms have the same set of UMI's
        for_uxi_infiles.append(open(sysOps.globaldatapath + output_prefix + 'for_uxi' + str(i) + '.fasta','rU'))
    for i in range(num_rev_uxis): # assumes all sequence forms have the same set of UMI's
        rev_uxi_infiles.append(open(sysOps.globaldatapath + output_prefix + 'rev_uxi' + str(i) + '.fasta','rU'))
     
    if amp_match_found:
        amp_match_infile = open(sysOps.globaldatapath + output_prefix + 'amp_match.txt','rU')
    rev_amp_infile = open(sysOps.globaldatapath + output_prefix + 'rev_amp0.fasta','rU')
    
    current_for_uxi_headers = [str() for i in range(num_for_uxis)]
    current_for_uxi_seqs = [str() for i in range(num_for_uxis)]
    current_rev_uxi_headers = [str() for i in range(num_rev_uxis)]
    current_rev_uxi_seqs = [str() for i in range(num_rev_uxis)]
    sysOps.throw_status('Writing sub-sampled data ...')
    for read_num in range(tot_retained_reads):
        for i in range(num_for_uxis):
            current_for_uxi_headers[i] = for_uxi_infiles[i].readline()
            current_for_uxi_seqs[i] = for_uxi_infiles[i].readline()
        for i in range(num_rev_uxis):
            current_rev_uxi_headers[i] = rev_uxi_infiles[i].readline()
            current_rev_uxi_seqs[i] = rev_uxi_infiles[i].readline()
        if amp_match_found:
            amp_match = amp_match_infile.readline()
        rev_amp_header = rev_amp_infile.readline()
        rev_amp_seq = rev_amp_infile.readline()
        for readcount_index in range(tot_readcount_indices):
            if (per_subsample_indices[readcount_index] < readcounts[readcount_index]
                and subsampled_readindices[readcount_index][per_subsample_indices[readcount_index]] == read_num):
                for i in range(num_for_uxis):
                    all_for_uxi_filehandles[readcount_index][i].write(current_for_uxi_headers[i])
                    all_for_uxi_filehandles[readcount_index][i].write(current_for_uxi_seqs[i])
                for i in range(num_rev_uxis):
                    all_rev_uxi_filehandles[readcount_index][i].write(current_rev_uxi_headers[i])
                    all_rev_uxi_filehandles[readcount_index][i].write(current_rev_uxi_seqs[i])
                if amp_match_found:
                    all_amp_match_handles[readcount_index].write(amp_match)
                all_rev_amp_filehandles[readcount_index].write(rev_amp_header)
                all_rev_amp_filehandles[readcount_index].write(rev_amp_seq)
                per_subsample_indices[readcount_index] += 1
    
    for i in range(num_for_uxis):
        for_uxi_infiles[i].close()
        for readcount_index in range(tot_readcount_indices):
            all_for_uxi_filehandles[readcount_index][i].close()
    for i in range(num_rev_uxis): 
        rev_uxi_infiles[i].close()
        for readcount_index in range(tot_readcount_indices):
            all_rev_uxi_filehandles[readcount_index][i].close()
    for readcount_index in range(tot_readcount_indices):
        if amp_match_found:
            all_amp_match_handles[readcount_index].close()
        all_rev_amp_filehandles[readcount_index].close()
    if amp_match_found:
        amp_match_infile.close()

def check_seqform_match(seq_np,this_seqform_params_P,this_seqform_params_U,max_frac_mismatch):
    pass_filter = True
    
    for my_param in this_seqform_params_P:
        pass_filter = get_subrecord(seq_np, my_param, max_frac_mismatch, True, None) #filter based on primer-sequence
        if not pass_filter:
            break
    
    if pass_filter:
        for my_param in this_seqform_params_U:
            pass_filter = get_subrecord(seq_np, my_param, max_frac_mismatch, False, None)
            if not pass_filter:
                break
    
    return pass_filter      
            