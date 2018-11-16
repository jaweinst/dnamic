from Bio import SeqIO
from Bio import Seq
import numpy as np
import sysOps
import fileOps
import itertools
import alignOps
import os
from libOps import baseTally

def assign_umi_amplicons(trg_umi_cluster_file, trg_umi_fasta, amp_match_file, amp_seq_fasta, outfilename):
    #function will tally reads counted for each target umi across each amplicon-call, and return a csv file with the following columns:
    #(target umi cluster-index),(leading amplicon-call),(reads for leading amplicon-call),(total reads counted)
    
    sysOps.throw_status('Loading cluster-file ' + sysOps.globaldatapath + trg_umi_cluster_file)
    trg_umi_cluster_dict = fileOps.load_cluster_file_to_dictionary(trg_umi_cluster_file)
    #outputs dictionary with entries {uxi-sequence: [uxi-cluster-index, read-number]}

    trg_umi_handle = open(sysOps.globaldatapath +trg_umi_fasta, "rU")
    amp_seq_handle = open(sysOps.globaldatapath +amp_seq_fasta, "rU")
    realign_amplicons = False
    amp_match_handle = None
    try:
        sysOps.throw_status('Loading ' + sysOps.globaldatapath +amp_match_file)
        amp_match_handle = open(sysOps.globaldatapath +amp_match_file, "rU")
    except:
        sysOps.throw_status(sysOps.globaldatapath +amp_match_file + ' not found. Alignments will occur from sequence-consenses directly.')
        realign_amplicons = True
        if not sysOps.check_file_exists('amplicon_refs.txt'):
            sysOps.throw_exception('Error: ' + sysOps.globaldatapath + 'amplicon_refs.txt not found.')
            sysOps.exitProgram()
            
    trg_umi_dict = dict()
    trg_amp_seq_dict = dict()
    
    for trg_umi_record, amp_seq_record in itertools.izip(SeqIO.parse(trg_umi_handle, "fasta"),SeqIO.parse(amp_seq_handle, "fasta")):
        
        if not realign_amplicons:
            amp_match = int(amp_match_handle.readline().strip('\n'))
        else:
            amp_match = -1
            
        trg_umi_seq = str(trg_umi_record.seq)
        if trg_umi_seq in trg_umi_cluster_dict:
            trg_umi_index = str(trg_umi_cluster_dict[trg_umi_seq][0]) #uxi cluster-index
            if trg_umi_index in trg_umi_dict:
                if amp_match in trg_umi_dict[trg_umi_index]:
                    trg_umi_dict[trg_umi_index][amp_match] += 1  #add 1, because every read is being entered
                else:
                    trg_umi_dict[trg_umi_index][amp_match] = 1
            else:
                trg_umi_dict[trg_umi_index] = dict()
                trg_amp_seq_dict[trg_umi_index] = baseTally()
                trg_umi_dict[trg_umi_index][amp_match] = 1
            
            trg_amp_seq_dict[trg_umi_index].add_record(str(amp_seq_record.seq),1)
    
    trg_umi_handle.close()
    amp_seq_handle.close()
    if not realign_amplicons:
        amp_match_handle.close()
    
    csvfile = open(sysOps.globaldatapath + outfilename,'w')
    fastafile = open(sysOps.globaldatapath + outfilename[:outfilename.rfind('.')] + '.fasta','w')
    ref_sequences = list()
    if realign_amplicons and sysOps.check_file_exists('amplicon_refs.txt'):
        with open(sysOps.globaldatapath + 'amplicon_refs.txt','rU') as ref_file_handle:
            for ref_line in ref_file_handle:
                [ref_name,ref_seq] = ref_line.strip('\n').upper().split('|')
                # amplicon_refs.txt will contain sequences in reverse complementary orientation. We therefore reverse both complementarity and order
                ref_sequences.append([str(Seq.Seq(my_ref_seq).reverse_complement()) for my_ref_seq in reversed(ref_seq.split(','))])
        mySettings = fileOps.read_settingsfile_to_dictionary('libsettings.txt')
        max_mismatch_amplicon = float(mySettings["-max_mismatch_amplicon"][0])
        trg_umi_index_dict = dict()
    
    accepted_consensus_sequences = 0
    inadmis_consensus_sequences = 0
    for trg_umi_index in trg_umi_dict:
        max_tally = 0
        tot_tally = 0
        
        for amp_match in trg_umi_dict[trg_umi_index]:
            
            my_tally = trg_umi_dict[trg_umi_index][amp_match]
            
            if my_tally >= max_tally:
                max_tally = int(my_tally)
                max_match = int(amp_match)
                
            tot_tally += int(my_tally)
        
        consensus_seq = str(trg_amp_seq_dict[trg_umi_index].get_str_consensus())
        
        if realign_amplicons:
            # perform direct, un-gapped alignment of consensus_seq to reference options to obtain max_match
            max_match = -1
            max_tally = -1 # exclude max_tally as count, since alignment is happening post-consensus
            min_mismatch_count = -1
            for i in range(len(ref_sequences)):
                all_subamplicons_pass = True
                start_index = 0
                tot_mismatches = 0
                for j in range(len(ref_sequences[i])): # loop through sub-amplicon-sequences
                    ref_subamplicon_len = len(ref_sequences[i][j])
                    my_mismatches, minlen = alignOps.count_mismatches(ref_sequences[i][j], consensus_seq[start_index:(start_index+ref_subamplicon_len)])
                    if minlen == 0:
                        all_subamplicons_pass = False
                        break
                    all_subamplicons_pass = all_subamplicons_pass and (my_mismatches/float(minlen) <= max_mismatch_amplicon)
                    start_index += ref_subamplicon_len
                    tot_mismatches += my_mismatches
                if all_subamplicons_pass and (max_match < 0 or min_mismatch_count < tot_mismatches):
                    max_match = int(i)
                    min_mismatch_count = int(tot_mismatches)
            
        if max_match >= 0:
            csvfile.write(trg_umi_index + "," + str(max_match) + "," + str(max_tally) + "," + str(tot_tally) + "\n")
            fastafile.write(">" + trg_umi_index + '\n')
            fastafile.write(consensus_seq + '\n')
            if realign_amplicons:
                trg_umi_index_dict[trg_umi_index] = True
            accepted_consensus_sequences += 1
        else:
            inadmis_consensus_sequences += 1
    
    csvfile.close()
    fastafile.close()
    sysOps.throw_status('Discarded ' + str(inadmis_consensus_sequences) + '/' + str(accepted_consensus_sequences+inadmis_consensus_sequences) + ' sequences in writing ' + sysOps.globaldatapath + outfilename + ' due to inadequate amplicon match.')
    
    if realign_amplicons:
        # create a new consensus pairing file that's filtered with the accepted trg umi indices
        [dirnames,filenames] = sysOps.get_directory_and_file_list()
        consensus_filenames = [filename for filename in filenames if filename.startswith('consensus')]
        for consensus_filename in consensus_filenames: # find all consensus files present
            accepted_consensus_sequences = 0
            inadmis_consensus_sequences = 0
            os.rename(sysOps.globaldatapath + consensus_filename, sysOps.globaldatapath + 'unfiltered_' + consensus_filename)
            with open(sysOps.globaldatapath + consensus_filename,'w') as new_consensus_file:
                with open(sysOps.globaldatapath + 'unfiltered_' + consensus_filename,'rU') as old_consensus_file:
                    for old_consensus_file_line in old_consensus_file:
                        consensus_list = old_consensus_file_line.strip('\n').split(',') # [uei_index, bcn_umi_index, trg_umi_index, read_count, (additional variables)]
                        if consensus_list[2] in trg_umi_index_dict:
                            new_consensus_file.write(old_consensus_file_line)
                            accepted_consensus_sequences += 1
                        else:
                            inadmis_consensus_sequences += 1
            sysOps.throw_status('Discarded ' + str(inadmis_consensus_sequences) + '/' + str(accepted_consensus_sequences+inadmis_consensus_sequences) + ' consensus-pairings in writing ' + sysOps.globaldatapath + consensus_filename + ' due to inadequate amplicon match.')
        if len(consensus_filenames) == 0:
            sysOps.throw_exception('Error: no consensus files available to update with realigned amplicon information. Exiting.')
            sysOps.exitProgram()
    

def assign_umi_pairs(uei_cluster_file,bcn_umi_cluster_file,trg_umi_cluster_file, uei_fasta_file, bcn_umi_fasta_file,trg_umi_fasta_file, outfile_prefix,filter_val=0.75,include_inv_amp=False):
    
    #at most filter_val fraction of total bases in given uxi allowed to be the same 
    
    #Cluster-files have row-formats: uxi-cluster-index_uxi-sequence_read-number
    #load_cluster_file_to_dictionary outputs dictionary with entries {uxi-sequence: [uxi-cluster-index, read-number]}
    
    sysOps.throw_status("Finalizing consensus UMI sequences ...")
    uei_cluster_dict = fileOps.load_cluster_file_to_dictionary(uei_cluster_file)
    bcn_umi_cluster_dict = fileOps.load_cluster_file_to_dictionary(bcn_umi_cluster_file)
    trg_umi_cluster_dict = fileOps.load_cluster_file_to_dictionary(trg_umi_cluster_file)
    
    uei_clust_readcount_tally = dict()
    bcn_umi_clust_readcount_tally = dict()
    trg_umi_clust_readcount_tally = dict()
    
    # initiate tally dictionaries addressed by clust index
    # one element per file_set_index (if invalid amplicon sequences are being excluded, only first index will be populated)
    for uei_seq in uei_cluster_dict:
        uei_clust_readcount_tally[str(uei_cluster_dict[uei_seq][0])] = [0,0] 
    for umi_seq in bcn_umi_cluster_dict:
        bcn_umi_clust_readcount_tally[str(bcn_umi_cluster_dict[umi_seq][0])] = [0,0]
    for umi_seq in trg_umi_cluster_dict:
        trg_umi_clust_readcount_tally[str(trg_umi_cluster_dict[umi_seq][0])] = [0,0]
    
    #outputs dictionary with entries {uxi-sequence: [uxi-cluster-index, read-number]}
    #generate list of list of lists with index-order uei,umi
    uei_umi_dict = dict()
    inadmis_seq_count = 0
    admis_seq_count = 0

    uei_fasta_list = [uei_fasta_file]
    bcn_umi_fasta_list = [bcn_umi_fasta_file]
    trg_umi_fasta_list = [trg_umi_fasta_file]
    if include_inv_amp:
        uei_fasta_list.append(uei_fasta_file[:uei_fasta_file.find('.')] + '_amp_inv' + uei_fasta_file[uei_fasta_file.find('.'):])
        bcn_umi_fasta_list.append(bcn_umi_fasta_file[:bcn_umi_fasta_file.find('.')] + '_amp_inv' + bcn_umi_fasta_file[bcn_umi_fasta_file.find('.'):])
        trg_umi_fasta_list.append(trg_umi_fasta_file[:trg_umi_fasta_file.find('.')] + '_amp_inv' + trg_umi_fasta_file[trg_umi_fasta_file.find('.'):])

    file_set_index = 0
    sysOps.throw_status("Inputting data to UEI-UMI dictionary using file-sets: " + str(uei_fasta_list) + ", " + str(bcn_umi_fasta_list) + ", " + str(trg_umi_fasta_list))
    
    for uei_fasta,bcn_umi_fasta,trg_umi_fasta in itertools.izip(uei_fasta_list,bcn_umi_fasta_list,trg_umi_fasta_list):
        uei_handle = open(sysOps.globaldatapath + uei_fasta, "rU")
        bcn_umi_handle = open(sysOps.globaldatapath + bcn_umi_fasta, "rU")
        trg_umi_handle = open(sysOps.globaldatapath + trg_umi_fasta, "rU")
        
        for uei_record,bcn_umi_record,trg_umi_record in itertools.izip(SeqIO.parse(uei_handle, "fasta"),SeqIO.parse(bcn_umi_handle, "fasta"),SeqIO.parse(trg_umi_handle, "fasta")):
            uei_seq = str(uei_record.seq)
            bcn_umi_seq = str(bcn_umi_record.seq)
            trg_umi_seq = str(trg_umi_record.seq)
            max_uei_frac = max(np.bincount([('ACGT').index(s) for s in uei_seq]))/float(len(uei_seq))
            max_bcn_umi_frac = max(np.bincount([('ACGT').index(s) for s in bcn_umi_seq]))/float(len(bcn_umi_seq))
            max_trg_umi_frac = max(np.bincount([('ACGT').index(s) for s in trg_umi_seq]))/float(len(trg_umi_seq))
            if max_uei_frac<=filter_val and max_bcn_umi_frac<=filter_val and max_trg_umi_frac<=filter_val:
                uei_clust_ind = str(uei_cluster_dict[uei_seq][0])
                bcn_umi_clust_ind = str(bcn_umi_cluster_dict[bcn_umi_seq][0])
                trg_umi_clust_ind = str(trg_umi_cluster_dict[trg_umi_seq][0])
                uei_clust_readcount_tally[uei_clust_ind][file_set_index] += 1
                bcn_umi_clust_readcount_tally[bcn_umi_clust_ind][file_set_index] += 1
                trg_umi_clust_readcount_tally[trg_umi_clust_ind][file_set_index] += 1
                
                pair_str = bcn_umi_clust_ind + "_" + trg_umi_clust_ind
                if uei_clust_ind in uei_umi_dict and uei_umi_dict[uei_clust_ind][2] == file_set_index:
                    #if uei from read has already been inserted into uei-umi dictionary
                    if pair_str in uei_umi_dict[uei_clust_ind][0]: #if bcn-trg pair has already been added to this uei entry
                        pair_ind = uei_umi_dict[uei_clust_ind][0].index(pair_str)
                        uei_umi_dict[uei_clust_ind][1][pair_ind] += 1
                    else: # uei in uei_umi_dict -- but corresponding UMI-pair not found in existing list
                        uei_umi_dict[uei_clust_ind][0].append(pair_str)
                        uei_umi_dict[uei_clust_ind][1].append(1)
                    admis_seq_count += 1
                elif uei_clust_ind not in uei_umi_dict: # uei not yet in uei_umi_dict -- create new list
                    uei_umi_dict[uei_clust_ind] = [[pair_str],[1],int(file_set_index)]
                    admis_seq_count += 1
                else:
                    inadmis_seq_count += 1 # if UEI has been found but not as part of the first file_set_index for which it was detected (this depends on orderring any invalid-amplicon files second in the fasta-lists above), then disregard
            else:
                inadmis_seq_count += 1
        uei_handle.close()
        bcn_umi_handle.close()
        trg_umi_handle.close()
        file_set_index += 1
    
    sysOps.throw_status('Did not use ' + str(inadmis_seq_count) + '/' + str(admis_seq_count + inadmis_seq_count) + ' pairings due to repetitive base-usage in UMI or UEI sequence.')
    #elements of uei_umi_dict are now list of cluster indices (ordered uei,bcn,trg,# times that element has been called)
    
    #convert embedded dictionaries into list
    list_output = list()
    for uei_el in uei_umi_dict:
        for i in range(len(uei_umi_dict[uei_el][0])):
            pair_str = uei_umi_dict[uei_el][0][i]
            [bcn_umi_el,trg_umi_el] = pair_str.split('_')
            list_output.append([int(uei_el),int(bcn_umi_el),int(trg_umi_el),int(uei_umi_dict[uei_el][1][i]),int(uei_umi_dict[uei_el][2])])
    
    del uei_umi_dict
    list_output.sort(key = lambda row: (row[0], row[1], row[2], -row[3], row[4])) #sort by uei-cluster, then beacon-umi-cluster, then target-umi-cluster, then decreasing read-count        
    
    sysOps.throw_status("Writing file ...")
    with open(sysOps.globaldatapath +outfile_prefix + "_filter" + str(filter_val) + "_uei_umi.csv" ,'w') as outfile_handle:
        for row in list_output:
            outfile_handle.write(','.join([str(s) for s in row]) + "\n")
            
    sysOps.throw_status("Tallying clusters ...")
    uei_clust_counts = [[0,0],[0,0],[0,0]]
    bcn_umi_clust_counts = [[0,0],[0,0],[0,0]]
    trg_umi_clust_counts = [[0,0],[0,0],[0,0]]
    uei_clust_counts_inclusive = [[0,0],[0,0],[0,0]]
    bcn_umi_clust_counts_inclusive = [[0,0],[0,0],[0,0]]
    trg_umi_clust_counts_inclusive = [[0,0],[0,0],[0,0]]
    
    # with file_set_index=0 corresponding to amplicon-valid and file_set_index=1 corresponding to amplicon-invalid and
    # a cluster is counted for file_set_index = 0 if none of its members are valid 
    for uei_clust_ind in uei_clust_readcount_tally:
        file_set_index = 0
        this_readcount = uei_clust_readcount_tally[uei_clust_ind][file_set_index] 
        if this_readcount == 0:
            file_set_index = 1
            this_readcount = uei_clust_readcount_tally[uei_clust_ind][file_set_index] 
        if this_readcount > 0:
            uei_clust_counts[min(this_readcount,3)-1][file_set_index] += 1
        uei_clust_counts_inclusive[min(uei_clust_readcount_tally[uei_clust_ind][0],3)-1][0] += 1
        uei_clust_counts_inclusive[min(uei_clust_readcount_tally[uei_clust_ind][1],3)-1][1] += 1
                
    for umi_clust_ind in bcn_umi_clust_readcount_tally:
        file_set_index = 0
        this_readcount = bcn_umi_clust_readcount_tally[umi_clust_ind][file_set_index] 
        if this_readcount == 0:
            file_set_index = 1
            this_readcount = bcn_umi_clust_readcount_tally[umi_clust_ind][file_set_index] 
        if this_readcount > 0:
            bcn_umi_clust_counts[min(this_readcount,3)-1][file_set_index] += 1
        bcn_umi_clust_counts_inclusive[min(bcn_umi_clust_readcount_tally[umi_clust_ind][0],3)-1][0] += 1
        bcn_umi_clust_counts_inclusive[min(bcn_umi_clust_readcount_tally[umi_clust_ind][1],3)-1][1] += 1
                
    for umi_clust_ind in trg_umi_clust_readcount_tally:
        file_set_index = 0
        this_readcount = trg_umi_clust_readcount_tally[umi_clust_ind][file_set_index] 
        if this_readcount == 0:
            file_set_index = 1
            this_readcount = trg_umi_clust_readcount_tally[umi_clust_ind][file_set_index] 
        if this_readcount > 0:
            trg_umi_clust_counts[min(this_readcount,3)-1][file_set_index] += 1
        trg_umi_clust_counts_inclusive[min(trg_umi_clust_readcount_tally[umi_clust_ind][0],3)-1][0] += 1
        trg_umi_clust_counts_inclusive[min(trg_umi_clust_readcount_tally[umi_clust_ind][1],3)-1][1] += 1
    
    with open(sysOps.globaldatapath + outfile_prefix + '_clust_stats.txt','w') as out_stats:
        tot_file_sets = 1
        if include_inv_amp:
            tot_file_sets = 2 
        for file_set_index in range(tot_file_sets):
            out_stats.write('uei:' + str(file_set_index) + ':' + ','.join([str(uei_clust_counts[i][file_set_index]) for i in range(3)]) + '\n')
            out_stats.write('bcn_umi:' + str(file_set_index) + ':' + ','.join([str(bcn_umi_clust_counts[i][file_set_index]) for i in range(3)]) + '\n')
            out_stats.write('trg_umi:' + str(file_set_index) + ':' + ','.join([str(trg_umi_clust_counts[i][file_set_index]) for i in range(3)]) + '\n')            
  
    sysOps.throw_status("Completed.")
    return

def assign_consensus_pairs(pairing_csv_file, min_pairing_readcount):
    '''
    Assumes CSV file with columns:
    1. UEI cluster-index
    2. Beacon UMI cluster-index
    3. Target UMI cluster-index
    4. Read-number
    '''
    
    sysOps.throw_status('Loading pairing file ' + pairing_csv_file + ' ...')
    uei_clust_index_dict = dict()
    
    with open(sysOps.globaldatapath +pairing_csv_file, 'rU') as csvfile:
        for line in csvfile:
            row = line.strip('\n').split(',')
            index_str = str(row[0]) #UEI cluster-index
            if index_str in uei_clust_index_dict:
                uei_clust_index_dict[index_str].append([int(row[1]), int(row[2]), int(row[3]), int(row[4])]) #append dictionary entry as list with row having indices of beacon- and target-umi clusters, the read-number, and the set-index (will all be 0 if invalid-amplicon reads are excluded)
            else:
                uei_clust_index_dict[index_str] = [[int(row[1]), int(row[2]), int(row[3]), int(row[4])]]
    
    #replace each entry with umi pairing having plurality of reads, in same indexed format
    sysOps.throw_status('Generating consensus-pairs ...')
    discarded_ueis = 0
    accepted_ueis = 0
    for uei_clust_el in uei_clust_index_dict:
        maxcount = 0
        secondmaxcount = 0 #detect ties, discard if tie exists
        maxcount_pair_bcn_index = -1
        maxcount_pair_trg_index = -1
        maxcount_set_index = -1
        for row in uei_clust_index_dict[uei_clust_el]:
            if(row[2]>=min_pairing_readcount and row[2]>maxcount): 
                secondmaxcount = int(maxcount)
                if maxcount_set_index >= 0 and maxcount_set_index != row[3]:
                    sysOps.throw_exception('Error: set-index mismatch.')
                    sysOps.exitProgram()
                maxcount_pair_bcn_index = int(row[0])
                maxcount_pair_trg_index = int(row[1])
                maxcount = int(row[2])
                maxcount_set_index = int(row[3])
            elif(row[2]>=min_pairing_readcount and row[2]>secondmaxcount):
                secondmaxcount = int(row[2])
                
        if maxcount>=min_pairing_readcount and maxcount > secondmaxcount: 
            # note: this condition requires that not only must the uei have at least min_pairing_readcount, 
            # but the plurality-tally be must min_pairing_readcount as well
            uei_clust_index_dict[uei_clust_el] = list([int(maxcount_pair_bcn_index),
                                                       int(maxcount_pair_trg_index),
                                                       int(maxcount),int(maxcount_set_index)])
            accepted_ueis += 1
        else:
            uei_clust_index_dict[uei_clust_el] = list()
            discarded_ueis += 1
    
    sysOps.throw_status('Outputting consensus-pairs with at least ' + str(min_pairing_readcount) + ' read-plurality. Accepted ' + str(accepted_ueis) + ' UEIs, discarded ' + str(discarded_ueis) + ' UEIs ...')
    #index outputted as uei-index, beacon-umi-index, target-umi-index, read-count        
    outfile_handle = open(sysOps.globaldatapath +"consensus_" + str(min_pairing_readcount) + "r_" + pairing_csv_file ,'w')

    for uei_clust_el in uei_clust_index_dict:
        if len(uei_clust_index_dict[uei_clust_el]) > 0:
            outfile_handle.write(uei_clust_el + "," + ",".join([str(s) for s in uei_clust_index_dict[uei_clust_el]]) + "\n")
    
    outfile_handle.close()    
     
    return

def print_final_results(trgcalls_filename,trgseq_filename):
    
    #output final_*.csv containing columns (index, -1 (beacon)/ target-amplicon match, 
    #                                            x, y, ..., segment
    #output final_feat*.csv containing columns (index, features, consensus sequence (if target)
    #
    [dirnames,filenames] = sysOps.get_directory_and_file_list()
    seq_dat_filename = [filename for filename in filenames if filename.startswith('seq_params')]
    seq_dat_filename = seq_dat_filename[0][len('seq_params_'):]
    
    for result_dat_file in filenames:
        if (result_dat_file.startswith('Xumi_') and not (sysOps.check_file_exists('final_' + result_dat_file))):
            key_dat_file = 'key' + seq_dat_filename[(seq_dat_filename.find('_')):]
            if sysOps.check_file_exists(key_dat_file):
                coords_dict = dict()
                sysOps.throw_status('Generating final output for ' + sysOps.globaldatapath + str(result_dat_file))
                result = np.loadtxt(sysOps.globaldatapath + result_dat_file, delimiter=',')
                for i in range(result.shape[0]):
                    coords_dict[str(int(result[i,0]))] = ','.join([str(x) for x in result[i,1:]])

                trg_match_dict = dict()                
                trg_match_file = open(sysOps.globaldatapath + trgcalls_filename,'rU')
                trg_seq_file = open(sysOps.globaldatapath + trgseq_filename,'rU')
                
                for line, fasta_record in itertools.izip(trg_match_file,SeqIO.parse(trg_seq_file, "fasta")):
                    [trg_umi_index, max_match, max_tally, tot_tally]  = line.strip('\n').split(',')
                    trg_match_dict[trg_umi_index] = [str(max_match),str(max_tally),str(tot_tally),str(fasta_record.seq)]
    
                trg_match_file.close()
                trg_seq_file.close()
                
                outfile = open(sysOps.globaldatapath + '//final_' + result_dat_file,'w')
                outfile_feat = open(sysOps.globaldatapath + '//final_feat_' + result_dat_file,'w')
                
                bcn_excluded = 0
                trg_excluded = 0
                with open(sysOps.globaldatapath + key_dat_file,'rU') as key_file:
                    for line in key_file:
                        [bcn0trg1,orig_index,mle_index] = line.strip('\n').split(',')
                        #key file columns: 0 or 1 (for beacon or target, respectively), cluster-index, MLE processing index
                        if mle_index in coords_dict:
                            outfile.write(orig_index + ',' + coords_dict[mle_index] + '\n')
                            if bcn0trg1 == '0':
                                outfile_feat.write(orig_index + ',-1,-1,-1,N\n')
                            else:
                                outfile_feat.write(orig_index + ',' + ','.join(trg_match_dict[orig_index]) + '\n')
                        else:
                            if bcn0trg1 == '0':
                                bcn_excluded += 1
                            else:
                                trg_excluded += 1
                sysOps.throw_status(str(bcn_excluded) + ' beacons, ' + str(trg_excluded) + ' targets excluded from final estimation')
                outfile.close()
                outfile_feat.close()
        
            else:
                sysOps.throw_exception(sysOps.globaldatapath + key_dat_file + ' does not exist.')
    return