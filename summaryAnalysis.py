import sysOps
import matOps
import fileOps
import itertools
import os
import numpy as np
from Bio import SeqIO
from Bio import Seq

def gather_rarefaction_data(conditions_filename = 'conditions.csv', outfilename = 'rarefaction_file.txt', raw_uxi_files = ['_for_uxi0.fasta','_for_uxi1.fasta','_rev_uxi0.fasta']):
    
    #use conditions conditions_filename to specify output order
    dirnames = list()
    with open(sysOps.globaldatapath + conditions_filename, 'rU') as conditions_handle:
        for myline in conditions_handle:
            thisline = myline.strip('\n').split(',')
            dirnames.append('lib_' + str(thisline[0]) + '_' + str(thisline[1]) + '_' + str(thisline[2]))
    
    outfile_1r = open(sysOps.globaldatapath +'1r_' + outfilename,'w')
    outfile_2r = open(sysOps.globaldatapath +'2r_' + outfilename,'w')
    outfile_3r = open(sysOps.globaldatapath +'3r_' + outfilename,'w')
    
    for dir in dirnames:
        print 'Gathering rarefaction data for directory ' + sysOps.globaldatapath + dir
        sum_reads_raw = 0
        with open(sysOps.globaldatapath +dir + '/' + raw_uxi_files[0],'rU') as uxi_file_handle:
            #first UMI/UEI file in list to count raw reads
            for uxi_record in SeqIO.parse(uxi_file_handle,'fasta'):
                sum_reads_raw += 1
        
        subsample = 500
        terminate = False
        while not terminate:
            all_diversities = []
            try:
                for my_raw_uxi_file in raw_uxi_files:
                    try:
                        cluster_file_handle = open(sysOps.globaldatapath +dir + '/thresh1_identical_sub' + str(subsample) + my_raw_uxi_file,'rU')
                        consensus_pairing_csv_file = dir + '/consensus_2r_sub' + str(subsample) + 'pairing_filter0.75_uei_umi.csv'
                    except:
                        terminate = True
                        try:
                            cluster_file_handle = open(sysOps.globaldatapath +dir + '/thresh1_identical_' + my_raw_uxi_file,'rU')
                            consensus_pairing_csv_file = dir + '/consensus_2r_pairing_filter0.75_uei_umi.csv'
                        except:
                            sysOps.throw_exception('Directory ' + sysOps.globaldatapath + dir + ' does not contain clustered file' +  sysOps.globaldatapath +dir + '/thresh1_identical_' + my_raw_uxi_file + '. Skipping ...')
                            break
                        
                        subsample = sum_reads_raw
                        
                    cluster_dict = dict()
                    for myline in cluster_file_handle:
                        thisline = myline.strip('\n').split('_')
                        if thisline[0] in cluster_dict:
                            cluster_dict[thisline[0]] += int(thisline[2])
                        else:
                            cluster_dict[thisline[0]] = int(thisline[2])
                            
                    cluster_file_handle.close()
    
                    diversity = [0,0,0] #first element is 1-read-gated diversity, second is 2-read-gated, third is 3-read-gated
                    for el in cluster_dict:
                        if cluster_dict[el]>=3:
                            diversity[0] += 1
                            diversity[1] += 1
                            diversity[2] += 1
                        elif cluster_dict[el]>=2:
                            diversity[0] += 1
                            diversity[1] += 1
                        else:
                            diversity[0] += 1
                            
                    all_diversities.append(diversity)
    
                #if sysOps.check_file_exists(consensus_pairing_csv_file):
                if False: #temp
                    sysOps.throw_status('Found ' + sysOps.globaldatapath + consensus_pairing_csv_file + '.')
                    min_uei_count = 2  
                    min_umi_readcount = 2
                    outname = 'minb' + str(min_uei_count) + 'v' + str(0) + '_' + str(min_umi_readcount) + 'r_filter0.75'
                    wmat_outfilename = 'noabundcorr_wmat_' + outname + '.csv'
                    sysOps.throw_status('Calling matOps.generate_wmat()')
                    [num_unique_trg, num_unique_bcn, trg_dict] = matOps.generate_wmat(consensus_pairing_csv_file, min_umi_readcount, min_umi_readcount, min_uei_count, wmat_outfilename = None)
                    
                    if num_unique_bcn>0:
                        filtered_minb_diversity_2r = [num_unique_bcn, sum([trg_dict[trg_el] for trg_el in trg_dict]), num_unique_trg]
                    else:
                        filtered_minb_diversity_2r = [0,0,0]
                else:
                    sysOps.throw_status(sysOps.globaldatapath + consensus_pairing_csv_file + ' not found.')
                    filtered_minb_diversity_2r = []
                    
                outfile_1r.write(','.join([dir, str(subsample), ','.join([str(my_diversity[0]) for my_diversity in all_diversities])]) + '\n')
                outfile_2r.write(','.join([dir, str(subsample), ','.join([str(my_diversity[1]) for my_diversity in all_diversities]), ','.join([str(s) for s in filtered_minb_diversity_2r])]) + '\n')
                outfile_3r.write(','.join([dir, str(subsample), ','.join([str(my_diversity[2]) for my_diversity in all_diversities])]) + '\n')                 
            
            except:
                terminate = True
                
            subsample *= 2
        
    outfile_1r.close()
    outfile_2r.close()
    outfile_3r.close()
    
def gather_raw_read_stats(conditions_filename = 'conditions.csv', outfilename = 'stats_file.txt'):
    #use conditions conditions_filename to specify output order
    dirnames = list()
    with open(sysOps.globaldatapath + conditions_filename, 'rU') as conditions_handle:
        for myline in conditions_handle:
            thisline = myline.strip('\n').split(',')
            dirnames.append('lib_' + str(thisline[0]) + '_' + str(thisline[1]) + '_' + str(thisline[2]))
    
    raw_stats_outfile = open(sysOps.globaldatapath + 'raw_' + outfilename,'w')
    for dir in dirnames:
        [accepted,discarded,amp_discarded,all_amp_matches] = fileOps.gather_raw_read_stats(dir + '/' ,raw_uxi_file='_for_uxi0.fasta',amp_match_file = '_amp_match.txt')
        thisline = ','.join([dir , ','.join([str(accepted),str(discarded),str(amp_discarded)]), ','.join([str(x) for x in all_amp_matches])])
        print sysOps.globaldatapath + thisline
        raw_stats_outfile.write(thisline + '\n')
    raw_stats_outfile.close()
    
    return

def gather_stats(conditions_filename = 'conditions.csv', outfilename = 'stats_file.txt'):
    #use conditions conditions_filename to specify output order
    dirnames = list()
    with open(sysOps.globaldatapath + conditions_filename, 'rU') as conditions_handle:
        for myline in conditions_handle:
            thisline = myline.strip('\n').split(',')
            dirnames.append('lib_' + str(thisline[0]) + '_' + str(thisline[1]) + '_' + str(thisline[2]))
    
    outfile = open(sysOps.globaldatapath +outfilename,'w')
    for dir in dirnames:
        print 'Gathering stats for directory ' + sysOps.globaldatapath + dir
        [all_reads, plur_nt_counts, all_identical, all_amp_matches] = fileOps.gather_lib_stats(dir + '/' ,raw_uxi_files=['_for_uxi0.fasta','_for_uxi1.fasta','_rev_uxi0.fasta'],amp_match_file = '_amp_match.txt')
        thisline = ','.join([dir , ','.join([str(x) for x in all_reads]), ','.join([str(x) for x in plur_nt_counts]), ','.join([str(x) for x in all_identical]), ','.join([str(x) for x in all_amp_matches])])
        outfile.write(thisline + '\n')
    outfile.close()
            
    return

def gather_cluster_stats(conditions_filename = 'conditions.csv', outfilename = 'cluster_stats_file.txt'):
    outfile = open(sysOps.globaldatapath +outfilename,'w')
    rank_outfile = open(sysOps.globaldatapath + 'rank_' + outfilename,'w')
    tot_amplicons = 3
    
    #use conditions conditions_filename to specify output order
    dirnames = list()
    with open(sysOps.globaldatapath + conditions_filename, 'rU') as conditions_handle:
        for myline in conditions_handle:
            thisline = myline.strip('\n').split(',')
            dirnames.append('lib_' + str(thisline[0]) + '_' + str(thisline[1]) + '_' + str(thisline[2]))
            
    for dir in dirnames:
        print 'Gathering cluster stats for directory ' + dir
        clusters = [[0,0,0],[0,0,0],[0,0,0]]
        rank100 = [list([0]*100) for i in range(tot_amplicons)]
        with open(sysOps.globaldatapath +dir + '/trg_amplicon_calls.csv','rU') as trg_amp_handle:
            for my_line in trg_amp_handle:
                thisline = [int(x) for x in my_line.strip('\n').split(',')]
                for i in range(100):
                    if rank100[thisline[1]][i]<thisline[3]:
                        rank100[thisline[1]].insert(i,thisline[3])
                        rank100[thisline[1]] = rank100[thisline[1]][:100]
                        break
                    
                if thisline[3] >= 3:
                    clusters[0][thisline[1]] += 1
                    clusters[1][thisline[1]] += 1
                    clusters[2][thisline[1]] += 1
                elif thisline[3] >= 2:
                    clusters[0][thisline[1]] += 1
                    clusters[1][thisline[1]] += 1
                else:
                    clusters[0][thisline[1]] += 1
                    
        thisline = ','.join([dir, ','.join([str(x) for x in clusters[0]]), ','.join([str(x) for x in clusters[1]]), ','.join([str(x) for x in clusters[2]])])
        thisline_rank = ','.join([dir, ','.join([str(x) for x in rank100[0]]), ','.join([str(x) for x in rank100[1]]), ','.join([str(x) for x in rank100[2]])])

        #format of output: 1-read amplicon stats (amplicon 0, amplicon 1, amplicon 2), 1-read amplicon stats (amplicon 0, amplicon 1, amplicon 2), 
        outfile.write(thisline + '\n')
        rank_outfile.write(thisline_rank + '\n')

    outfile.close()
    rank_outfile.close()
    return
