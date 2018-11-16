import os
import libOps
import sysOps
import alignOps
import dnamicOps
import hashAlignments
import parallelOps
import time
import matOps
import optimOps
import parseOps
import itertools
import numpy as np
from Bio import SeqIO

# This code functions as the main control line for sub-routines

class masterProcess:
    def __init__(self, my_tasks = list()):
            
        self.task_log_file = None
        self.my_tasks = my_tasks
        self.my_starttime = time.time()
        
    def generate_uxi_library(self):
        # Perform sequence analysis (read-parsing, clustering, pairing UEIs/UMIs, sub-sampling data for rarefaction analyses)
        
        if not sysOps.check_file_exists('uxi_lib_tasklist.csv'):
            # create task list for library processing
            [subdirnames, filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath)
            with open(sysOps.globaldatapath + 'uxi_lib_tasklist.csv','w') as task_input_file_handle:
                for subdir in subdirnames:
                    if sysOps.check_file_exists(subdir + '//libsettings.txt'):
                        task_input_file_handle.write('generate_uxi_library;' + sysOps.globaldatapath + subdir + '//\n')
                            
        original_datapath = str(sysOps.globaldatapath)
        [my_task,time_start] = parallelOps.get_next_open_task('tasklog.csv', 'uxi_lib_tasklist.csv', 'generate_uxi_library')
        if not (my_task is None):

            sysOps.initiate_runpath(str(my_task[1]))
            myLibObj = libOps.libObj(settingsfilename = 'libsettings.txt', output_prefix = '_')
            if not sysOps.check_file_exists(myLibObj.output_prefix + 'lib_stats.txt'):
                myLibObj.partition_fastq_library(discarded_sequence_path = "discarded_sequences.fastq", mean_phred_score_path = "mean_phred_scores.txt")
            self.generate_cluster_analysis()
                
            libOps.subsample(myLibObj.seqform_for_params,myLibObj.seqform_rev_params, myLibObj.output_prefix)
            [subdirnames, filenames] = sysOps.get_directory_and_file_list()
            dirnames = list([subdirname for subdirname in subdirnames if subdirname.startswith('sub')])
            sysOps.throw_status('Performing cluster analysis on sub-directories: ' + str(dirnames))
            for dirname in dirnames:
                sysOps.initiate_runpath(str(my_task[1]) + dirname + '//')
                self.generate_cluster_analysis()
        
            sysOps.globaldatapath = str(original_datapath)   
            if not parallelOps.close_task('tasklog.csv', ';'.join(my_task), time_start):
                sysOps.throw_exception('Task ' + str(my_task) + ' no longer exists in log ' + sysOps.globaldatapath + 'tasklog.csv' + ' -- exiting.')
                sysOps.exitProgram()
        
    def dnamic_inference(self, smle_infer = False, msmle_infer = False, segment_infer = False, compute_local_solutions_only = True):
        # Perform image inference on the basis of raw output of DNA microscopy sequence analysis
        
        # Basic settings
        read_thresh = 2
        min_uei_count = 2
        output_dim = 2
        version = 1.0
        infer_dir = ''
        
        # raw data files
        consensus_pairing_csv_file = "..//consensus_" + str(read_thresh) + "r_pairing_filter0.75_uei_umi.csv"
        outname = 'minuei' + str(min_uei_count) + 'DMv' + str(version) + '_' + str(read_thresh) + 'r_filter0.75'
        wmat_outfilename = 'wmat_' + outname + '.csv'
        param_name = 'minuei' + str(min_uei_count) + 'dim' + str(output_dim) + 'DMv' + str(version) + '_.csv' 
        imagemodule_input_filename = 'data_' + param_name
        key_filename = 'key_' + param_name
        if not sysOps.check_file_exists('microscopy_tasklist.csv'):
            [subdirnames, filenames] = sysOps.get_directory_and_file_list(sysOps.globaldatapath)
            with open(sysOps.globaldatapath + 'microscopy_tasklist.csv','w') as task_input_file_handle:
                for subdir in subdirnames:
                    if sysOps.check_file_exists(subdir + '//libsettings.txt'):
                        task_input_file_handle.write('infer_smle;' + sysOps.globaldatapath + subdir + '//\n')
                        task_input_file_handle.write('infer_msmle;' + sysOps.globaldatapath + subdir + '//\n')
                        task_input_file_handle.write('infer_segment;' + sysOps.globaldatapath + subdir + '//\n')
                        task_input_file_handle.write('infer_ptmle;' + sysOps.globaldatapath + subdir + '//\n')
                        
        original_datapath = str(sysOps.globaldatapath)
        if smle_infer:
            infer_dir = 'infer_smle//'
            [my_task,time_start] = parallelOps.get_next_open_task('tasklog.csv', 'microscopy_tasklist.csv', 'infer_smle')
        elif msmle_infer:
            infer_dir = 'infer_msmle//'
            [my_task,time_start] = parallelOps.get_next_open_task('tasklog.csv', 'microscopy_tasklist.csv', 'infer_msmle')
        elif segment_infer:
            infer_dir = 'infer_segment//'
            [my_task,time_start] = parallelOps.get_next_open_task('tasklog.csv', 'microscopy_tasklist.csv', 'infer_segment')
        else:
            infer_dir = 'infer_ptmle//'
            [my_task,time_start] = parallelOps.get_next_open_task('tasklog.csv', 'microscopy_tasklist.csv', 'infer_ptmle')
                            
        if not (my_task is None):

            sysOps.initiate_runpath(str(my_task[1]))
            
            [subdirnames, filenames] = sysOps.get_directory_and_file_list()
            dirnames = list(["."])
            subdirnames_nodatayet = [subdirname for subdirname in subdirnames if subdirname.startswith('sub') and (not sysOps.check_file_exists(subdirname + '//' + imagemodule_input_filename))]
            subdirnames_nodatayet = [subdirnames_nodatayet[i] for i in np.argsort(-np.array([int(subdirname[3:].strip('/')) for subdirname in subdirnames_nodatayet]))] # sort by descending read count
            subdirnames_dataalready = [subdirname for subdirname in subdirnames if subdirname.startswith('sub') and (sysOps.check_file_exists(subdirname + '//' + imagemodule_input_filename))]
            subdirnames_dataalready = [subdirnames_dataalready[i] for i in np.argsort(-np.array([int(subdirname[3:].strip('/')) for subdirname in subdirnames_dataalready]))] # sort by descending read count
            dirnames.extend(subdirnames_nodatayet)
            dirnames.extend(subdirnames_dataalready)
            sysOps.throw_status('Checking directories ' + sysOps.globaldatapath + ' ... ' + str(dirnames) + ' for infer-subdirectories.')
            for dirname in dirnames: # make inference directories
                try:
                    with open(sysOps.globaldatapath + dirname + '//' + infer_dir + 'tmpfile.txt','w') as tmpfile:
                        tmpfile.write('test')
                    os.remove(sysOps.globaldatapath + dirname + '//' + infer_dir + 'tmpfile.txt')
                    sysOps.throw_status('Directory ' + sysOps.globaldatapath + dirname + '//' + infer_dir + ' found already created.')
                except:
                    os.mkdir(sysOps.globaldatapath + dirname + '//' + infer_dir)
                    sysOps.throw_status('Created directory ' + sysOps.globaldatapath + dirname + '//' + infer_dir)
                
            for dirname in dirnames:
                sysOps.initiate_runpath(str(my_task[1]) + dirname + '//' + infer_dir)
                sysOps.initiate_statusfilename()
                sysOps.throw_status('Assigned path ' + sysOps.globaldatapath)
                        
                if not(sysOps.check_file_exists(key_filename) 
                       and sysOps.check_file_exists(imagemodule_input_filename)
                       and sysOps.check_file_exists('read_' + imagemodule_input_filename)
                       and sysOps.check_file_exists('seq_params_' + imagemodule_input_filename)):
                        
                    sysOps.throw_status('Calling matOps.generate_wmat()')
                    
                    trg_dict = matOps.generate_wmat(consensus_pairing_csv_file, read_thresh, min_uei_count, wmat_outfilename)
                    sysOps.throw_status('Completed matOps.generate_wmat()')
                    matOps.print_imagemodule_input(trg_dict,imagemodule_input_filename,key_filename,output_dim)
                    #print_imagemodule_input outputs            
                    #    1. File key_filename containing 3 columns: 0 or 1 (for beacon or target, respectively), cluster-index, MLE processing index
                    #    2. imagemodule_input_filename containing 3 columns: MLE processing index for beacon, MLE processing index for target, uei-count, max UEI read count
                    #    3. Summary file containing: Number of beacons inputted to MLE, number of targets inputted to MLE, 
                else:
                    sysOps.throw_status('Image-module input pre-computed. Proceeding ...')
                
                #optimOps.test_ffgt()
                
                if sysOps.check_file_exists(imagemodule_input_filename):
                    if segment_infer:
                        optimOps.run_mle(imagemodule_input_filename,False,False,True,compute_local_solutions_only,) # segmentation only
                    elif msmle_infer:
                        optimOps.run_mle(imagemodule_input_filename,False,True,False,compute_local_solutions_only) # msMLE
                    elif smle_infer:
                        optimOps.run_mle(imagemodule_input_filename,True,False,False,compute_local_solutions_only) # sMLE
                    else:
                        optimOps.run_mle(imagemodule_input_filename,False,False,False,compute_local_solutions_only) # ptMLE
                    
                    if not compute_local_solutions_only:
                        dnamicOps.print_final_results('..//trg_amplicon_calls.csv', '..//trg_amplicon_calls.fasta')
                    else:
                        sysOps.exitProgram()
                else:
                    sysOps.throw_status('Could not locate ' + sysOps.globaldatapath + imagemodule_input_filename)
            
            sysOps.globaldatapath = str(original_datapath)
            if not parallelOps.close_task('tasklog.csv', ';'.join(my_task), time_start):
                sysOps.throw_exception('Task ' + str(my_task) + ' no longer exists in log ' + sysOps.globaldatapath + 'tasklog.csv' + ' -- exiting.')
                sysOps.exitProgram()        
        
        return 
        
    def generate_cluster_analysis(self):
        # Perform clustering analysis of UMI and UEI sequences, consolidate pairings and determine consenses of these pairings
        
        sysOps.initiate_statusfilename()
        missing_uxi_files = sysOps.find_missing_uxi_files('libsettings.txt', '_')
        if len(missing_uxi_files)>0:
            sysOps.throw_exception('Missing uxi files: ' + str(missing_uxi_files))
        
        if(sysOps.check_file_exists('_for_uxi0.fasta')):
            sysOps.throw_status("Clustering for_uxi0")
            clustering_up_to_date_1 = hashAlignments.initiate_hash_alignment('_for_uxi0.fasta')
        else:
            clustering_up_to_date_1 = True
            sysOps.throw_status(sysOps.globaldatapath + '_for_uxi0.fasta does not exist. Skipping.')
        
        if(sysOps.check_file_exists('_for_uxi1.fasta')):
            sysOps.throw_status("Clustering for_uxi1")
            clustering_up_to_date_2 = hashAlignments.initiate_hash_alignment('_for_uxi1.fasta')
        else:
            clustering_up_to_date_2 = True
            sysOps.throw_status(sysOps.globaldatapath + '_for_uxi1.fasta does not exist. Skipping.')
            
        if(sysOps.check_file_exists('_rev_uxi0.fasta')):
            sysOps.throw_status("Clustering rev_uxi0")
            clustering_up_to_date_3 = hashAlignments.initiate_hash_alignment('_rev_uxi0.fasta')
        else:
            clustering_up_to_date_3 = True
            sysOps.throw_status(sysOps.globaldatapath + '_rev_uxi0.fasta does not exist. Skipping.')
            
        if (clustering_up_to_date_1 and clustering_up_to_date_2 and clustering_up_to_date_3):
            
            filter_val = 0.75 #maximum fraction of same-base permitted in a single UMI/UEI
            min_pairing_readcount = 2
            sysOps.throw_status('Clustering completed. Beginning final output.')
            
            if (sysOps.check_file_exists('thresh1_identical__for_uxi0.fasta') and sysOps.check_file_exists('thresh1_identical__for_uxi1.fasta') and sysOps.check_file_exists('thresh1_identical__rev_uxi0.fasta') 
                and not (sysOps.check_file_exists('consensus_pairing_filter' + str(filter_val) + '_uei_umi.csv'))):
                if not sysOps.check_file_exists("pairing_filter" + str(filter_val) + "_uei_umi.csv"):
                    dnamicOps.assign_umi_pairs('thresh1_identical__for_uxi1.fasta','thresh1_identical__for_uxi0.fasta','thresh1_identical__rev_uxi0.fasta', 
                                                 '_for_uxi1.fasta' , '_for_uxi0.fasta', '_rev_uxi0.fasta', 
                                                 'pairing',filter_val,False) # final parameter = False: excluding invalid amplicon sequences

                dnamicOps.assign_consensus_pairs("pairing_filter" + str(filter_val) + "_uei_umi.csv",min_pairing_readcount)
            else:
                sysOps.throw_status('Consensus-pairing file found pre-computed.')
                
            if (sysOps.check_file_exists('thresh1_identical__rev_uxi0.fasta') and not sysOps.check_file_exists('trg_amplicon_calls.csv')):
                #assign amplicon-identities to target umi's
                sysOps.throw_status('Assigning amplicon-identities and consensus sequences to target umis.')
                dnamicOps.assign_umi_amplicons('thresh1_identical__rev_uxi0.fasta','_rev_uxi0.fasta','_amp_match.txt', '_rev_amp0.fasta', 'trg_amplicon_calls.csv') 
                
    def crosscomparison_analysis(self, args):
        
        sysOps.initiate_statusfilename()
        list_of_dirs = list()
        
        file_to_compare = args[1]
        
        with open(sysOps.globaldatapath + args[2],'rU') as csvfile:
            for myline in csvfile:
                thisline = myline.strip('\n').split(',')  
                subdir = 'lib_' + str(thisline[0]) + '_' + str(thisline[1]) + '_' + str(thisline[2])
                list_of_dirs.append(subdir)
        
        print "Beginning comparison analysis"
        print "File to compare = " + file_to_compare
        print "Directories = " + ",".join(list_of_dirs)

        try:
            os.mkdir(sysOps.globaldatapath + 'cross_comparisons')
        except:
            sysOps.throw_exception('cross_comparisons directory already exists. Terminating comparison analysis.')
            sysOps.exitProgram()
        
        shared_num_unique_matrix = list()
        unshared_num_unique_matrix = list()
        shared_read_abund_matrix = list()
        unshared_read_abund_matrix = list()
        
        for i in range(len(list_of_dirs)):
            shared_num_unique_matrix.append(list([-1]*len(list_of_dirs)))
            unshared_num_unique_matrix.append(list([-1]*len(list_of_dirs)))
            shared_read_abund_matrix.append(list([-1]*len(list_of_dirs)))
            unshared_read_abund_matrix.append(list([-1]*len(list_of_dirs)))
                
        for ind1 in range(len(list_of_dirs)):
            for ind2 in range(ind1):
                dir1 = list_of_dirs[ind1]
                dir2 = list_of_dirs[ind2]
                clustfile1 = dir1 + "//" + file_to_compare
                clustfile2 = dir2 + "//" + file_to_compare
                dir1_abbrev = dir1[(dir1.rfind('/')+1):] #remove superdirectory structure of path -- requires individual directories have unique names
                dir2_abbrev = dir2[(dir2.rfind('/')+1):]
                sysOps.throw_status('Began writing cross_comparisons//' + dir1_abbrev + "_" + dir2_abbrev + "_" + file_to_compare)
                [num_unique_shared,num_unique_unshared,read_abundance_shared,read_abundance_unshared] = alignOps.compare(clustfile1, clustfile2, dir1_abbrev + "_" + dir2_abbrev + "_" + file_to_compare, False)
                sysOps.throw_status('Completed writing cross_comparisons//' + dir1_abbrev + "_" + dir2_abbrev + "_" + file_to_compare)
                shared_num_unique_matrix[ind1][ind2] = num_unique_shared[0]
                shared_num_unique_matrix[ind2][ind1] = num_unique_shared[1]
                unshared_num_unique_matrix[ind1][ind2] = num_unique_unshared[0]
                unshared_num_unique_matrix[ind2][ind1] = num_unique_unshared[1]
                print str(num_unique_unshared[0]) + '-> unshared_num_unique_matrix[ ' + str(ind1) + '][' + str(ind2) + ']'
                shared_read_abund_matrix[ind1][ind2] = read_abundance_shared[0]
                shared_read_abund_matrix[ind2][ind1] = read_abundance_shared[1]
                unshared_read_abund_matrix[ind1][ind2] = read_abundance_unshared[0]
                unshared_read_abund_matrix[ind2][ind1] = read_abundance_unshared[1]
                    
        print shared_num_unique_matrix
        print unshared_num_unique_matrix
        print shared_read_abund_matrix
        print unshared_read_abund_matrix
        
        with open('comparison_matrices.csv','w') as compare_matrix_file:
            for i1 in range(len(list_of_dirs)):
                compare_matrix_file.write(','.join([str(j) for j in shared_num_unique_matrix[i1]]) + '\n')
                
            for i2 in range(len(list_of_dirs)):
                compare_matrix_file.write(','.join([str(j) for j in unshared_num_unique_matrix[i2]]) + '\n')
                
            for i3 in range(len(list_of_dirs)):
                compare_matrix_file.write(','.join([str(j) for j in shared_read_abund_matrix[i3]]) + '\n')
                
            for i4 in range(len(list_of_dirs)):
                compare_matrix_file.write(','.join([str(j) for j in unshared_read_abund_matrix[i4]]) + '\n')
        
        
        
        
        