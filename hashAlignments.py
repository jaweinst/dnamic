import sysOps
import os
import alignOps
import clustOps

def output_hashed_mismatch_alignment(identical_uxi_file,mismatch_pos, outputfilename):
    # Function will generate and fill in a dictionary of UXI-sequences when position mismatch_pos is IGNORED
    # This will consolidate all sequences identical up to 1 nucleotide (in ungapped alignment)
    hash_dict = dict()
    with open(sysOps.globaldatapath + identical_uxi_file ,'rU') as infile:
        for line in infile:
            line_list = line.strip('\n').split('_')
            if(len(line_list) == 3):
                [uxi_seq,uxi_ind,uxi_readnum] = line_list
                uxi_substr = uxi_seq[:mismatch_pos] + uxi_seq[(mismatch_pos+1):]
                if uxi_substr in hash_dict:
                    hash_dict[uxi_substr].append(uxi_ind + '_' + uxi_readnum)
                else:
                    hash_dict[uxi_substr] = [uxi_ind + '_' + uxi_readnum]
                    
    with open(sysOps.globaldatapath + outputfilename ,'w') as outfile:
        for uxi_substr in hash_dict:
            outfile.write(uxi_substr + ':' + ','.join(hash_dict[uxi_substr]) + '\n')
    
def generate_linkage_file(identical_uxi_file, hash_dict_file_list, linked_filename, P = 0.0):
    # Function will generate DIRECTIONAL linkages between UXIs
    # A UXI will directionally link to another UXI if and only if it has at least the other UXI's read-abundance (+ factor P, default set to 0)
    upper_multiplier = 1 + P
    #load identical_uxi_file into dictionary using INDICES
    identical_uxi_dict_by_index = dict()
    max_index = 0
    with open(sysOps.globaldatapath + identical_uxi_file ,'rU') as infile:
        for line in infile:
            line_list = line.strip('\n').split('_')
            if(len(line_list) == 3):
                [uxi_seq,uxi_ind,uxi_readnum] = line_list
                identical_uxi_dict_by_index[uxi_ind] = [[uxi_seq, str(uxi_readnum), int(uxi_readnum)],[str(uxi_ind)]]
                max_index = max(max_index, int(uxi_ind))
    
    for hash_dict_filename in hash_dict_file_list:
        with open(sysOps.globaldatapath + hash_dict_filename ,'rU') as infile:
            for line in infile:
                [uxi_substr, uxi_inds_readnums] = line.strip('\n').split(':')
                uxi_inds_readnums = uxi_inds_readnums.split(',')
                len_uxi_list = len(uxi_inds_readnums)
                for i in range(len_uxi_list):
                    [uxi_ind_i,uxi_readnum_i] = uxi_inds_readnums[i].split('_')
                    uxi_readnum_i = int(uxi_readnum_i)
                    for j in range(i):
                        [uxi_ind_j,uxi_readnum_j] = uxi_inds_readnums[j].split('_')
                        uxi_readnum_j = int(uxi_readnum_j)
                        if uxi_readnum_i*upper_multiplier >= uxi_readnum_j:
                            identical_uxi_dict_by_index[uxi_ind_i][0][2] += uxi_readnum_j # add to RND
                            identical_uxi_dict_by_index[uxi_ind_i][1].append(uxi_ind_j)
                        if uxi_readnum_j*upper_multiplier >= uxi_readnum_i:
                            identical_uxi_dict_by_index[uxi_ind_j][0][2] += uxi_readnum_i # add to RND
                            identical_uxi_dict_by_index[uxi_ind_j][1].append(uxi_ind_i)
                            
    with open(sysOps.globaldatapath + linked_filename ,'w') as outfile:
        for uxi_ind in range(max_index+1):
            #print out dictionary in numerically ascending order of index (this sorting is critical for downstream processing)
            uxi_ind_str = str(uxi_ind)
            if uxi_ind_str in identical_uxi_dict_by_index:
                my_dict_el = identical_uxi_dict_by_index[uxi_ind_str]
                my_dict_el[0][2] = str(my_dict_el[0][2]) #convert all RNDs back to strings
                outfile.write(','.join(my_dict_el[0]) + ':' + ','.join(my_dict_el[1]) + '\n')


def initiate_hash_alignment(uxi_file, P = 0.0):
    '''
    Takes in specific uxi_file, already formatted from source, consolidates identical sequences, performs hash-alignment, 
    and clusters them. Each of these tasks is skipped, in order, if it's found up-to-date based on dates-of-modification. 
    
    '''
        
    identical_uxi_file = 'identical_' + uxi_file
        
    consolidation_up_to_date = False
    clustering_up_to_date = False
    alignment_up_to_date = False
    
    [dirnames,filenames] = sysOps.get_directory_and_file_list()
    
    if identical_uxi_file in filenames: 
        consolidation_up_to_date = (os.stat(sysOps.globaldatapath + identical_uxi_file).st_mtime > os.stat(sysOps.globaldatapath + uxi_file).st_mtime)
        #if time of last modification of identical-consolidation file is later than time of modification/writing of uxi_file
        sysOps.throw_status(['Consolidation up-to-date = ' + str(consolidation_up_to_date), sysOps.statuslogfilename])
    
    if ('linked_' + identical_uxi_file) in filenames:
        alignment_up_to_date = (os.stat(sysOps.globaldatapath + 'linked_' + identical_uxi_file).st_mtime > os.stat(sysOps.globaldatapath + identical_uxi_file).st_mtime)
        #if time of last modification of threshold-clustering file is later than time of modification/writing of uxi_file
        sysOps.throw_status(['Alignment up-to-date = ' + str(alignment_up_to_date), sysOps.statuslogfilename])
            
    if ('thresh1_' + identical_uxi_file) in filenames:
        clustering_up_to_date = (os.stat(sysOps.globaldatapath + 'thresh1_' + identical_uxi_file).st_mtime > os.stat(sysOps.globaldatapath + identical_uxi_file).st_mtime)
        #if time of last modification of threshold-clustering file is later than time of modification/writing of uxi_file
        sysOps.throw_status(['Clustering up-to-date = ' + str(clustering_up_to_date), sysOps.statuslogfilename])
   
    if not (consolidation_up_to_date and alignment_up_to_date):
        
        #write placeholder file
        with open(sysOps.globaldatapath + 'thresh1_' + identical_uxi_file, 'w') as placeholderfile:
            placeholderfile.write('In progress.')
            
        if not consolidation_up_to_date:
            sysOps.throw_status(['Consolidation not up to date, consolidating file '  + sysOps.globaldatapath + uxi_file, sysOps.statuslogfilename])
            [num_elements,uxi_len] = alignOps.consolidate_uxi(uxi_file, start_index = 0, prefix = '', include_inv_amp = False)
        else: #fetch uxi_len
            sysOps.throw_status(['Consolidation up to date, reading from file ' + sysOps.globaldatapath + identical_uxi_file, sysOps.statuslogfilename])
            with open(sysOps.globaldatapath +identical_uxi_file,'rU') as uxi_handle:
                for uxi_line in uxi_handle:
                    split_str = uxi_line.split('_')
                    if(len(split_str)==3):
                        uxi_len = len(split_str[0]) #first element of identical-sequence file is U(M/E)I sequence itself
                        break
        
        for mismatch_pos in range(uxi_len): #output members (indexed by ) of substrings (and abundances) corresponding to all characters except for the one at mismatch_pos
                                            #format as follows -- substring: member1-index_abundance1,member2-index_abundance2,...
            sysOps.throw_status(['Performing hash alignment on position ' + str(mismatch_pos), sysOps.statuslogfilename])
            output_hashed_mismatch_alignment(identical_uxi_file,mismatch_pos,'mis' + str(mismatch_pos) + '_' + identical_uxi_file)
        
        sysOps.throw_status(['Hash alignments complete. Proceeding to assemble linked file.', sysOps.statuslogfilename])
        generate_linkage_file(identical_uxi_file, ['mis' + str(mismatch_pos) + '_' + identical_uxi_file for mismatch_pos in range(uxi_len)],"linked_" + identical_uxi_file,P)
            
        #now that linkage file has been constructed, delete hash-alignment files
        for hash_filename in ['mis' + str(mismatch_pos) + '_' + identical_uxi_file for mismatch_pos in range(uxi_len)]:
            os.remove(sysOps.globaldatapath + hash_filename)
        
    if not clustering_up_to_date:
        sysOps.delay_with_alertfile('_cluster_inprog' + uxi_file)
        clustOps.threshold_cluster_uxi_prelinked(alignOps.load_linkage_file_to_list("linked_" + identical_uxi_file), identical_uxi_file ,1, P)
        clustering_up_to_date = True
        sysOps.remove_alertfile('_cluster_inprog' + uxi_file)
           
                    
    return clustering_up_to_date
