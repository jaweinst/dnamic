import sysOps
import os
import numpy as np
import itertools
from Bio import SeqIO
from string import printable

def get_revcomp(str_seq):
    # simple reverse complement function for string
    revcomp = ''
    in_base =  'ACGTNWSacgtnws'
    out_base = 'TGCANWStgcanws'
    for i in range(len(str_seq)-1,-1,-1):
        revcomp += out_base[in_base.find(str_seq[i])]
    return revcomp

def rigid_conjoin(str_seq1,str_seq2,min_overlap):
    # requires str_seq1 and str_seq2 to be conjoined IN THAT ORDER
    # returns length of overlap
    len1 = len(str_seq1)
    len2 = len(str_seq2)
    lower_str_seq1 = str(str_seq1.lower())
    lower_str_seq2 = str(str_seq2.lower())
    for overlap in range(min_overlap,len2):
        if lower_str_seq2[:overlap] == lower_str_seq1[(len1-overlap):len1]:
            return len1-overlap, str_seq1[:(len1-overlap)] + str_seq2
    
    return -1, ''
    
def rm_hidden_char(mystr):
    return ''.join(mychar for mychar in mystr if mychar in printable)
    
def generate_data_layout(data_layout_file = 'data_layout.csv'): 
    # Format of data_layout_file as follows:
    # Sample     Sample name, description, etc
    # Barcode    Number/search-term for run directories
    # Run        Run directory-1
    # Run        Run directory-2
    # Run        etc
    # Beacon     Beacon oligo-1
    # Beacon     Beacon oligo-2 
    # Beacon     etc
    # Target     Target oligo-1
    # Target     Target oligo-2 
    # Target     etc
    # OE1a       OE-primer-1
    # OE4b       OE-primer-2
    # Amplicon   Amplicon file-1
    # Amplicon   Amplicon file-2
    # Standardize-amplicon-start    TRUE/left blank
    
    data_layout_dict = dict()
    with open(data_layout_file,'rU') as csvfile:
        curr_sample = None
        for myline in csvfile:
            thisline = rm_hidden_char(myline).strip('\n').split(',')
            if len(thisline) >= 2:
                if thisline[0].lower() == 'sample':
                    curr_sample = thisline[1]
                    data_layout_dict[curr_sample] = dict()
                else:
                    if thisline[0].lower() not in data_layout_dict[curr_sample]:
                        data_layout_dict[curr_sample][thisline[0].lower()] = list()
                    data_layout_dict[curr_sample][thisline[0].lower()].append(thisline[1])

    final10_sbs12_sbs3 = 'CTTCCGATCT'     
    for sample in data_layout_dict:
        missing_keys = [my_key for my_key in ['barcode','run','beacon','target','oe1a','oe4b','amplicon'] if my_key not in data_layout_dict[sample]]
        if (len(missing_keys)>0):
            sysOps.throw_status('Skipping sample ' + str(sample) + ' due to missing keys:' + str(missing_keys))
        else:
            source_for = list()
            source_rev = list()
            for run_index in range(len(data_layout_dict[sample]['run'])):
                run_dir = data_layout_dict[sample]['run'][run_index]
                if not run_dir.endswith('//'):
                    run_dir += '//'
                run_dir_exists = False
                try: # try opening run_dir for writing
                    with open(run_dir + 'test.txt','w'):
                        run_dir_exists = True
                    os.remove(run_dir + 'test.txt')
                except:
                    sysOps.throw_status('Skipping run-directory ' + str(run_dir))
                if run_dir_exists:
                    [subdirnames, filenames] = sysOps.get_directory_and_file_list(run_dir)
                    this_sample_run_R1 = list(['..//' + run_dir + filename 
                                               for filename in filenames if (data_layout_dict[sample]['barcode'][0]+'_' in filename
                                                                             and 'R1' in filename)])
                    this_sample_run_R2 = list([filename[:(filename.find('R1'))] 
                                               + 'R2' 
                                               +  filename[(filename.find('R1')+2):] for filename in this_sample_run_R1])
                    source_for.extend(this_sample_run_R1) # since new directory is being created, adding an additional level to the path
                    source_rev.extend(this_sample_run_R2)
            source_for =  ','.join(source_for)
            source_rev =  ','.join(source_rev)  
            # join oe sequences
            seqform_for = list()
            find_index, conjoined_oe_seq = rigid_conjoin(get_revcomp(data_layout_dict[sample]['oe1a'][0]),data_layout_dict[sample]['oe4b'][0],10)

            for beacon_oligo_index in range(len(data_layout_dict[sample]['beacon'])):
                revcomp_bcn_oligo = get_revcomp(data_layout_dict[sample]['beacon'][beacon_oligo_index])
                revcomp_bcn_oligo = revcomp_bcn_oligo[(revcomp_bcn_oligo.find(final10_sbs12_sbs3)+len(final10_sbs12_sbs3)):]
                oe_start_index, conjoined_bcn_oe_seq = rigid_conjoin(revcomp_bcn_oligo,conjoined_oe_seq,10)
                
                uei_start_index = np.min(np.array([(oe_start_index+conjoined_bcn_oe_seq[oe_start_index:].upper().find(my_char))
                                                    for my_char in 'NWSRY' if my_char in conjoined_bcn_oe_seq[oe_start_index:].upper()]))
                uei_end_index = 1+np.max(np.array([(oe_start_index+conjoined_bcn_oe_seq[oe_start_index:].upper().rfind(my_char))
                                                    for my_char in 'NWSRY' if my_char in conjoined_bcn_oe_seq[oe_start_index:].upper()]))
                
                my_seqform_for = list()
                my_seqform_for.append('U_' + conjoined_bcn_oe_seq[1:oe_start_index] + '_1:' + str(oe_start_index))
                my_seqform_for.append('P_' + conjoined_bcn_oe_seq[oe_start_index:uei_start_index] + '_' + str(oe_start_index) + ':' + str(uei_start_index))
                my_seqform_for.append('U_' + conjoined_bcn_oe_seq[uei_start_index:uei_end_index] + '_' + str(uei_start_index) + ':' + str(uei_end_index))
                my_seqform_for.append('P_' + conjoined_bcn_oe_seq[uei_end_index:(uei_end_index+2)] + '_' + str(uei_end_index) + ':' + str(uei_end_index+2))
                my_seqform_for = '|'.join(my_seqform_for)
                if my_seqform_for not in seqform_for:
                    seqform_for.append(str(my_seqform_for))
            
            my_amplicons = list()
            for amplicon_file in data_layout_dict[sample]['amplicon']:
                if amplicon_file.upper() == 'N': # amplicon left blank
                    my_amplicons.append(list(['N','N']))
                else:
                    [subdirnames, filenames] = sysOps.get_directory_and_file_list()
                    if amplicon_file in filenames:
                        for record in SeqIO.parse(amplicon_file, "fasta"):
                            my_amplicons.append(list([str(record.id), str(record.seq)]))
                    else:
                        sysOps.throw_status('Skipping ' + str(amplicon_file))
            
            primer_amplicon_pairs = list()
            primer_amplicon_starts = list()
            seqform_rev = list()
            print str(my_amplicons)
            for amplicon in my_amplicons:
                revcomp_amplicon = get_revcomp(amplicon[1].lower())
                for target_oligo_index in range(len(data_layout_dict[sample]['target'])):
                    target_oligo = data_layout_dict[sample]['target'][target_oligo_index]
                    target_oligo = target_oligo[(target_oligo.find(final10_sbs12_sbs3)+len(final10_sbs12_sbs3)):]
                    if revcomp_amplicon == 'n':
                        randprim_len = len(target_oligo) - (1 + np.max(np.array([target_oligo.upper().rfind(my_char) for my_char in 'ACGT'])))
                        target_oligo = target_oligo[:(len(target_oligo) - randprim_len)]
                        my_seqform_rev = list()
                        my_seqform_rev.append('U_' + target_oligo[1:len(target_oligo)] + '_1:' + str(len(target_oligo)))
                        my_seqform_rev.append('A_' + str(len(target_oligo)+randprim_len) + ':')
                        my_seqform_rev = '|'.join(my_seqform_rev)
                        if my_seqform_rev not in seqform_rev:
                            seqform_rev.append(str(my_seqform_rev))
                    else:
                        find_index, conjoined_amplicon_seq = rigid_conjoin(target_oligo,revcomp_amplicon,10)
                        if find_index >= 0:
                            primer_overlap = len(target_oligo) + len(revcomp_amplicon) - len(conjoined_amplicon_seq)
                            primer_amplicon_pairs.append(amplicon[0] + '|'
                                                         + get_revcomp(conjoined_amplicon_seq[(find_index+primer_overlap):]) + ','
                                                         + get_revcomp(conjoined_amplicon_seq[find_index:(find_index+primer_overlap)]))
                            my_seqform_rev = list()
                            my_seqform_rev.append('U_' + conjoined_amplicon_seq[1:find_index] + '_1:' + str(find_index))
                            my_seqform_rev.append('A_' + str(find_index) + ':')
                            primer_amplicon_starts.append(int(find_index))
                            my_seqform_rev = '|'.join(my_seqform_rev)
                            if my_seqform_rev not in seqform_rev:
                                seqform_rev.append(str(my_seqform_rev))
                    
            # finally, print libsettings.txt
            my_libdir = 'lib_' + str(sample) + '//'
            os.mkdir(my_libdir)
                        
            if ('standardize-amplicon-start' in data_layout_dict[sample] 
                and data_layout_dict[sample]['standardize-amplicon-start'][0].lower() == 'true'):
                
                max_amplicon_start = int(np.max(np.array(primer_amplicon_starts)))
                with open(my_libdir + 'amplicon_refs.txt','w') as outfile:
                    new_seqform_rev = list()
                    for my_seqform_rev in seqform_rev:
                        elements = my_seqform_rev.split('|')
                        elements[len(elements)-1] = 'A_' + str(max_amplicon_start) + ':'
                        elements = '|'.join(elements)
                        if elements not in new_seqform_rev:
                            new_seqform_rev.append(str(elements))
                    seqform_rev = list(new_seqform_rev)
                    for primer_amplicon_pair,primer_amplicon_start in itertools.izip(primer_amplicon_pairs,primer_amplicon_starts):
                        outfile.write(primer_amplicon_pair[:(len(primer_amplicon_pair) + primer_amplicon_start - max_amplicon_start)] + '\n')
                primer_amplicon_pairs = list() # omit from libsettings
                
            with open(my_libdir + 'libsettings.txt','w') as outfile:
                outfile.write('-source_for ' + source_for + '\n')
                outfile.write('-source_rev ' + source_rev + '\n')
                for this_seqform_for in seqform_for:
                    outfile.write('-seqform_for ' + this_seqform_for + '\n')
                for this_seqform_rev in seqform_rev:
                    outfile.write('-seqform_rev ' + this_seqform_rev + '\n')
                for primer_amplicon_pair in primer_amplicon_pairs:
                    outfile.write('-amplicon ' + primer_amplicon_pair + '\n')
                if 'max-mismatch' in data_layout_dict[sample]:
                    outfile.write('-max_mismatch ' + data_layout_dict[sample]['max-mismatch'][0] + '\n')
                if 'max-mismatch-amplicon' in data_layout_dict[sample]:
                    outfile.write('-max_mismatch_amplicon ' + data_layout_dict[sample]['max-mismatch-amplicon'][0] + '\n')
                if 'min-mean-qual' in data_layout_dict[sample]:
                    outfile.write('-min_mean_qual ' + data_layout_dict[sample]['min-mean-qual'][0] + '\n')
                if 'filter-amplicon-window' in data_layout_dict[sample]:
                    outfile.write('-filter_amplicon_window ' + data_layout_dict[sample]['filter-amplicon-window'][0] + '\n')
                if 'amplicon-terminate' in data_layout_dict[sample]:
                    for this_amplicon_terminate in data_layout_dict[sample]['amplicon-terminate']:
                        outfile.write('-amplicon_terminate ' + this_amplicon_terminate + '\n')
            
    return

    
    
    
