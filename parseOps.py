import sysOps
import alignOps
import itertools
import numpy as np
from Bio import Seq
from Bio import SeqIO

# parseOps.py contains a library of parsing functions for the specific outputs of the amplicon pipeline

def ambig_seq_to_np(seq,seq_bool_vec,capital_bool_vec,ambiguous_vec):
    base_order = 'ACGT'
    for i in range(len(seq)):
        for j in range(4):
            seq_bool_vec[(i*4)+j] = alignOps.has_ambiguity(seq[i].upper(), base_order[j])
            capital_bool_vec[(i*4)+j] = seq[i].isupper()
        ambiguous_vec[i] = (seq[i].upper() not in base_order) # True if ambiguous base

def seq_to_np(seq,buffer_vec,mylen):
    # assumes seq is upper-case with length mylen
    base_order = 'ACGT'
    buffer_vec[:] = np.False_
    for i in range(mylen):
        if seq[i] in base_order:
            buffer_vec[(i*4)+base_order.index(seq[i])] = np.True_
            
def np_to_seq(buffer_vec,binary_vec,list_buff):
    
    str_end = 0
    base_order = 'ACGT'
    base_indices = np.arange(4)
    for i in np.where(binary_vec)[0]:
        base_index = base_indices[buffer_vec[(i*4):((i+1)*4)]]
        if base_index.shape[0] == 1:
            list_buff[str_end] = base_order[base_index[0]]
        else:
            list_buff[str_end] = 'N'
        str_end += 1

    return ''.join(list_buff[:str_end])

def parse_seqform(parseable,amplicon_option = None):
    '''
    parse input from -seqform_for or -seqform_rev tag in settings file
    parseable must contain integers separated by '|' characters, X_position1:position2
    X is one of the following characters
    1. P -- primer
    2. S -- spacer
    3. A -- amplicon
    4. U -- uxi
    X's may be redundant (there may be multiple primers, spacers, and amplicons)
    If form is X_N_position1:position2 (with a string between 2 underscores), N represents a sequence to which the input is aligned and match-score stored (N's in case of uxi)
    Final form of returned my_seqform dictionary entry is:
    Character1: [[[positionA1,positionA2],filter-sequence A (="" if none given)],[[positionB1,positionB2],filter-sequence B (="" if none given)]]
    '''
    my_seqform = dict()
    parseable = parseable.split("|")
    for this_parseable in parseable:
        my_elements = this_parseable.split("_")
        try:
            if(len(my_elements) < 3):
                my_char = my_elements[0].upper()
                seq = ""
                boundaries = my_elements[1].split(":")
            else:
                my_char = my_elements[0].upper()
                seq = my_elements[1]
                boundaries = my_elements[2].split(":")
                
            if(len(boundaries[0])==0):
                boundaries = [None, int(boundaries[1])]
            elif(len(boundaries[1])==0):
                boundaries = [int(boundaries[0]), None]
            else:
                boundaries = [int(boundaries[0]),int(boundaries[1])]
                if(boundaries[1]-boundaries[0] != len(seq) and len(my_elements)==3):
                    sysOps.throw_exception('Error: mismatch between filter boundary-indices and filter string-size, boundaries=' + str(boundaries) + ", seq=" + seq)
                
        except:
            print "Error parsing seqform " + this_parseable
            sysOps.throw_exception(["Error parsing seqform " + this_parseable])
        
            
        if my_char not in "PSAU":
            sysOps.throw_status(["Ignoring this_parseable=" + this_parseable + " -- unrecognized character-type."])
        else:
            if my_char == "A" and type(amplicon_option) == str and type(boundaries[1]) != int:
                start_pos = int(boundaries[0])
                for sub_seq in amplicon_option.split(','):
                    len_sub_seq = len(sub_seq)
                    seq_bool_vec = np.zeros(4*len_sub_seq,dtype=np.bool_)
                    capital_bool_vec = np.zeros(4*len_sub_seq,dtype=np.bool_)
                    ambig_vec = np.zeros(len_sub_seq,dtype=np.bool_)
                    ambig_seq_to_np(sub_seq, seq_bool_vec, capital_bool_vec, ambig_vec)
                    if my_char in my_seqform:
                        my_seqform[my_char].append([[start_pos,start_pos + len_sub_seq], seq_bool_vec[:],capital_bool_vec, ambig_vec])
                    else:
                        my_seqform[my_char] = [[[start_pos,start_pos + len_sub_seq], seq_bool_vec,capital_bool_vec, ambig_vec]]       
                    start_pos += len_sub_seq
                # since original type(boundaries[1]) != int, re-set final boundaries[1] = None
                my_seqform[my_char][len(my_seqform[my_char])-1][0][1] = None
            else:
                seq_bool_vec = np.zeros(4*len(seq),dtype=np.bool_)
                capital_bool_vec = np.zeros(4*len(seq),dtype=np.bool_)
                ambig_vec = np.zeros(len(seq),dtype=np.bool_)
                ambig_seq_to_np(seq, seq_bool_vec, capital_bool_vec, ambig_vec)
                if my_char in my_seqform:
                    my_seqform[my_char].append([boundaries, seq_bool_vec,capital_bool_vec, ambig_vec])
                else:
                    my_seqform[my_char] = [[boundaries, seq_bool_vec,capital_bool_vec, ambig_vec]]                      
    
    return my_seqform
