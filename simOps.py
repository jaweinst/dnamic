import numpy as np
import sysOps
import fileOps
import libOps
import optimOps
from Bio import SeqIO
from Bio import Seq
import os
import itertools
from scipy import misc
from numba import jit, types

@jit("int64(int64[:,:],float64[:,:],float64[:],float64[:],int64,int64,int64,int64,int64,float64,float64,int64)",nopython=True)
def generate_ueis(uei_arr,sim_pos,sorted_unif_rand_res,cumul_rate,Nbcn,Ntrg,Nuei,sim_dims,phys_dims,t_term,tot_rate,uei_ind):
    t_coeff = np.power(t_term,-phys_dims/2.0)
    uei_ind_thiscall = 0
    for i in range(0,Nbcn):
        for j in range(Nbcn,Nbcn+Ntrg):
            sqdist = 0.0
            for d in range(sim_dims):
                sqdist += (sim_pos[i,d] - sim_pos[j,d])*(sim_pos[i,d] - sim_pos[j,d])
            cumul_rate[0] += sim_pos[i,sim_dims]*sim_pos[j,sim_dims]*t_coeff*np.exp(-sqdist/t_term)
                
            while cumul_rate[0]/tot_rate >= sorted_unif_rand_res[uei_ind]:
                uei_arr[uei_ind_thiscall,0] = i
                uei_arr[uei_ind_thiscall,1] = j-Nbcn
                uei_ind+=1
                uei_ind_thiscall+=1
                if uei_ind == Nuei:
                    break # no more UEI's to generate
            if uei_ind >= Nuei:
                break
    return uei_ind
            
@jit("float64(float64[:,:],int64,int64,int64,int64,float64)",nopython=True)
def sum_partition_function(sim_pos,Nbcn,Ntrg,sim_dims,phys_dims,t_term):
    my_sum = 0.0
    t_coeff = np.power(t_term,-phys_dims/2.0)
    for i in range(Nbcn):
        for j in range(Nbcn,Nbcn+Ntrg):
            sqdist = 0.0
            for d in range(sim_dims):
                sqdist += (sim_pos[i,d] - sim_pos[j,d])*(sim_pos[i,d] - sim_pos[j,d])
            my_sum += sim_pos[i,sim_dims]*sim_pos[j,sim_dims]*t_coeff*np.exp(-sqdist/t_term)
    
    return my_sum

class simObj:
    def __init__(self, paramfilename):
    
        #Open parameter-file
    
        sysOps.throw_status("Reading from " + sysOps.globaldatapath + paramfilename + " ...")
        sim_settings = fileOps.read_settingsfile_to_dictionary(sysOps.globaldatapath + paramfilename)
        self.effic_monomer = float(sim_settings['-effic_monomer'][0])
        self.effic_dimer = float(sim_settings['-effic_dimer'][0])
        self.diffconst = float(sim_settings['-diffconst'][0])
        self.lin_cycles = int(sim_settings['-lin_cycles'][0])
        self.exp_cycles = int(sim_settings['-exp_cycles'][0])
        self.posfilename = str(sim_settings['-posfilename'][0])
        # position file contains columns: UMI-index (stored as-is for later), 0 for bcn/1 for trg, x-coordinate, y-coordinate 
        
        raw_image_csv = np.loadtxt(sysOps.globaldatapath + self.posfilename,delimiter=',')
        raw_image_csv = raw_image_csv[np.argsort(raw_image_csv[:,1]),:] # arranged as beacons followed by targets
        self.sim_pos = np.array(raw_image_csv[:,2:],dtype=np.float64)
        self.sim_dims = self.sim_pos.shape[1]
        self.Nbcn = np.sum(raw_image_csv[:,1]==0)
        self.Ntrg = np.sum(raw_image_csv[:,1]==1)
        self.Nuei = int(sim_settings['-uei_per_bcn_umi'][0])*self.Nbcn
        self.N_reads = int(sim_settings['-reads_per_uei'][0])*self.Nuei
        self.index_key = np.int64(raw_image_csv[:,0]) 
        
        self.sim_pos = np.append(self.sim_pos,np.ones([self.Nbcn + self.Ntrg, 1],dtype=np.float64),axis=1) #number of starting molecules is always = 1
        sysOps.throw_status("Assigned point-dimensionality to " + str(self.sim_dims))    
        
        return

    def sim_physics(self):
        sysOps.throw_status("Running DNA microscopy simulation.")
        # bcn-indices must range from 0 to Nbcn-1, trg-indices must range from Nbcn to Nbcn+Ntrg-1
        phys_dims = 3.0
        sysOps.throw_status("Using num_dims = " + str(self.sim_dims) + ", Nbcn = " + str(self.Nbcn) + ", Ntrg = " + str(self.Ntrg))
        tot_rate = 0.0
        # initialize all molecule numbers to 1
        self.sim_pos[:,self.sim_dims] = 1.0
        np.savetxt(sysOps.globaldatapath + "sim_index_key.csv", np.reshape(self.index_key,[self.Nbcn+self.Ntrg,1]), delimiter=',')
        
        sysOps.throw_status("BEGINNING PART 1")
        # PART 1: (a) estimate pairwise reaction rates (and record their sum)
        #         (b) simulate amplification with stochasticity (parameterized by effic_monomer)
        for C in range(1,self.lin_cycles + self.exp_cycles +1):
            # amplify with effic_monomer <=1
            for n in range(self.Nbcn+self.Ntrg):
                if C<=self.lin_cycles:
                    binom_res = np.random.binomial(1,self.effic_monomer) # if linear amplification step, keep template number constant at 1
                else:
                    binom_res = np.random.binomial(int(self.sim_pos[n,self.sim_dims]),self.effic_monomer)
                    # if exponential amplification step, template number varies over time
                self.sim_pos[n,self.sim_dims] += binom_res;
                
            np.savetxt(sysOps.globaldatapath + "molcountfile_" + str(C) + ".csv", 
                       np.reshape(np.int64(self.sim_pos[:,self.sim_dims]),[self.Nbcn+self.Ntrg,1]), delimiter=',')
    
            if C>self.lin_cycles: #only sum/print UEI-formation rates for exponential amplification steps
    
                # iterate through simulated PCR cycles, print relative rates to rate_file
                sysOps.throw_status("C=" + str(C) + " --> 8*D*d*t = " + str(8*self.diffconst*phys_dims*float(C)))
                        # diffconst multiplies time to incorporate that dependence
                t_term = 8*self.diffconst*phys_dims*float(C)
                # that need to be zoomed into so that individual UMI's for concatenation can be extracted
                tot_rate += sum_partition_function(self.sim_pos,self.Nbcn,self.Ntrg,self.sim_dims,phys_dims,t_term)

        sysOps.throw_status("BEGINNING PART 2");
        #PART 2: simulate UEI generation
        sysOps.throw_status("Generating " + str(self.Nuei) + " random numbers.");
        sorted_unif_rand_res = np.sort(np.random.rand(self.Nuei))
        uei_arr = np.zeros([self.Nuei,2],dtype=np.int64)
        uei_ind = 0
        inp_cumul_rate = np.array([0.0],dtype=np.float64)
        
        for C in range(self.lin_cycles+1, self.lin_cycles + self.exp_cycles + 1):
            self.sim_pos[:,self.sim_dims] = np.int64(np.loadtxt(sysOps.globaldatapath + "molcountfile_" + str(C) + ".csv",delimiter=','))
            t_term =8*self.diffconst*phys_dims*float(C)
            uei_arr[:] = -1
            prev_uei_ind = int(uei_ind)
            uei_ind = generate_ueis(uei_arr,self.sim_pos,sorted_unif_rand_res,inp_cumul_rate,self.Nbcn,self.Ntrg,self.Nuei,self.sim_dims,phys_dims,t_term,tot_rate,uei_ind)
            
            if uei_ind > prev_uei_ind:
                np.savetxt(sysOps.globaldatapath + "ueifile_" + str(C) + ".csv", uei_arr[:(uei_ind-prev_uei_ind),:], delimiter=',')
            sysOps.throw_status("C = " + str(C) + ". Current UEI-count: " + str(uei_ind) + '.')

        del sorted_unif_rand_res
    
        sysOps.throw_status("BEGINNING PART 3");
        #PART 3: simulate UEI amplification

        all_uei = np.array([])
        for C in range(self.lin_cycles+1,self.lin_cycles+self.exp_cycles+1):
            for i in range(all_uei.shape[0]):
                all_uei[i,2] += np.random.binomial(all_uei[i,2],self.effic_dimer)
                
            if sysOps.check_file_exists(sysOps.globaldatapath + "ueifile_" + str(C) + ".csv"):
                this_uei_arr = np.int64(np.loadtxt(sysOps.globaldatapath + "ueifile_" + str(C) + ".csv",delimiter=','))
                if len(this_uei_arr.shape) == 1:
                    this_uei_arr = np.array([this_uei_arr])
                if this_uei_arr.shape[0] > 0:
                    this_uei_arr = np.append(this_uei_arr,np.ones([this_uei_arr.shape[0],1]),axis=1)
                    if all_uei.shape[0] == 0:
                        all_uei = np.array(this_uei_arr)
                    else:
                        all_uei = np.concatenate([all_uei,this_uei_arr],axis=0)
    
        sysOps.throw_status("BEGINNING PART 4")
        tot_mol = np.sum(all_uei[:,2])
        #PART 4: output simulated reads
        my_N_reads = self.N_reads
    
        sysOps.throw_status('my_N_reads = ' + str(my_N_reads) + '/' + str(self.N_reads) + ',' + 
                            sysOps.globaldatapath + 'r' + str(my_N_reads) + '_sim_ueifile.csv')
        
        with open(sysOps.globaldatapath + "r" + str(my_N_reads) + "_sim_ueifile.csv",'w') as finalsimdata_outfile:
            sorted_unif_rand_reads = np.sort(np.random.rand(my_N_reads))

            #For downstream processing, need consensus-pairing file with the following comma-separated columns:
            #1. uei index
            #2. beacon-umi index
            #3. target-umi index
            #4. read-count
                
            read_ind = 0
            cumul_read_frac = 0.0
            for uei_index in range(uei_arr.shape[0]):
                cumul_read_frac += all_uei[uei_index,2]/tot_mol
                my_reads = 0
                while cumul_read_frac >= sorted_unif_rand_reads[read_ind]:
                    my_reads += 1
                    read_ind += 1
                    if read_ind == my_N_reads:
                        break #no more reads to generate
                    
                if my_reads > 1:
                    #only include those UEI's with at least 2 reads
                    finalsimdata_outfile.write(str(uei_index) + "," + str(all_uei[uei_index,0]) + "," + str(all_uei[uei_index,1]) + "," + str(my_reads) + '\n')
                
                if read_ind >= my_N_reads:
                    break
                    
        del sorted_unif_rand_reads
        
        my_N_reads *= 2
            
        sysOps.throw_status("SIMULATION COMPLETE")
        
        return

    def sim_reads(self):
        simLibObj = libOps.libObj(settingsfilename = 'libsettings.txt', output_prefix = '_')
        enforced_rev_read_len = 100
        [for_read_len,rev_read_len] = simLibObj.get_min_allowed_readlens(simLibObj.filter_amplicon_window)
        rev_read_len = int(enforced_rev_read_len)
        '''
        simLibObj.seqform_for_params and simLibObj.seqform_rev_params are already stored in current object's memory
        Form of these variables is a list of the following:
            Element 1: [start_pos,end_pos]
            Element 2: np.ndarray(seq_bool_vec, dtype=np.bool_)
            Element 3: np.ndarray(capital_bool_vec, dtype=np.bool_)
            Element 4: np.ndarray(ambig_vec, dtype=np.bool_)
        '''
        [subdirnames, filenames] = sysOps.get_directory_and_file_list()
        
        for_umi_seqs = list()
        rev_umi_seqs = list()
        rev_umi_amplicon_list = list()
        uei_seqs = list()
        base_order = 'ACGT'
        
        sysOps.throw_status('Generating simulated sequences ...')
        amplicon_list = list()
        if "-amplicon" in simLibObj.mySettings:
            amplicon_list = [simLibObj.mySettings["-amplicon"][i].upper().split(',') for i in range(len(simLibObj.mySettings["-amplicon"]))]
            
        for for_umi_i in range(self.Nbcn):
            for_param_index = np.random.randint(len(simLibObj.seqform_for_params))
            if len(simLibObj.seqform_for_params[for_param_index]) > 1:
                sysOps.throw_exception('Error: len(simLibObj.seqform_for_params[for_param_index]) = ' + str(len(simLibObj.seqform_for_params[for_param_index])))
                sysOps.exitProgram()
            my_for_umi_param = simLibObj.seqform_for_params[for_param_index][0]['U'][0]
            [start_pos,end_pos] = my_for_umi_param[0]
            seq_bool_vec = my_for_umi_param[1]
            my_for_umi = str('')
            for pos in range(end_pos-start_pos):
                possible_bases = np.where(seq_bool_vec[(pos*4):((pos+1)*4)])[0]
                my_for_umi += base_order[possible_bases[np.random.randint(possible_bases.shape[0])]]
                
            for_umi_seqs.append([int(for_param_index), str(my_for_umi)])
            
        for for_uei_i in range(self.Nuei):
            for_param_index = 0 # there should be no difference across UMI's
            my_for_uei_param = simLibObj.seqform_for_params[for_param_index][0]['U'][1]
            [start_pos,end_pos] = my_for_uei_param[0]
            seq_bool_vec = my_for_uei_param[1]
            my_for_uei = str('')
            for pos in range(end_pos-start_pos):
                possible_bases = np.where(seq_bool_vec[(pos*4):((pos+1)*4)])[0]
                my_for_uei += base_order[possible_bases[np.random.randint(possible_bases.shape[0])]]
                
            uei_seqs.append(str(my_for_uei))
        
        for rev_umi_i in range(self.Ntrg):
            rev_param_index = np.random.randint(len(simLibObj.seqform_rev_params))
            my_rev_umi_param = simLibObj.seqform_rev_params[rev_param_index][0]['U'][0]
            [start_pos,end_pos] = my_rev_umi_param[0]
            seq_bool_vec = my_rev_umi_param[1]
            my_rev_umi = str('')
            for pos in range(end_pos-start_pos):
                possible_bases = np.where(seq_bool_vec[(pos*4):((pos+1)*4)])[0]
                my_rev_umi += base_order[possible_bases[np.random.randint(possible_bases.shape[0])]]
            
            if len(amplicon_list) == 0:
                encoded_amplicon = str('')
            else:
                this_gsp_primer_amplicon_pair = list(amplicon_list[np.random.randint(len(amplicon_list))]) # already properly oriented # already properly oriented
                # generate single error on amplicon
                lenamp = len(this_gsp_primer_amplicon_pair[1])
                rand_loc = np.random.randint(lenamp)
                this_gsp_primer_amplicon_pair[1] = str(this_gsp_primer_amplicon_pair[1][:rand_loc] + base_order[np.random.randint(4)] + this_gsp_primer_amplicon_pair[1][(rand_loc+1):])
                encoded_amplicon = ''.join(this_gsp_primer_amplicon_pair)
            
            tmp_umi_index = float(rev_umi_i)
            
            if tmp_umi_index == 0:
                encoded_amplicon += base_order[0] 
            else:
                for myexponent in range(int(np.floor(np.log(tmp_umi_index)/np.log(4.0))),-1,-1):
                    mydigit = np.floor(tmp_umi_index/np.power(4.0,myexponent))
                    encoded_amplicon += base_order[int(mydigit)] 
                    tmp_umi_index -= mydigit*np.power(4.0,myexponent)
                
            rev_umi_seqs.append([int(rev_param_index), str(my_rev_umi), str(encoded_amplicon)])
        
        sysOps.throw_status('Writing simulated reads ...')
        
        for filename in filenames:
            if filename.endswith('_sim_ueifile.csv'):
                ueifile = np.int64(np.loadtxt(sysOps.globaldatapath + filename,delimiter=','))
                newdirname =filename[:filename.find('_')] 
                read_list = list()
                for i in range(ueifile.shape[0]):
                    for myread in range(ueifile[i,3]):
                        read_list.append(np.array([ueifile[i,:3]]))
                read_list = np.concatenate(read_list,axis = 0) # re-write array so that there is now one row per read
                # randomly permute:
                read_list = read_list[np.random.permutation(read_list.shape[0]),:]
                
                for_chararray = np.chararray((for_read_len))
                rev_chararray = np.chararray((rev_read_len))
                for_fastq_outfile = open(newdirname + '_for.fastq', "w")
                rev_fastq_outfile = open(newdirname + '_rev.fastq', "w")
                for i in range(read_list.shape[0]):
                    for_param_index = for_umi_seqs[read_list[i,1]][0]
                    for_umi_seq = for_umi_seqs[read_list[i,1]][1]
                    rev_param_index = rev_umi_seqs[read_list[i,2]][0] # both beacon and target indices are at this point are independently indexed from 0
                    rev_umi_seq = rev_umi_seqs[read_list[i,2]][1]
                    rev_amp_seq = rev_umi_seqs[read_list[i,2]][2]
                    uei_seq = uei_seqs[read_list[i,0]]
                    
                    for j in range(for_read_len):
                        for_chararray[j] = 'N'
                    for j in range(rev_read_len):
                        rev_chararray[j] = 'N'
                        
                    my_for_umi_param = simLibObj.seqform_for_params[for_param_index][0]['U'][0]
                    [start_pos,end_pos] = my_for_umi_param[0]
                    for j in range(end_pos-start_pos):
                        for_chararray[j+start_pos] = for_umi_seq[j]
                        
                    my_for_uei_param = simLibObj.seqform_for_params[for_param_index][0]['U'][1]
                    [start_pos,end_pos] = my_for_uei_param[0]
                    for j in range(end_pos-start_pos):
                        for_chararray[j+start_pos] = uei_seq[j]
                        
                    for my_for_param in simLibObj.seqform_for_params[for_param_index][0]['P']:
                        [start_pos,end_pos] = my_for_param[0]
                        for j in range(end_pos-start_pos):
                            for_chararray[j+start_pos] = base_order[np.where(my_for_param[1][(4*j):(4*(j+1))])[0][0]]
                        
                    my_rev_umi_param = simLibObj.seqform_rev_params[rev_param_index][0]['U'][0]
                    [start_pos,end_pos] = my_rev_umi_param[0]
                    for j in range(end_pos-start_pos):
                        rev_chararray[j+start_pos] = rev_umi_seq[j]
                    my_rev_amp_param = simLibObj.seqform_rev_params[rev_param_index][0]['A'][0]
                    start_pos = my_rev_amp_param[0][0]
                    for j in range(len(rev_amp_seq)):
                        rev_chararray[j+start_pos] = rev_amp_seq[j]
                    
                    if 'P' in simLibObj.seqform_rev_params[rev_param_index][0]:
                        for my_rev_param in simLibObj.seqform_rev_params[rev_param_index][0]['P']:
                            [start_pos,end_pos] = my_rev_param[0]
                            for j in range(end_pos-start_pos):
                                rev_chararray[j+start_pos] = base_order[np.where(my_rev_param[1][(4*j):(4*(j+1))])[0][0]]
                    
                    for_record = SeqIO.SeqRecord(Seq.Seq(for_chararray.tostring()))
                    for_record.id = '-' + str(i) + '-' + str(read_list[i,1])
                    for_record.description = ''
                    for_record.letter_annotations['phred_quality'] = list([30 for j in range(for_read_len)])
                    rev_record = SeqIO.SeqRecord(Seq.Seq(rev_chararray.tostring()))
                    rev_record.id = '-' + str(i) + '-' + str(read_list[i,2])
                    rev_record.description = ''
                    rev_record.letter_annotations['phred_quality'] = list([30 for j in range(rev_read_len)])
                    SeqIO.write(for_record, for_fastq_outfile, "fastq")
                    SeqIO.write(rev_record, rev_fastq_outfile, "fastq")
                    
                for_fastq_outfile.close()
                rev_fastq_outfile.close()
                os.mkdir(newdirname)
                with open('libsettings.txt','rU') as oldsettingsfile:
                    with open(newdirname + '//libsettings.txt', 'w') as newsettingsfile:
                        for oldsettings_row in oldsettingsfile:
                            if oldsettings_row.startswith('-source_for'):
                                newsettingsfile.write('-source_for ..//' + newdirname + '_for.fastq\n')
                            elif oldsettings_row.startswith('-source_rev'):
                                newsettingsfile.write('-source_rev ..//' + newdirname + '_rev.fastq\n')
                            else:
                                newsettingsfile.write(oldsettings_row)
                
        sysOps.throw_status('Done.')
        return
        
def initiate_sim():
    this_simObj = simObj('sim_params.csv')
    this_simObj.sim_physics()
    this_simObj.sim_reads()

if __name__ == '__main__':
    initiate_sim()
