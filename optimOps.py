import numpy as np
import scipy
import sysOps
import itertools
import scipy
from scipy import misc
import fileOps
from numpy import linalg as LA
from scipy.sparse.linalg import LinearOperator
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from numpy import random
import random
from numpy.random import rand
from importlib import import_module
from numba import jit, types
import os
import time

def test_ffgt(randw = 10, s = 1.0, spat_dims = 2, Nbcn = 1000, Ntrg = 1000):
    # Test function for FFGT N-body summation
    # Will simulate Nbcn beacons and Ntrg targets with positional variance randw
    # Summation is done with Gaussian kernel w_i := \sum_j exp(-(x_{ij}^2)/s + A_i + A_j)
    # A_i and A_j are randomly assigned as normally distribution variables with variance = 1
    
    Numi = Nbcn + Ntrg
    Xumi = randw*np.random.rand(Numi,spat_dims+2)
    Xumi[:,spat_dims] = np.random.randn(Numi)
    Xumi[:,spat_dims+1] = np.random.randn(Numi)
    
    dXumi = np.zeros([Numi,spat_dims+2])
    
    dx_bcn = np.zeros(spat_dims+2)
    dx_trg = np.zeros(spat_dims+2)
    
    # Perform exact calculation
    sysOps.throw_status('Beginning exact ' + str(Nbcn) + 'x' + str(Ntrg) + ' computation.')
    for i in range(Nbcn):
        for j in range(Ntrg):
            dx2 = np.sum(np.square(np.subtract(Xumi[i,:spat_dims],Xumi[j+Nbcn,:spat_dims])))
            dx_bcn[:] = 0.0
            dx_trg[:] = 0.0
            
            # assign gradient coefficients (that will multiply exp_factor = w_{ij})
            dx_bcn[:spat_dims] = -(2.0/s)*np.subtract(Xumi[i,:spat_dims],Xumi[j+Nbcn,:spat_dims])
            dx_trg[:spat_dims] = -dx_bcn[:spat_dims]
            dx_bcn[spat_dims] = 1.0
            dx_trg[spat_dims+1] = 1.0
            exp_factor = np.exp(-(dx2/s) + Xumi[i,spat_dims] + Xumi[j+Nbcn,spat_dims+1])
            dXumi[i,:] += np.multiply(dx_bcn,exp_factor)
            dXumi[j+Nbcn,:] += np.multiply(dx_trg,exp_factor)

    sysOps.throw_status('Completed exact computation. Continuing to test FFGT.')
    
    imagemodule_input_filename = 'test_ffgt.csv'
    with open(sysOps.globaldatapath + 'seq_params_' + imagemodule_input_filename,'w') as params_file:
            params_file.write('-Nbcn ' + str(int(Nbcn)) + '\n')         #total number of (analyzeable) bcn UMI's
            params_file.write('-Ntrg ' + str(int(Ntrg)) + '\n')         #total number of (analyzeable) trg UMI's
            params_file.write('-Nuei ' + str(int(0)) + '\n')            #total number of UEI's
            params_file.write('-Nassoc ' + str(int(0)) + '\n')          #total number of unique associations
            params_file.write('-spat_dims ' + str(spat_dims) + '\n')    #output dimensionality
            params_file.write('-err_bound ' + str(0.3) + '\n')          #maximum error
            
    this_mle = mleObj(imagemodule_input_filename)
    os.remove(sysOps.globaldatapath + 'seq_params_' + imagemodule_input_filename)
 
    umi_incl_ffgt = np.ones(this_mle.Numi,dtype=np.bool) #assign no outliers
    
    # call function to assign parameters on the basis of
    L, Q, min_x = this_mle.get_ffgt_args(Xumi[umi_incl_ffgt,:])  
    
    # Initial set up before call_ffgt(): ensure that all UMIs are accounted for in summation
    has_bcn_arr = np.zeros(this_mle.Numi,dtype=np.bool_)
    has_bcn_arr[:Nbcn] = np.True_   # beacon UMIs exist in UMIs 0..Nbcn-1
    has_trg_arr = np.zeros(this_mle.Numi,dtype=np.bool_)
    has_trg_arr[Nbcn:] = np.True_   # beacon UMIs exist in UMIs Nbcn..Nbcn+Ntrg-1
    
    # call FFGT to generate gradients and weight-sums
    (min_exp_amp_bcn, min_exp_amp_trg, 
     this_mle.prev_axial_bcn_fft, this_mle.prev_axial_trg_fft, 
     Xbcn_grad_sub, Xtrg_grad_sub) = call_ffgt(Xumi[umi_incl_ffgt,:],
                                             has_bcn_arr,has_trg_arr,
                                             this_mle.x_umi_polynom_tuples_buff[umi_incl_ffgt,:],
                                             min_x,this_mle.glo_indices,L,Q,this_mle.s,this_mle.max_nu,this_mle.spat_dims,True,True)
    
    sysOps.throw_status('Xbcn_grad.shape = ' + str(Xbcn_grad_sub.shape) + ' , Xtrg_grad.shape = ' + str(Xtrg_grad_sub.shape))
    sysOps.throw_status('Maximum bcn fractional error for 1st call: ' + 
                        str(np.max(np.divide(np.abs(np.subtract(Xbcn_grad_sub[has_bcn_arr,spat_dims],
                                                                dXumi[has_bcn_arr,spat_dims])),
                                             np.minimum(Xbcn_grad_sub[has_bcn_arr,spat_dims],
                                                        dXumi[has_bcn_arr,spat_dims])))))
    sysOps.throw_status('Maximum trg fractional error for 1st call: ' + 
                        str(np.max(np.divide(np.abs(np.subtract(Xtrg_grad_sub[has_trg_arr,spat_dims],
                                                                dXumi[has_trg_arr,spat_dims+1])),
                                             np.minimum(Xtrg_grad_sub[has_trg_arr,spat_dims],
                                                        dXumi[has_trg_arr,spat_dims+1])))))
    with open('test_ffgt1.csv','w') as outfile:
        for i in range(Nbcn):
            outfile.write("0," + str(i) + "," + ','.join([str(s) for s in Xbcn_grad_sub[has_bcn_arr][i,:]]) + "," + ','.join([str(s) for s in dXumi[i,:]]) + "\n")
        for i in range(Ntrg):
            outfile.write("1," + str(i) + "," + ','.join([str(s) for s in Xtrg_grad_sub[has_trg_arr][i,:]]) + "," + ','.join([str(s) for s in dXumi[Nbcn+i,:]]) + "\n")
    this_mle.prev_Q = float(Q)
    this_mle.prev_L = float(L)
    this_mle.prev_min_bcn_sumw = np.min(Xbcn_grad_sub[has_bcn_arr][:,this_mle.spat_dims])
    this_mle.prev_min_trg_sumw = np.min(Xtrg_grad_sub[has_trg_arr][:,this_mle.spat_dims])
    
    # Re-perform call: note that the only purpose of this is to re-randomize the orientation of the 
    # spatial grid used to perform the FFGT calculation relative to the inputted UMIs
    
    has_bcn_arr = np.zeros(this_mle.Numi,dtype=np.bool_)
    has_bcn_arr[:Nbcn] = np.True_
    has_trg_arr = np.zeros(this_mle.Numi,dtype=np.bool_)
    has_trg_arr[Nbcn:] = np.True_
    
    L, Q, min_x = this_mle.get_ffgt_args(Xumi[umi_incl_ffgt,:])
    (min_exp_amp_bcn, min_exp_amp_trg, 
     this_mle.prev_axial_bcn_fft, this_mle.prev_axial_trg_fft, 
     Xbcn_grad_sub, Xtrg_grad_sub) = call_ffgt(Xumi[umi_incl_ffgt,:],
                                             has_bcn_arr,has_trg_arr,
                                             this_mle.x_umi_polynom_tuples_buff[umi_incl_ffgt,:],
                                             min_x,this_mle.glo_indices,L,Q,this_mle.s,this_mle.max_nu,this_mle.spat_dims,True,True)
     
    sysOps.throw_status('Maximum bcn fractional error for 2nd call: ' + 
                        str(np.max(np.divide(np.abs(np.subtract(Xbcn_grad_sub[has_bcn_arr,spat_dims],
                                                                dXumi[has_bcn_arr,spat_dims])),
                                             np.minimum(Xbcn_grad_sub[has_bcn_arr,spat_dims],
                                                        dXumi[has_bcn_arr,spat_dims])))))
    sysOps.throw_status('Maximum trg fractional error for 2nd call: ' + 
                        str(np.max(np.divide(np.abs(np.subtract(Xtrg_grad_sub[has_trg_arr,spat_dims],
                                                                dXumi[has_trg_arr,spat_dims+1])),
                                             np.minimum(Xtrg_grad_sub[has_trg_arr,spat_dims],
                                                        dXumi[has_trg_arr,spat_dims+1])))))
    with open('test_ffgt2.csv','w') as outfile:
        for i in range(Nbcn):
            outfile.write("0," + str(i) + "," + ','.join([str(s) for s in Xbcn_grad_sub[has_bcn_arr][i,:]]) + "," + ','.join([str(s) for s in dXumi[i,:]]) + "\n")
        for i in range(Ntrg):
            outfile.write("1," + str(i) + "," + ','.join([str(s) for s in Xtrg_grad_sub[has_trg_arr][i,:]]) + "," + ','.join([str(s) for s in dXumi[Nbcn+i,:]]) + "\n")
    this_mle.prev_Q = float(Q)
    this_mle.prev_L = float(L)
    this_mle.prev_min_bcn_sumw = np.min(Xbcn_grad_sub[has_bcn_arr][:,this_mle.spat_dims])
    this_mle.prev_min_trg_sumw = np.min(Xtrg_grad_sub[has_trg_arr][:,this_mle.spat_dims])

    
def get_glo_indices(max_nu,spat_dims):
    # GLO = graded lexicographic order
    # Function enters into glo_indices an array
    # of int-index arrays containing graded lexicographic ordering of indices for each spatial dimension
    num_terms = 0

    for nu in range(max_nu+1):
        num_terms += np.prod(np.arange(nu+1,nu+spat_dims))/np.prod(np.arange(1,spat_dims))
    
    glo_indices = np.zeros([num_terms,spat_dims],dtype=np.int64)
    diag_term_indices = np.zeros(spat_dims,dtype=np.int64)

    on_glo_index = 1
    prev_order_final_index = 0
    
    for i in range(1,max_nu+1):
        for d in range(spat_dims):
            tmp_diagonal_index = int(on_glo_index)
            for j in range(diag_term_indices[d],prev_order_final_index+1):
                for d2 in range(spat_dims):
                    glo_indices[on_glo_index,d2] = glo_indices[j,d2] + int(d2==d)
                on_glo_index+=1
            diag_term_indices[d] = int(tmp_diagonal_index)
        prev_order_final_index = on_glo_index-1

    return glo_indices

def pt_mle(sub_mle,output_Xumi_filename):
    # perform unstructured "point-MLE" likelihood maximization
    
    # Initialize solution as random linear combination of top eigenvectors (with lowest magnitude eigenvalues) 
    # of UEI Graph-Laplacian matrix
    my_Xumi = sub_mle.seq_evecs.transpose().dot(np.sqrt(float(sub_mle.Numi)/float(sub_mle.seq_evecs.shape[0]))*np.random.randn(sub_mle.seq_evecs.shape[0],sub_mle.spat_dims))
    sub_mle.seq_evecs = None
    res = minimize(fun=sub_mle.calc_grad, x0=np.reshape(my_Xumi,sub_mle.Numi*sub_mle.spat_dims), args=(), 
                   method='L-BFGS-B', jac=True)
    # see calc_grad() for details on gradient calculation for optimization
    
    my_Xumi = np.reshape(res['x'],[sub_mle.Numi, sub_mle.spat_dims])
    np.savetxt(sysOps.globaldatapath + output_Xumi_filename,
               np.concatenate([sub_mle.index_key.reshape([sub_mle.Numi,1]), my_Xumi],axis = 1),
               fmt='%.10e',delimiter=',')
    return

def spec_mle(sub_mle, output_Xumi_filename = None):
    # perform structured "spectral MLE" (sMLE) likelihood maximization

    for i in range(sub_mle.seq_evecs.shape[0]): # normalize (just in case)
        sub_mle.seq_evecs[i,:] /= LA.norm(sub_mle.seq_evecs[i,:])
        
    submle_eignum = int(sub_mle.max_nontriv_eigenvec_to_calculate)
    all_seq_evecs = np.array(sub_mle.seq_evecs)
    X = None
    for eig_count in range(sub_mle.spat_dims,submle_eignum+1):
        # SOLVE SUB-MLE
        sysOps.throw_status('Optimizing eigencomponent ' + str(eig_count) + '/' + str(submle_eignum))
        if eig_count == sub_mle.spat_dims:
            X = np.eye(sub_mle.spat_dims,dtype=np.float64) 
            # initialize as identity matrix
        else:
            X = np.concatenate([X,np.zeros([1,sub_mle.spat_dims],dtype=np.float64)],axis = 0) 
            # add new eigenvector coefficients as degrees of freedom initialized to 0
            
        sub_mle.max_nontriv_eigenvec_to_calculate = eig_count # set number of degrees of freedom
        sub_mle.seq_evecs = all_seq_evecs[:eig_count,:] # set eigenvectors to sub-set
        
        # pre-calculate back-projection matrix: calculate inner-product of eigenvector matrix with itself, and invert to compensate for lack of orthogonalization between eigenvectors 
        sub_mle.backproj_mat = sub_mle.seq_evecs.transpose().dot(LA.inv(sub_mle.seq_evecs.dot(sub_mle.seq_evecs.transpose())))
        
        res = minimize(fun=sub_mle.calc_grad, 
                       x0=np.reshape(X,sub_mle.max_nontriv_eigenvec_to_calculate*sub_mle.spat_dims),
                       args=(), method='L-BFGS-B', jac=True)
    
        X = np.array(np.reshape(res['x'],[sub_mle.max_nontriv_eigenvec_to_calculate, sub_mle.spat_dims]))
        
        if eig_count == submle_eignum:
            my_Xumi = sub_mle.seq_evecs.transpose().dot(X)
            if not (output_Xumi_filename is None):
                np.savetxt(sysOps.globaldatapath +output_Xumi_filename,
                           np.concatenate([sub_mle.index_key.reshape([sub_mle.Numi,1]), my_Xumi],axis = 1),fmt='%i,%.10e,%.10e',delimiter=',')
                
                return
            else:
                break

# NUMBA declaration
@jit("void(int64[:,:],int64[:],int64[:,:],int64[:],int64[:],int64[:],bool_[:],bool_[:],int64)",nopython=True)
def sum_ueis_and_reassign_indices(bcn_sorted_uei_data,bcn_sorted_uei_data_starts,
                                  local_bcn_sorted_uei_data,local_index_lookup,
                                  sum_bcn_uei,sum_trg_uei,
                                  umi_inclusion_arr,assoc_inclusion_arr,Nbcn):
    
    # Function is called when new data arrays need to be used from a superset of UMIs to generate data arrays
    # corresponding to a UMI subset
    for n_super in range(Nbcn):
        # Note that looping through beacon UMIs is done a way to navigate through the FULL UEI matrix
        # sorted as bcn_sorted_uei_data
        # bcn_sorted_uei_data_starts stores data with same dimensionality as mleObj.uei_data, but sorted by beacon index
        #     Column ordering: beacon-index, target-index, uei_count
        for i in range(bcn_sorted_uei_data_starts[n_super],
                       bcn_sorted_uei_data_starts[n_super+1]):
            bcn_index = bcn_sorted_uei_data[i,0]
            trg_index = bcn_sorted_uei_data[i,1]
            if umi_inclusion_arr[bcn_index] and umi_inclusion_arr[trg_index]:
                # retention of UEI data in new data subset arrays requires that BOTH beacon- and target-UMIs
                # referred to by a UEI entry belong to the designated subset
                assoc_inclusion_arr[i] = True
                sum_bcn_uei[local_index_lookup[bcn_index]] += bcn_sorted_uei_data[i,2]
                sum_trg_uei[local_index_lookup[trg_index]] += bcn_sorted_uei_data[i,2]
                local_bcn_sorted_uei_data[i,0] = local_index_lookup[bcn_index]
                local_bcn_sorted_uei_data[i,1] = local_index_lookup[trg_index]
    
    return

def get_sparsest_cut(bcn_sorted_uei_data, bcn_sorted_uei_data_starts,umi_inclusion_arr, 
                     local_sum_bcn_uei, local_sum_trg_uei, local_Numi, randomize_cut = False):
    
    # Function performs symmetric Graph Laplacian decomposition, 
    # and sweeps the lowest-magnitude non-trivial eigenvector for the division between points that minimizes UEI conductance
    # Inputs:
    #    1. bcn_sorted_uei_data: uei_data that has not YET been sub-sampled according to the boolean array umi_inclusion_array, sorted by beacon index
    #    2. bcn_sorted_uei_data_starts: array of where UEIs start for each beacon UMI
    #    3. umi_inclusion_array: boolean array indicating which UMIs (whose indices in this array are referred to in first 2 columns of bcn_sorted_uei_data)
    #        will be analyzed in this function call
    #    4. local_sum_bcn_uei: (unassigned) total UEI counts belonging to beacons at designated UMI index local to True elements of umi_inclusion_arr
    #    5. local_sum_trg_uei: (unassigned) total UEI counts belonging to targets at designated UMI index local to True elements of umi_inclusion_arr
    #    6. local_Numi total True elements in umi_inclusion_arr
    
    local_sum_bcn_uei[:] = 0
    local_sum_trg_uei[:] = 0
    
    assoc_inclusion_arr = np.zeros(bcn_sorted_uei_data.shape[0],dtype=np.bool_) 
    # boolean array keeps track of which associations (corresponding to the rows of bcn_sorted_uei_data)
    # will be retained on account of the UMIs being included according to input array umi_inclusion_arr
    
    local_index_lookup = -np.ones(umi_inclusion_arr.shape[0],dtype=np.int64)
    # indicies will correspond to super-set's indices, values will correspond to sub-set indices
    local_index_lookup[umi_inclusion_arr] = np.arange(local_Numi)

    local_bcn_sorted_uei_data = np.array(bcn_sorted_uei_data)
    
    # tabulate local (sub-set) statistics, replace indices in local_bcn_sorted_uei_data with sub-set indicies
    sum_ueis_and_reassign_indices(bcn_sorted_uei_data,bcn_sorted_uei_data_starts,
                                  local_bcn_sorted_uei_data,local_index_lookup,
                                  local_sum_bcn_uei,local_sum_trg_uei,
                                  umi_inclusion_arr,assoc_inclusion_arr,bcn_sorted_uei_data_starts.shape[0]-1)
    
    local_Nassoc = np.sum(assoc_inclusion_arr)
    
    # remake UEI array to ONLY include associations retained according to input array umi_inclusion_arr
    # local_bcn_sorted_uei_data_starts and local_trg_sorted_uei_data_starts will store locations in 
    # local_bcn_sorted_uei_data and local_trg_sorted_uei_data, respectively, where beacon or target UMI's
    
    local_bcn_sorted_uei_data = local_bcn_sorted_uei_data[assoc_inclusion_arr,:]
    local_bcn_sorted_uei_data_starts = np.append(np.append(0,1+np.where(np.diff(local_bcn_sorted_uei_data[:,0])>0)[0]),
                                                 local_bcn_sorted_uei_data.shape[0])
    local_trg_sorted_uei_data = local_bcn_sorted_uei_data[np.argsort(local_bcn_sorted_uei_data[:,1]),:]
    local_trg_sorted_uei_data_starts = np.append(np.append(0,1+np.where(np.diff(local_trg_sorted_uei_data[:,1])>0)[0]),
                                                 local_trg_sorted_uei_data.shape[0])
    
    row_indices = np.arange(local_Numi + 2*local_Nassoc, dtype=np.int64)
    col_indices = np.arange(local_Numi + 2*local_Nassoc, dtype=np.int64)
    norm_uei_data = np.zeros(local_Numi + 2*local_Nassoc, dtype=np.float64)
    
    # Generate symmetric Graph Laplacian with local UEI data, initiate sparse matrix data object
    get_normalized_sparse_matrix(local_sum_bcn_uei,local_sum_trg_uei,
                                 row_indices,col_indices,
                                 norm_uei_data,local_bcn_sorted_uei_data,
                                 local_Nassoc,local_Numi,np.True_) # get symmetrized Laplacian to perform sparsest cut
    csc_op = csc_matrix((norm_uei_data, (row_indices, col_indices)), (local_Numi, local_Numi))
    
    if local_Numi <= 4:
        evals, evecs = LA.eigh(csc_op.toarray())
        eval_order = np.argsort(np.abs(evals))
        evecs = evecs[:,eval_order]
        evals = evals[eval_order]
    else:
        evals, evecs = scipy.sparse.linalg.eigsh(csc_op, k=2, M = None, which='SM', v0=None, ncv=None, maxiter=None, tol = 0)
        # note: symmetric Graph Laplacian is used here, so we can use Hermitian matrix decomposition above, which will accelerate the operation slightly
    
    # remove trivial eigenvector
    triv_eig_index = np.argmin(np.var(evecs[:,:2],axis = 0))
    top_nontriv_evec = evecs[:,np.where(np.arange(2) != triv_eig_index)[0][0]]
    ordered_symm_nontriv_evec_indices = np.argsort(top_nontriv_evec)
    cut_passed = np.zeros(local_Numi,dtype=np.bool_)
    
    min_conductance_ptr = np.array([0],dtype=np.float64)
    min_conductance_assoc_ptr = np.array([0],dtype=np.int64)
    local_Nbcn = int(local_bcn_sorted_uei_data_starts.shape[0]-1)
    sparsest_cut(min_conductance_ptr,min_conductance_assoc_ptr,
                 local_bcn_sorted_uei_data,local_bcn_sorted_uei_data_starts,
                 local_trg_sorted_uei_data,local_trg_sorted_uei_data_starts,
                 local_sum_bcn_uei, local_sum_trg_uei,
                 ordered_symm_nontriv_evec_indices,
                 cut_passed, 
                 local_Numi, local_Nbcn)

    if randomize_cut: # randomize choice of cut (disregard result of sparsest_cut for this purpose, even though conductance is measured)
        sysOps.throw_exception('Error: not supported by current version.')
        sysOps.exitProgram()
    
    # Outputs:
    #    1. min_conductance_ptr[0]: Minimum conductance cut value
    #    2. min_conductance_assoc_ptr[0]: Number of unique UMI-UMI associations crossing minimizing conductance
    #    3. cut_passed: boolean array indicating which UMIs (from True elements of umi_inclusion_arr) are on which side of cut
    #    4. local_bcn_sorted_uei_data: new UEI data set of the same type as bcn_sorted_uei_data, but corresponding to only True entries in umi_inclusion_arr, with new indices
    #    4. local_bcn_sorted_uei_data_starts: new UEI data set of the same type as bcn_sorted_uei_data_starts, but corresponding to only True entries in umi_inclusion_arr, with new indices
    return (min_conductance_ptr[0], min_conductance_assoc_ptr[0], cut_passed, local_bcn_sorted_uei_data, local_bcn_sorted_uei_data_starts)

# NUMBA declaration
@jit("void(float64[:],int64[:],int64[:,:],int64[:],int64[:,:],int64[:],int64[:],int64[:],int64[:],bool_[:],int64,int64)",nopython=True)
def sparsest_cut(min_conductance_ptr,min_conductance_assoc_ptr,
                 local_bcn_sorted_uei_data,local_bcn_sorted_uei_data_starts,
                 local_trg_sorted_uei_data,local_trg_sorted_uei_data_starts,
                 local_sum_bcn_ueis, local_sum_trg_ueis,
                 ordered_symm_nontriv_evec_indices,
                 cut_passed, 
                 local_Numi, local_Nbcn):
    # Function sweeps the lowest-magnitude non-trivial eigenvector for the division between points that minimizes UEI conductance
    # Note: top_symm_nontriv_evec must be delivered from an eigen-decomposition of a symmetrized Graph laplacian in order for sparsest cut to work properly
    # assumes indexing of bcn_sorted_uei_data is done according to same indices as top_nontriv_evec
    
    n = ordered_symm_nontriv_evec_indices[0]
    cut_passed[n] = True
    
    if n < local_Nbcn:
        my_cut_assoc = local_bcn_sorted_uei_data_starts[n+1]-local_bcn_sorted_uei_data_starts[n]
    else:
        my_cut_assoc = local_trg_sorted_uei_data_starts[n-local_Nbcn+1]-local_trg_sorted_uei_data_starts[n-local_Nbcn]
        
    my_cut_flow = local_sum_bcn_ueis[n] + local_sum_trg_ueis[n]
    left_volume = int(my_cut_flow) # left side's volume all corresponds to edges flowing to right
    right_volume = -int(my_cut_flow)
    for n in range(local_Numi):
        right_volume += local_sum_bcn_ueis[n] + local_sum_trg_ueis[n]
        
    min_cut_conductance = 1.0 # flow divided by graph volume
    min_cut_conductance_assoc = int(my_cut_assoc)
    min_cut_index = 0
    
    for my_cut_index in range(1,local_Numi-1): # my_cut_index is the index BEFORE the cut
        n = ordered_symm_nontriv_evec_indices[my_cut_index]
        if n < local_Nbcn:
            for i in range(local_bcn_sorted_uei_data_starts[n],local_bcn_sorted_uei_data_starts[n+1]):
                if cut_passed[local_bcn_sorted_uei_data[i,1]]:
                    my_cut_flow -= local_bcn_sorted_uei_data[i,2]
                    my_cut_assoc -= 1
                else:
                    my_cut_flow += local_bcn_sorted_uei_data[i,2]
                    my_cut_assoc += 1
        else:
            for i in range(local_trg_sorted_uei_data_starts[n-local_Nbcn],local_trg_sorted_uei_data_starts[n-local_Nbcn+1]):
                if cut_passed[local_trg_sorted_uei_data[i,0]]:
                    my_cut_flow -= local_trg_sorted_uei_data[i,2]
                    my_cut_assoc -= 1
                else:
                    my_cut_flow += local_trg_sorted_uei_data[i,2]
                    my_cut_assoc += 1
                    
        cut_passed[n] = True
        left_volume += local_sum_bcn_ueis[n] + local_sum_trg_ueis[n]
        right_volume -= local_sum_bcn_ueis[n] + local_sum_trg_ueis[n]
        
        my_cut_conductance = float(my_cut_flow)/float(min(left_volume,right_volume)) # definition of conductance
        if my_cut_conductance < min_cut_conductance:
            min_cut_conductance = float(my_cut_conductance)
            min_cut_conductance_assoc = int(my_cut_assoc)
            min_cut_index = int(my_cut_index)
    
    for n in range(min_cut_index+1):
        cut_passed[ordered_symm_nontriv_evec_indices[n]] = False
    
    for n in range(min_cut_index+1,local_Numi):
        cut_passed[ordered_symm_nontriv_evec_indices[n]] = True
    
    min_conductance_ptr[0] = min_cut_conductance
    min_conductance_assoc_ptr[0] = min_cut_conductance_assoc
    
    return

def rec_sparsest_cut(bcn_sorted_uei_data,bcn_sorted_uei_data_starts,umi_inclusion_array,
                     my_start_index,stopping_conductance,stopping_assoc,maxumi,minumi,minuei,randomize_cut = False, recursive_counter = 0):
    # Function is recursively called on progressively smaller subsets of data
    # Cuts are performed using the spectral approximation to the sparsest cut, through calls to get_sparsest_cut()
    # Reminder: an ASSOCIATION is a unique UMI/UMI pairing
    # uei_data arrays have size = (total associations,3) --> 3 columns = (beacon UMI index, target UMI index, number of UEIs for this association)
    # Inputs:
    #    1. bcn_sorted_uei_data: uei_data that has not YET been sub-sampled according to the boolean array umi_inclusion_array, sorted by beacon index
    #    2. bcn_sorted_uei_data_starts: array of where UEIs start for each beacon UMI
    #    3. umi_inclusion_array: boolean array indicating which UMIs (whose indices in this array are referred to in first 2 columns of bcn_sorted_uei_data)
    #        will be analyzed in this function call
    #    4. my_start_index: current GROUPING index, ensures that when group indices are returned, they are non-overlapping
    #    5. stopping_conductance:  required stop-cut criterion (if != None),  if sparsest cut conductance falls above this, do not perform cut
    #    6. stopping_assoc:  required stop-cut criterion (if != None),  if associations across sparsest cut fall above this, do not perform cut
    #    7. maxumi: required stop-cut criterion (if != None), number of UMI in current partition <= maxumi
    #    8. minumi: required FULL CUT (segment each point separately) criterion, number of UMI in current partition < minumi
    #    9. minuei: UEI pruning criteria --> no UMI may remain within a partition if by restricting analysis to that partition it has fewer than this number of UEIs
    
    # umi_inclusion_array is passed as a boolean array for which only True elements are addressed in this call of rec_sparsest_cut()
    local_Numi = np.sum(umi_inclusion_array)
    
    if local_Numi < minumi or recursive_counter == 999:
        return np.add(my_start_index,np.arange(local_Numi,dtype=np.int64)) # if number of True elements in umi_inclusion_array is below a minimum, return
    
    local_sum_bcn_uei = np.zeros(local_Numi,dtype=np.int64)
    local_sum_trg_uei = np.zeros(local_Numi,dtype=np.int64)
    
    (min_conductance, min_conductance_assoc, cut_passed,
     local_bcn_sorted_uei_data,
     local_bcn_sorted_uei_data_starts) = get_sparsest_cut(bcn_sorted_uei_data, bcn_sorted_uei_data_starts,umi_inclusion_array, 
                                                          local_sum_bcn_uei, local_sum_trg_uei, local_Numi,randomize_cut)
    # returned values of get_sparsest_cut():
    #    1. min_conductance: UEIs(connecting PART A, PART B)/min(UEIs from PART A, UEIs from PART B) --> with BOTH PART A and PART B among the True elements of umi_inclusion_array
    #    2. min_conductance_assoc: number of distinct beacon UMI - target UMI associations connecting PART A to PART B
    #    3. cut_passed: boolean array designating which UMIs belong to PART A and PART B of cut
    #    4. local_bcn_sorted_uei_data: UEI data sub-set corresponding to portion of bcn_sorted_uei_data corresponding to UMIs corresponding to True elements of umi_inclusion_array (sorted by beacon index, ie the first column)
    #    5. local_bcn_sorted_uei_data_starts: integer array containing start indices of beacon UMIs in local_bcn_sorted_uei_data
         
    if ((stopping_conductance is None or min_conductance >= stopping_conductance)
        and (stopping_assoc is None or min_conductance_assoc >= stopping_assoc)
        and (maxumi is None or local_Numi <= maxumi)):
        sysOps.throw_status('Found block of size ' + str(local_Numi) + ', min_conductance_assoc = ' + str(min_conductance_assoc))
        return np.multiply(my_start_index,np.ones(local_Numi,dtype=np.int64))
    
    assoc_inclusion_arr = np.ones(local_bcn_sorted_uei_data.shape[0],dtype=np.bool_)
    for i in np.where(cut_passed[local_bcn_sorted_uei_data[:,0]] != cut_passed[local_bcn_sorted_uei_data[:,1]])[0]:
        local_sum_bcn_uei[local_bcn_sorted_uei_data[i,0]] -= local_bcn_sorted_uei_data[i,2]
        local_sum_trg_uei[local_bcn_sorted_uei_data[i,1]] -= local_bcn_sorted_uei_data[i,2]
        assoc_inclusion_arr[i] = np.False_
        
    remove_assoc = np.zeros(local_bcn_sorted_uei_data.shape[0],dtype=np.bool_)
    while True:
        remove_assoc = np.multiply(assoc_inclusion_arr,
                                   np.add(local_sum_bcn_uei[local_bcn_sorted_uei_data[:,0]]<minuei,
                                          local_sum_trg_uei[local_bcn_sorted_uei_data[:,1]]<minuei))
        if np.sum(remove_assoc) == 0:
            break

        for i in np.where(remove_assoc)[0]:
            local_sum_bcn_uei[local_bcn_sorted_uei_data[i,0]] -= local_bcn_sorted_uei_data[i,2]
            local_sum_trg_uei[local_bcn_sorted_uei_data[i,1]] -= local_bcn_sorted_uei_data[i,2]
        
        assoc_inclusion_arr = np.multiply(assoc_inclusion_arr,~remove_assoc)
        
    index_link_array = np.arange(local_Numi,dtype=np.int64)
    if np.sum(assoc_inclusion_arr) > 0: 
        # perform single linkage clustering on the current partitioned data sets to ensure that after pruning, to establish contiguous data sets given pruning so far in the function
        min_contig_edges(index_link_array,np.int64(cut_passed),
                         local_bcn_sorted_uei_data[assoc_inclusion_arr,:],
                         np.sum(assoc_inclusion_arr))
        
    sorted_index_link_array = np.argsort(index_link_array)
    index_link_starts = np.append(np.append(0,1+np.where(np.diff(index_link_array[sorted_index_link_array])>0)[0]),sorted_index_link_array.shape[0])
    grp_inclusion = np.zeros(local_Numi,dtype=np.bool_)
    grp_indices = -np.ones(local_Numi,dtype=np.int64)
    for i in range(index_link_starts.shape[0]-1):
        grp_inclusion[sorted_index_link_array[index_link_starts[i]:index_link_starts[i+1]]] = np.True_ # set to true those items in the boolean array that will be addressed during this call to rec_sparsest_cut()
        grp_indices[grp_inclusion] = rec_sparsest_cut(local_bcn_sorted_uei_data,local_bcn_sorted_uei_data_starts,
                                                      grp_inclusion,
                                                      my_start_index,stopping_conductance,stopping_assoc,maxumi,minumi,minuei,randomize_cut,recursive_counter+1)
        my_start_index = np.max(grp_indices[grp_inclusion])+1
        grp_inclusion[sorted_index_link_array[index_link_starts[i]:index_link_starts[i+1]]] = np.False_ # re-set
    
    return grp_indices # return segmented indices for True items in umi_inclusion_array
    
def generate_complete_indexed_arr(arr):
    
    # Function generates complete-indexing for 2D array's non-negative entries
    # Input: arbitrary 2D int-array
    # Output: 2D array with non-negative entries set to indices counting consecutively from 0, lookup array for looking up original entries on the basis of new indexing system
    
    if len(arr.shape) > 1:
        tmp_arr = np.reshape(arr,arr.shape[0]*arr.shape[1])
    else:
        tmp_arr = np.array(arr)
    tmp_arr_sorted = np.argsort(tmp_arr)
    tmp_arr_sorted_starts = np.append(np.append(0,1+np.where(np.diff(tmp_arr[tmp_arr_sorted])>0)[0]),
                                      tmp_arr_sorted.shape[0])
    index_lookup = -np.ones(tmp_arr_sorted_starts.shape[0]-1,dtype=np.int64) # elements will be original values
    on_index = 0
    for i in range(tmp_arr_sorted_starts.shape[0]-1):
        if tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]]] >= 0:
            index_lookup[on_index] = tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]]]
            tmp_arr[tmp_arr_sorted[tmp_arr_sorted_starts[i]:tmp_arr_sorted_starts[i+1]]] = on_index
            on_index += 1
            
    if len(arr.shape) > 1:
        return np.reshape(tmp_arr,[arr.shape[0],arr.shape[1]]), index_lookup
    else:
        return np.array(tmp_arr), index_lookup
    
def segmentation_analysis(imagemodule_input_filename, this_mle, stopping_conductance, min_conductance_assoc, maxumi = None, minumi =50, minuei =2, eig_analysis_umi = 5000):
    # Perform segmentation analysis
    # Input:
    #    1. imagemodule_input_filename: UEI data input file
    #    2. this_mle: mleObj object for performing eigendecomposition
    #    3. stopping_conductance: min-conductance threshold  (if != None)
    #    4. min_conductance_assoc: required  (if != None) number of UMI-UMI associations across putative cut in order to stop cutting
    #    5. maxumi: required stop-cut criterion (if != None), number of UMI in current partition <= maxumi
    #    8. minumi: required FULL CUT (segment each point separately) criterion, number of UMI in current partition < minumi
    #    9. minuei: UEI pruning criteria --> no UMI may remain within a partition if by restricting analysis to that partition it has fewer than this number of UEIs
    #    10. eig_analysis_umi: number of UMI in partition that warrants eigen-analysis on that partition
    
    bcn_sorted_uei_data = np.array(this_mle.uei_data[np.argsort(this_mle.uei_data[:,0]),:])
    bcn_sorted_uei_data_starts = np.append(np.append(0,1+np.where(np.diff(bcn_sorted_uei_data[:,0])>0)[0]),
                                               bcn_sorted_uei_data.shape[0])
    segmentation_assignments = rec_sparsest_cut(bcn_sorted_uei_data,bcn_sorted_uei_data_starts,np.ones(this_mle.Numi,dtype=np.bool_),
                                                0,stopping_conductance,min_conductance_assoc,maxumi,minumi,minuei,randomize_cut = False, recursive_counter = 0)
    segmentation_assignments, old_segment_lookup = generate_complete_indexed_arr(segmentation_assignments)
    
    umi_per_segment = np.zeros(np.max(segmentation_assignments)+1,dtype=np.int64)
    for n in range(this_mle.Numi):
        umi_per_segment[segmentation_assignments[n]] += 1
    
    # perform 3D eigendecompositions for each segment individually
    uei_segment_indices = -np.ones(this_mle.uei_data.shape[0],dtype=np.int64)
    for i in range(this_mle.uei_data.shape[0]):
        bcn_segment = segmentation_assignments[this_mle.uei_data[i,0]]
        trg_segment = segmentation_assignments[this_mle.uei_data[i,1]]
        if (bcn_segment == trg_segment and bcn_segment >= 0):
            uei_segment_indices[i] = bcn_segment
            
    uei_segment_indices_sort = np.argsort(uei_segment_indices)
    uei_segment_indices_sort_starts = np.append(np.append(0,1+np.where(np.diff(uei_segment_indices[uei_segment_indices_sort])>0)[0]),
                                                uei_segment_indices_sort.shape[0])
    if uei_segment_indices[uei_segment_indices_sort[0]] < 0:
        uei_segment_indices_sort_starts = uei_segment_indices_sort_starts[1:] # ignore negative (ie unassigned) indices
    
    segment_coords = np.zeros([this_mle.Numi, 3],dtype=np.float64)
    for i in range(uei_segment_indices_sort_starts.shape[0]-1):
        start = uei_segment_indices_sort_starts[i]
        end = uei_segment_indices_sort_starts[i+1]
        if umi_per_segment[uei_segment_indices[uei_segment_indices_sort[start]]] >= eig_analysis_umi:
            uei_indices = uei_segment_indices_sort[start:end]
            sub_mle = mleObj(None, None, this_mle, np.array(this_mle.uei_data[uei_indices,:]))
            sub_mle.max_nontriv_eigenvec_to_calculate = 3
            sub_mle.eigen_decomp(None)
            segment_coords[sub_mle.index_key,:] = sub_mle.seq_evecs[:3,:].T
        
    np.savetxt(sysOps.globaldatapath + 'Xumi_segment_' + str(stopping_conductance) + '_' + imagemodule_input_filename,
               np.concatenate([this_mle.index_key.reshape([this_mle.Numi,1]),
                               segmentation_assignments.reshape([this_mle.Numi,1]),
                               segment_coords],axis = 1),fmt='%i,%i,%.10e,%.10e,%.10e',delimiter=',')
    
def run_mle(imagemodule_input_filename, smle = False, multiscale = False, segment = False, compute_local_solutions_only = False):
    # Primary function call for image inference and segmentation
    # Inputs:
    #     imagemodule_input_filename: UEI data input file
    #     other arguments: boolean settings for which subroutine to run
    
    # Initiating the amplification factors involves examining the solution when all positions are equal
    # This gives, for UMI k: n_{k\cdot} = \frac{n_{\cdot\cdot}}{(\sum_{i\neq k} e^{A_i})(\sum_j e^{A_j})/(e^{A_k}(\sum_j e^{A_j})) + 1} 
    
    minuei = 2
    tot_tess_iter = 10
    final_eignum = 50
    DEFAULT_SUBMLE_EIGNUM = 50
    minumi = 1000
    this_mle = mleObj(imagemodule_input_filename)
    this_mle.reduce_to_largest_linkage_cluster()
    
    if segment:
        for stopping_conductance in [0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26]:
            if (not sysOps.check_file_exists('placeholder_segment_' + str(stopping_conductance) + '.csv')
                and not sysOps.check_file_exists('Xumi_segment_' + str(stopping_conductance) + '_' + imagemodule_input_filename)):
                with open(sysOps.globaldatapath + 'placeholder_segment_' + str(stopping_conductance) + '.csv', 'w') as placeholderfile:
                    placeholderfile.write('placeholder')
                sysOps.throw_status('Performing segmentation on the UEI matrix with stopping_conductance = ' + str(stopping_conductance))
                segmentation_analysis(imagemodule_input_filename, this_mle, stopping_conductance , None, maxumi=None,minumi =50, minuei =2, eig_analysis_umi = 5000)
                os.remove(sysOps.globaldatapath + 'placeholder_segment_' + str(stopping_conductance) + '.csv')
            
        # determine if this data-set is a subsample -- if so, do not continue
        parent_dir = os.path.abspath(os.path.join('..', os.pardir)).strip('/')
        parent_dir = parent_dir[(parent_dir.rfind('/')+1):]
        if not parent_dir.startswith('sub'):
            for stopping_conductance in [0.005, 0.01, 0.02, 0.04, 0.06]:
                if (not sysOps.check_file_exists('placeholder_segment_' + str(stopping_conductance) + '.csv')
                    and not sysOps.check_file_exists('Xumi_segment_' + str(stopping_conductance) + '_' + imagemodule_input_filename)):
                    with open(sysOps.globaldatapath + 'placeholder_segment_' + str(stopping_conductance) + '.csv', 'w') as placeholderfile:
                        placeholderfile.write('placeholder')
                    segmentation_analysis(imagemodule_input_filename, this_mle, stopping_conductance , this_mle.spat_dims + 1,maxumi=250000,minumi =5000, minuei =2, eig_analysis_umi = 5000)
                    os.remove(sysOps.globaldatapath + 'placeholder_segment_' + str(stopping_conductance) + '.csv')
            
        else:
            sysOps.throw_status('Parent directory determined to have been subsampled. Not continuing with detailed inferences.')
            
        while not compute_local_solutions_only: 
            [subdirnames, filenames] = sysOps.get_directory_and_file_list()
            if True in [filename.startswith('placeholder') for filename in filenames]:
                sysOps.throw_status('Placeholder file/s found. Waiting ...')
                time.sleep(60)
            else:
                break
            
        sysOps.throw_status('All segmentations found computed.')
    elif smle and not multiscale and not sysOps.check_file_exists('Xumi_uniscale_' + imagemodule_input_filename):
        sysOps.throw_status('Running sMLE. Initiating ...')
        this_mle.max_nontriv_eigenvec_to_calculate = 100
        this_mle.eigen_decomp(imagemodule_input_filename)
        this_mle.print_status = False
        spec_mle(this_mle, output_Xumi_filename = 'Xumi_smle_' + imagemodule_input_filename)
    elif not sysOps.check_file_exists('Xumi_pt_' + imagemodule_input_filename):
        sysOps.throw_status('Running ptMLE. Initiating eigenvectors ...')
        this_mle.max_nontriv_eigenvec_to_calculate = 100
        this_mle.eigen_decomp(imagemodule_input_filename)
        sysOps.throw_status('Continuing optimization ...')
        pt_mle(this_mle,'Xumi_pt_' + imagemodule_input_filename)
    else:
        sysOps.throw_status('In run_mle, no open task to be performed. Returning.')


def generate_new_subdata(uei_data, this_mle):
    # Function will assign new local indices starting from 0, and will output key to index_key
    # Assumes mutual exclusivity of beacon and target indices in uei_data, however does not assume indexing from 0 consecutively
    
    bcn_uei_index_sort = np.argsort(uei_data[:,0])
    trg_uei_index_sort = np.argsort(uei_data[:,1])
    non_redundant_bcn_starts = np.append(np.append(0,1+np.where(np.diff(uei_data[bcn_uei_index_sort,0])>0)[0]),
                                         uei_data.shape[0])
    non_redundant_trg_starts = np.append(np.append(0,1+np.where(np.diff(uei_data[trg_uei_index_sort,1])>0)[0]),
                                         uei_data.shape[0])
    
    this_mle.Nbcn = non_redundant_bcn_starts.shape[0]-1
    this_mle.Ntrg = non_redundant_trg_starts.shape[0]-1
    this_mle.Numi = this_mle.Nbcn + this_mle.Ntrg
    this_mle.index_key = -np.ones(this_mle.Numi,dtype=np.int64)
    this_mle.uei_data = np.array(uei_data)
    for n in range(this_mle.Nbcn):
        this_mle.index_key[n] = this_mle.uei_data[bcn_uei_index_sort[non_redundant_bcn_starts[n]],0]
        for i in range(non_redundant_bcn_starts[n],non_redundant_bcn_starts[n+1]):
            this_mle.uei_data[bcn_uei_index_sort[i],0] = n
            
    for n in range(this_mle.Ntrg):
        this_mle.index_key[n + this_mle.Nbcn] = this_mle.uei_data[trg_uei_index_sort[non_redundant_trg_starts[n]],1]
        for i in range(non_redundant_trg_starts[n],non_redundant_trg_starts[n+1]):
            this_mle.uei_data[trg_uei_index_sort[i],1] = n + this_mle.Nbcn 
    
    this_mle.sum_bcn_uei = np.zeros(this_mle.Numi,dtype=np.int64)
    this_mle.sum_trg_uei = np.zeros(this_mle.Numi,dtype=np.int64)
    this_mle.sum_self_bcn_uei = np.zeros(this_mle.Numi,dtype=np.int64)
    this_mle.sum_self_trg_uei = np.zeros(this_mle.Numi,dtype=np.int64)
    this_mle.sum_bcn_assoc = np.zeros(this_mle.Numi,dtype=np.int64)
    this_mle.sum_trg_assoc = np.zeros(this_mle.Numi,dtype=np.int64)

    for i in range(uei_data.shape[0]):
        if this_mle.uei_data[i,0] == this_mle.uei_data[i,1]:
            this_mle.sum_self_bcn_uei[this_mle.uei_data[i,0]] += this_mle.uei_data[i,2]
            this_mle.sum_self_trg_uei[this_mle.uei_data[i,1]] += this_mle.uei_data[i,2]
        else:
            this_mle.sum_bcn_uei[this_mle.uei_data[i,0]] += this_mle.uei_data[i,2]
            this_mle.sum_trg_uei[this_mle.uei_data[i,1]] += this_mle.uei_data[i,2]
        this_mle.sum_bcn_assoc[this_mle.uei_data[i,0]] += 1
        this_mle.sum_trg_assoc[this_mle.uei_data[i,1]] += 1

    this_mle.Nuei = np.sum(this_mle.uei_data[:,2])
    this_mle.Nassoc = this_mle.uei_data.shape[0]
    
    return
    
class mleObj:
    # object for all image inference
    
    def __init__(self, imagemodule_input_filename, inp_settings = None, inp_mle = None, inp_data = None):
        
        self.seq_evecs = None
        self.backproj_mat = None
        self.uei_data = None
        self.prev_axial_bcn_fft = None
        self.prev_axial_trg_fft = None
        self.prev_Q = None
        self.prev_L = None
        self.prev_min_bcn_sumw = None
        self.prev_min_trg_sumw = None
        self.sumw = None
        self.ffgt_call_count = 0 # will act as an internal counter of FFGT-calls
        self.sum_bcn_uei = None
        self.sum_trg_uei = None
        self.sum_bcn_assoc = None
        self.sum_trg_assoc = None
        self.bcn_amp_factors = None
        self.trg_amp_factors = None
        self.sum_self_bcn_uei = None
        self.sum_self_trg_uei = None
        self.Numi = None
        self.print_status = True
        self.rel_err_bound = True # by default, err_bound will be evaluated as worst-case relative to the FFGT output value  
        
        # counts and indices in inp_data, if this is included in input, take precedence over read-in numbers from inp_settings and imagemodule_input_filename
        
        if type(inp_settings) == dict:
            self.mleSettings = dict(inp_settings)
        elif type(imagemodule_input_filename) == str:
            self.mleSettings = fileOps.read_settingsfile_to_dictionary('seq_params_' + imagemodule_input_filename)
        else:
            self.mleSettings = dict(inp_mle.mleSettings)
        
        print(str(self.mleSettings))
            
        try:
            if not(inp_mle is None):
                self.Nbcn = int(inp_mle.Nbcn)
            elif type(self.mleSettings) == dict:
                self.Nbcn = int(self.mleSettings['-Nbcn'][0])
            else:
                self.Nbcn = None
        except:
            sysOps.throw_exception('ERROR: No Nbcn value entered.')
            sysOps.exitProgram()
            
        try:
            if not(inp_mle is None):
                self.Ntrg = int(inp_mle.Ntrg)
            elif type(self.mleSettings) == dict:
                self.Ntrg = int(self.mleSettings['-Ntrg'][0])
            else:
                self.Ntrg = None
        except:
            sysOps.throw_exception('ERROR: No Ntrg value entered.')
            sysOps.exitProgram()
            
        
        try:
            if not(inp_mle is None):
                self.Nuei = int(inp_mle.Nuei)
            else:
                self.Nuei = int(self.mleSettings['-Nuei'][0])
        except:
            sysOps.throw_exception('ERROR: No Nuei value entered.')
            sysOps.exitProgram()
            
        try:
            if not(inp_mle is None):
                self.Nassoc = int(inp_mle.Nassoc)
            else:
                self.Nassoc = int(self.mleSettings['-Nassoc'][0])
        except:
            sysOps.throw_exception('ERROR: No Nassoc value entered.')
            sysOps.exitProgram()
            
        try:
            if not(inp_mle is None):
                self.spat_dims = int(inp_mle.spat_dims)
            else:
                self.spat_dims = int(self.mleSettings['-spat_dims'][0])
        except:
            sysOps.throw_exception('ERROR: No spat_dims value entered.')
            sysOps.exitProgram()
        
        try:
            if not(inp_mle is None):
                self.s = float(inp_mle.s)
            else:
                self.s = float(self.mleSettings['-s'][0])
        except:
            self.s = 1.0
            
        try:
            if not(inp_mle is None):
                self.print_status = bool(inp_mle.print_status)
            else:
                self.print_status = (self.mleSettings['-print_status'][0] == 'y' or self.mleSettings['-print_status'][0] == 'Y')
        except:
            pass
            
        try:
            if not(inp_mle is None):
                self.err_bound = float(inp_mle.err_bound)
            else:
                self.err_bound = float(self.mleSettings['-err_bound'][0])
        except:
            sysOps.throw_exception('ERROR: No err_bound value entered.')
            sysOps.exitProgram()
            
        if (imagemodule_input_filename is None) and (inp_data is None) and not(inp_mle is None):
            self.uei_data = np.array(inp_mle.uei_data)
            self.Numi = int(inp_mle.Numi)
            self.sum_bcn_assoc = np.array(inp_mle.sum_bcn_assoc)
            self.sum_trg_assoc = np.array(inp_mle.sum_trg_assoc)
            self.sum_bcn_uei = np.array(inp_mle.sum_bcn_uei)
            self.sum_trg_uei = np.array(inp_mle.sum_trg_uei)
            self.sum_self_bcn_uei = np.array(inp_mle.sum_self_bcn_uei)
            self.sum_self_trg_uei = np.array(inp_mle.sum_self_trg_uei)
            self.bcn_amp_factors = np.array(inp_mle.bcn_amp_factors)
            self.trg_amp_factors = np.array(inp_mle.trg_amp_factors)
            self.index_key = np.array(inp_mle.index_key)
            self.max_nontriv_eigenvec_to_calculate = int(inp_mle.max_nontriv_eigenvec_to_calculate)
            self.seq_evecs = csc_matrix(inp_mle.seq_evecs)
            self.backproj_mat = csc_matrix(inp_mle.backproj_mat) # inverse of the inner product between seq_evecs and itself
            self.max_nu = int(inp_mle.max_nu)
            self.glo_indices = np.array(inp_mle.glo_indices)
            self.x_umi_polynom_tuples_buff =  np.array(inp_mle.x_umi_polynom_tuples_buff)
            self.x_umi =  np.array(inp_mle.x_umi)
            self.diag_el =  np.array(inp_mle.diag_el)
            self.norm_el =  np.array(inp_mle.norm_el)
        else:
            self.load_data(imagemodule_input_filename,inp_data) # requires inputted value of Nbcn if inp_data = None
            # if inp_data is not None, self.Nbcn, self.Ntrg, etc will all have values replaced
            
            # Perform up-front calculation of the required max_nu to generate err_bound
            self.max_nu = 5
            while True:
                eps_nu = np.power(2*np.pi*(self.max_nu + 1), -0.5) * np.power(np.e*np.pi /( (self.max_nu + 1)*np.sqrt(self.spat_dims)), self.max_nu + 1) / ( 1 - (np.e*np.pi/((self.max_nu + 1)*np.sqrt(self.spat_dims))))
                if self.print_status:
                    sysOps.throw_status('eps_nu = ' + str(eps_nu) + ', (np.square(1 + eps_nu) - 1.0) = ' + str((np.square(1 + eps_nu) - 1.0)))
                if (np.square(1 + eps_nu) - 1.0) < self.err_bound:
                    break
                self.max_nu += 1
            if self.print_status:
                sysOps.throw_status('Using maximum polynomial order ' + str(self.max_nu) + ' to achieve (1+eps_nu)^2 < 1 + ' +str(self.err_bound) + '. Assembling GLO indices.')
            self.glo_indices = get_glo_indices(self.max_nu,self.spat_dims)
            
            # allocate memory for FFGT calls
            self.x_umi_polynom_tuples_buff = np.zeros([self.Numi,len(self.glo_indices)],dtype=np.double)
            self.x_umi = np.array([]) # class variable reserved for later use
            self.diag_el = np.array([]) # class variable reserved for later use
            self.norm_el = np.array([]) # class variable reserved for later use
            
            try:
                self.max_nontriv_eigenvec_to_calculate = int(self.mleSettings['-max_nontriv_eigenvec_to_calculate'][0])
                self.max_nontriv_eigenvec_to_calculate = min(self.max_nontriv_eigenvec_to_calculate, self.Numi - 1) # Numi-1 rather than Numi, since eigen_decomp calls eigs() with a request for max_nontriv_eigenvec_to_calculate+1 eigenvectors
                if self.print_status:
                    sysOps.throw_status('Assigned self.max_nontriv_eigenvec_to_calculate = ' + str(self.max_nontriv_eigenvec_to_calculate))
            except:
                if self.print_status:
                    sysOps.throw_status('No max_nontriv_eigenvec_to_calculate value entered. Proceeding with non-spectral MLE.')
                self.max_nontriv_eigenvec_to_calculate = None
            
        
    def load_data(self,infilename,inp_data=None):
        # Load raw UEI data from infilename
        # if inp_data != None, use this as UEI data, not necessary indexed from 0 consecutively (this will be taken care of by call to generate_new_subdata())
        
        self.Numi = self.Nbcn + self.Ntrg
        self.index_key = np.arange(self.Numi,dtype=np.int64)
        if not(infilename is None) and type(inp_data) != np.ndarray:
            sysOps.throw_status('Loading data from ' + sysOps.globaldatapath + infilename)
            if not sysOps.check_file_exists(infilename):
                sysOps.throw_status(sysOps.globaldatapath + infilename + ' does not exist. Exiting.')
                sysOps.exitProgram()
                return
            self.uei_data = np.loadtxt(sysOps.globaldatapath + infilename,dtype=np.int64,delimiter=',')
            # Loaded data will contain indices starting at 0 for both beacons and targets.
            # We will therefore ensure these are immediately altered
            if np.max(self.uei_data[:,0])+1 != self.Nbcn or np.max(self.uei_data[:,1])+1 != self.Nbcn + self.Ntrg:
                sysOps.throw_exception('Inputted uei_data is not consistent with inputted Nbcn and/or Ntrg.')
                sysOps.throw_exception('np.max(self.uei_data[:,0])+1 = ' + str(np.max(self.uei_data[:,0])+1) + ', Nbcn = ' + str(self.Nbcn))
                sysOps.throw_exception('np.max(self.uei_data[:,1])+1 = ' + str(np.max(self.uei_data[:,1])+1) + ', Nbcn+Ntrg = ' + str(self.Nbcn+self.Ntrg))
                sysOps.exitProgram()
            
            if np.min(self.uei_data[:,1]) < np.max(self.uei_data[:,0]):
                sysOps.throw_exception('Error: inputted UEI-date from ' + sysOps.globaldatapath + infilename + ' has non-overlapping beacon/target indices.')
                sysOps.exitProgram()

            if self.uei_data.shape[0] != self.Nassoc:
                sysOps.throw_exception('Nassoc error.')
                sysOps.exitProgram()
            
            if np.sum(self.uei_data[:,2]) != self.Nuei:
                sysOps.throw_exception('Nuei error.')
                sysOps.exitProgram()
            
            self.uei_data = self.uei_data[:,:3] # Additional numbers (such as read-count) may be included in output. This extra data is excluded from the data array.
            if self.print_status:
                sysOps.throw_status('Data loaded. Adding UEI counts ..')
            self.sum_bcn_uei = np.zeros(self.Numi,dtype=np.int64)
            self.sum_trg_uei = np.zeros(self.Numi,dtype=np.int64)
            self.sum_bcn_assoc = np.zeros(self.Numi,dtype=np.int64)
            self.sum_trg_assoc = np.zeros(self.Numi,dtype=np.int64)
            
            if type(self.sum_self_bcn_uei) != np.ndarray or type(self.sum_self_trg_uei) != np.ndarray:
                self.sum_self_bcn_uei = np.zeros(self.Numi,dtype=np.int64)
                self.sum_self_trg_uei = np.zeros(self.Numi,dtype=np.int64)
            
            for i in range(self.Nassoc):
                self.sum_bcn_uei[self.uei_data[i,0]] += self.uei_data[i,2]
                self.sum_trg_uei[self.uei_data[i,1]] += self.uei_data[i,2]
                self.sum_bcn_assoc[self.uei_data[i,0]] += 1
                self.sum_trg_assoc[self.uei_data[i,1]] += 1

        elif infilename is None and type(inp_data) == np.ndarray:
            # Inputted data is assumed to contain non-overlapping indices
            if self.print_status:
                sysOps.throw_status('Assigning sub-data.')
            generate_new_subdata(inp_data, self) # # Assumes mutual exclusivity of beacon and target indices in inp_data
        else:
            sysOps.throw_exception('Error: load_data() input pattern not supported.')
            sysOps.exitProgram()
            
        # initiate amplification factors
        valid_bcn_indices = ((self.sum_bcn_uei+self.sum_self_bcn_uei) > 0)
        valid_trg_indices = ((self.sum_trg_uei+self.sum_self_trg_uei) > 0)
        
        min_valid_count = min(np.min(self.sum_bcn_uei[valid_bcn_indices] + self.sum_self_bcn_uei[valid_bcn_indices]),
                              np.min(self.sum_trg_uei[valid_trg_indices] + self.sum_self_trg_uei[valid_trg_indices]))
        
        # all valid amplification factors set to values >=log(2)>0. invalid amplification factors set to 0
        self.bcn_amp_factors = np.zeros(self.Numi,dtype=np.float64)
        self.trg_amp_factors = np.zeros(self.Numi,dtype=np.float64)
        self.bcn_amp_factors[valid_bcn_indices] = np.log((self.sum_bcn_uei[valid_bcn_indices]
                                                          +self.sum_self_bcn_uei[valid_bcn_indices])*float(2.0/min_valid_count))
        self.trg_amp_factors[valid_trg_indices] = np.log((self.sum_trg_uei[valid_trg_indices]
                                                          +self.sum_self_trg_uei[valid_trg_indices])*float(2.0/min_valid_count))
        if self.print_status:
            sysOps.throw_status('Data read-in complete. Found ' + str(np.sum(~valid_bcn_indices)) + ' empty beacon indices and ' + str(np.sum(~valid_trg_indices)) + ' empty target indices.')
        return
        
    def print_res(self,X,outfilename,inp_single_pt_assignments = None, inp_single_pt_mle = None):
        sysOps.throw_status('Printing result to ' + outfilename + '... X.shape = ' + str(X.shape))
        Xumi = np.zeros([self.Numi,self.spat_dims+2],dtype=np.float64)
        Xumi[:,self.spat_dims] = self.bcn_amp_factors
        Xumi[:,self.spat_dims+1] = self.trg_amp_factors
        if not(type(self.seq_evecs) is None):
            Xumi[:,:self.spat_dims] = self.seq_evecs.dot(X)
        else:
            Xumi[:,:self.spat_dims] = np.reshape(X,[self.Numi,self.spat_dims])
        
        if type(inp_single_pt_assignments) == list and type(inp_single_pt_mle) == mleObj:
            # if true single-points have been collapsed to stored UMI positions, expand this data
            final_Xumi = np.zeros([inp_single_pt_assignments.shape[0],self.spat_dims+2],dtype=np.float64)
            for n in range(inp_single_pt_assignments.shape[0]):
                final_Xumi[n,:] = Xumi[inp_single_pt_assignments[n],:]
            ref_mle = inp_single_pt_mle
        else:
            final_Xumi = Xumi
            ref_mle = self
        
        np.savetxt(sysOps.globaldatapath + outfilename, 
                   np.concatenate([np.reshape(ref_mle.sum_bcn_uei,[ref_mle.Numi,1]), 
                                   np.reshape(ref_mle.sum_trg_uei,[ref_mle.Numi, 1]),
                                   np.reshape(ref_mle.index_key,[ref_mle.Numi, 1]), final_Xumi], axis = 1),delimiter = ',')
        sysOps.throw_status('Result printed.')

    def eigen_decomp(self,imagemodule_input_filename):
    # Assemble linear manifold from data using "local linearity" assumption
    # assumes uei_data beacon- and target-indices at this point has non-overlapping indices 
        
        if type(imagemodule_input_filename) != str or (not (sysOps.check_file_exists('evecs_' + imagemodule_input_filename))):
            sysOps.throw_status('Forming row-normalized linear operator before eigen-decomposition ...') 
            
            row_indices = np.arange(self.Numi + 2*self.Nassoc, dtype=np.int64)
            col_indices = np.arange(self.Numi + 2*self.Nassoc, dtype=np.int64)
            norm_uei_data = np.zeros(self.Numi + 2*self.Nassoc, dtype=np.float64)

            get_normalized_sparse_matrix(self.sum_bcn_uei, self.sum_trg_uei,
                                         row_indices,col_indices,
                                         norm_uei_data,self.uei_data,
                                         self.Nassoc,self.Numi,np.False_)
            
            csc_op = csc_matrix((norm_uei_data, (row_indices, col_indices)), (self.Numi, self.Numi))
            
            # Clear extraneous memory usage before eigendecomposition
            del row_indices
            del col_indices
            del norm_uei_data
            
            sysOps.throw_status('Generating ' + str(self.max_nontriv_eigenvec_to_calculate) + '+1 eigenvectors ...') 
            if self.max_nontriv_eigenvec_to_calculate+2 >= self.Numi:
                # require complete eigen-decomposition
                evals_large, evecs_large = LA.eig(csc_op.toarray())
            else:
                evals_large, evecs_large = scipy.sparse.linalg.eigs(csc_op, k=self.max_nontriv_eigenvec_to_calculate+1, M = None, which='LR', v0=None, ncv=None, maxiter=None, tol = 0)
            evals_large = np.real(evals_large) # set to real components
            evecs_large = np.real(evecs_large)
            sysOps.throw_status('Done.') 
            
            # Since power method may not return eigenvectors in correct order, sort
            triv_eig_index = np.argmin(np.var(evecs_large,axis = 0))
            top_nontriv_indices = np.where(np.arange(evecs_large.shape[1]) != triv_eig_index)[0]
            # remove trivial (translational) eigenvector
            evals_large = evals_large[top_nontriv_indices]
            evecs_large = evecs_large[:,top_nontriv_indices]
            eval_order = np.argsort(np.abs(evals_large))
            evals_small = evals_large[eval_order[:self.max_nontriv_eigenvec_to_calculate]]
            evecs_small = evecs_large[:,eval_order[:self.max_nontriv_eigenvec_to_calculate]]

            if type(imagemodule_input_filename) == str:
                sysOps.throw_status('Printing linear manifold with non-trivial eigenvalues.')
                with open(sysOps.globaldatapath + 'evals_' + imagemodule_input_filename,'w') as evals_file:
                    evals_file.write(','.join([str(v) for v in evals_small]) + '\n')
                
                # write eigenvectors as rows in output file
                np.savetxt(sysOps.globaldatapath + 'evecs_' + imagemodule_input_filename, 
                           np.transpose(evecs_small[:,:self.max_nontriv_eigenvec_to_calculate]),fmt='%.10e',delimiter=',')
                self.seq_evecs = np.transpose(evecs_small)
            else:
                sysOps.throw_status('Assigning top ' + str(self.max_nontriv_eigenvec_to_calculate) + ' eigenvectors to manifold.')
                self.seq_evecs = np.array(np.transpose(evecs_small))
        else:
            sysOps.throw_status('Eigen-decomposition files found pre-computed.')
            self.load_manifold(imagemodule_input_filename)

        return
    
    
    def load_manifold(self, imagemodule_input_filename, max_eig = None):
        # load pre-computed eigenvectors and eigenvalues from files
        evals = list()
        with open(sysOps.globaldatapath + 'evals_' + imagemodule_input_filename,'rU') as eval_file:
            for eval_line in eval_file:
                evals.extend([float(val) for val in eval_line.strip('\n').split(',')])
        if type(max_eig) == int and len(evals) > max_eig:
            evals = evals[:max_eig]
    
        self.seq_evecs = np.loadtxt(sysOps.globaldatapath + 'evecs_' + imagemodule_input_filename,dtype=np.float64,comments='#',delimiter=',')
        if type(max_eig) == int and self.seq_evecs.shape[0] > max_eig:
            sysOps.throw_status('Curtailing manifold read-in at maximum eigenvector count ' + str(max_eig))
            self.seq_evecs = self.seq_evecs[:max_eig,:]

        return evals
        
    
    def SL_cluster(self):
        # SL_cluster returns disjoint data-sets (single-linkage clustering based on the presence/absence of associations in the array self.uei_data
        
        index_link_array = np.arange(self.Numi,dtype=np.int64)
        
        min_contig_edges(index_link_array,np.ones(self.Numi,dtype=np.int64),self.uei_data,self.Nassoc)
        
        prelim_cluster_list = [list() for i in range(self.Numi)]
        
        for n in range(self.Numi):
            prelim_cluster_list[index_link_array[n]].append(int(n))
            
        sysOps.throw_status('Completed SL-clustering.')
        
        return [list(sub_list) for sub_list in prelim_cluster_list if len(sub_list)>0]
                
    def reduce_to_largest_linkage_cluster(self,assoc_retained = None):
        # Prune data set based on which contiguous set is largest
        # uei_data assumed to have non-overlapping indices for each distinct UMI
        
        init_uei = int(self.Nuei)
        init_assoc = int(self.Nassoc)
        
        if type(assoc_retained)==np.ndarray: # if boolean array containing associations to be retained, is inputted, confine linkage analysis to these exclusively
            self.uei_data = self.uei_data[assoc_retained,:]
            self.Nassoc = self.uei_data.shape[0]
            self.Nuei = np.sum(self.uei_data[:,2])
        
        sysOps.throw_status('Performing single-linkage clustering on UEI-data in preparation for MLE.')
        contig_sets = self.SL_cluster()
        sysOps.throw_status('Single-linkage clustering complete.')
        contig_sizes = [len(contig_sets[i]) for i in range(len(contig_sets))]
        contig_set = contig_sets[contig_sizes.index(max(contig_sizes))]
        contig_set.sort()
        
        #CHECKING FOR REDUNDANCIES
        for i in range(1,len(contig_set)):
            if contig_set[i] == contig_set[i-1]:
                sysOps.throw_exception('SL-cluster error.')
                sysOps.exitProgram()
                
        # contig_set values correspond to those currently in uei_data (columns 0 and 1) and enumerated as 0...self.Numi-1
        # index_key indices correspond to these same indices, and their elements correspond to ORIGINAL indices
        self.index_key = np.array(self.index_key[contig_set],dtype=np.int64) # update index_key

        # self.index_key indices now correspond to NEW indices ordered in contig_set, elements remain ORIGINAL indices
        
        # new_ind_lookup will have indices corresponding to PREVIOUS indices (note that self.Numi has not yet been updated)
        new_ind_lookup = -np.ones(self.Numi,dtype=np.int64)
        
        # initiate umi_retained as storing the do/do-not retain status of previous index set (corresponding to elements of contig_set)
        umi_retained = np.zeros(self.Numi,dtype=np.bool_)
        
        for i in range(len(contig_set)):
            new_ind_lookup[contig_set[i]] = int(i)
            umi_retained[contig_set[i]] = np.True_

        # bcn_new_ind_lookup and trg_new_ind_lookup now have elements corresponding to NEW indices (ordered in contig_sets[0] and contig_sets[1]) 
        
        sysOps.throw_status('Retained ' + str(np.sum(self.index_key < self.Nbcn)) + '/' + str(self.Nbcn) + ' beacon UMIs, ' + str(np.sum(self.index_key >= self.Nbcn)) + '/' + str(self.Ntrg) + ' target UMIs.')
        
        self.Numi = len(contig_set)
        orig_nbcn = int(self.Nbcn)
        self.Nbcn = np.sum(self.index_key < orig_nbcn)
        self.Ntrg = np.sum(self.index_key >= orig_nbcn)
        self.x_umi_polynom_tuples_buff = np.zeros([self.Numi,len(self.glo_indices)],dtype=np.double)
        
        uei_retained = np.zeros(self.Nassoc,dtype=np.bool_)
        
        for i in range(self.Nassoc):
            if umi_retained[self.uei_data[i,0]] and umi_retained[self.uei_data[i,1]]:
                uei_retained[i] = np.True_
            elif umi_retained[self.uei_data[i,0]] or umi_retained[self.uei_data[i,1]]:
                sysOps.throw_exception('ERROR: SL cluster inconsistency -- connectivity without co-partitioning.')
                sysOps.exitProgram()
        
        self.uei_data = self.uei_data[uei_retained,:]
        self.sum_self_bcn_uei = self.sum_self_bcn_uei[umi_retained]
        self.sum_self_trg_uei = self.sum_self_trg_uei[umi_retained]
        
        sysOps.throw_status('Retained ' + str(np.sum(self.uei_data[:,2])) + '/' + str(init_uei) + ' UEIs, ' + str(self.uei_data.shape[0]) + '/' + str(init_assoc) + ' associations.')
        
        self.sum_bcn_uei = np.zeros(self.Numi,dtype=np.int64)
        self.sum_trg_uei = np.zeros(self.Numi,dtype=np.int64)
        self.sum_bcn_assoc = np.zeros(self.Numi,dtype=np.int64)
        self.sum_trg_assoc = np.zeros(self.Numi,dtype=np.int64)
        self.Nuei = np.sum(self.uei_data[:,2])
        self.Nassoc = int(self.uei_data.shape[0])
        
        # replace uei_data indices
        for i in range(self.Nassoc):
            self.uei_data[i,0] = new_ind_lookup[self.uei_data[i,0]]
            self.uei_data[i,1] = new_ind_lookup[self.uei_data[i,1]]
            
        for i in range(self.Nassoc):
            self.sum_bcn_uei[self.uei_data[i,0]] += self.uei_data[i,2]
            self.sum_trg_uei[self.uei_data[i,1]] += self.uei_data[i,2]
            self.sum_bcn_assoc[self.uei_data[i,0]] += 1
            self.sum_trg_assoc[self.uei_data[i,1]] += 1
        
        # initiate amplification factors
        valid_bcn_indices = ((self.sum_bcn_uei+self.sum_self_bcn_uei) > 0)
        valid_trg_indices = ((self.sum_trg_uei+self.sum_self_trg_uei) > 0)
        
        min_valid_count = min(np.min(self.sum_bcn_uei[valid_bcn_indices]+self.sum_self_bcn_uei[valid_bcn_indices]),
                              np.min(self.sum_trg_uei[valid_trg_indices]+self.sum_self_trg_uei[valid_trg_indices]))
        
        # all valid amplification factors set to values >=log(2)>0. invalid amplification factors set to 0
        self.bcn_amp_factors = np.zeros(self.Numi,dtype=np.float64)
        self.trg_amp_factors = np.zeros(self.Numi,dtype=np.float64)
        self.bcn_amp_factors[valid_bcn_indices] = np.log((self.sum_bcn_uei[valid_bcn_indices]
                                                          +self.sum_self_bcn_uei[valid_bcn_indices])*float(2.0/min_valid_count))
        self.trg_amp_factors[valid_trg_indices] = np.log((self.sum_trg_uei[valid_trg_indices]
                                                          +self.sum_self_trg_uei[valid_trg_indices])*float(2.0/min_valid_count))
        
        return
        
            
    def get_ffgt_args(self, Xumi):
        # Returns parameters for FFGT function call
        # Calculations of error-bounds in forthcoming preprint
        
        max_width = 0
        min_x = np.zeros(self.spat_dims,dtype=np.double)
        for d in range(self.spat_dims):
            min_x[d] = np.min(Xumi[:,d])
            max_width = max(max_width,np.max(Xumi[:,d]) - min_x[d])
        
        max_width = max(max_width,np.sqrt(self.s)) # in case max_width = 0
            
        eps_nu = np.power(np.e*np.pi/(np.sqrt(self.spat_dims)*(self.max_nu+1)),
                          self.max_nu+1)*np.power(1.0 - (np.e*np.pi/(np.sqrt(self.spat_dims)*(self.max_nu+1))),-1.0)/np.sqrt(2*np.pi*(self.max_nu + 1.0))
        
        this_err_bound = float(self.err_bound)
        this_err_bound -= 2*eps_nu # removing this contribution from allowed error to be filled by remaining two sources
        
        # 2 sources of error that are independent of max_nu polynomial order:
        #    1. Finiteness of L (eps_L)
        #    2. Finiteness of Q (eps_Q) 
        
        # determine error from finiteness of L
        L = max_width*2.0
        eps_L = 2*self.spat_dims*max(self.Nbcn,self.Ntrg)*(self.s**(-self.spat_dims/2.0))*np.exp(-(L*L)/(4*self.s))/(1.0 - np.exp(-(2*L*L)/self.s))
        if self.prev_min_bcn_sumw != None and self.prev_min_trg_sumw != None:
            denom = 1.0
            if self.rel_err_bound:
                denom = min(self.prev_min_bcn_sumw, self.prev_min_trg_sumw)
            while eps_L/denom > this_err_bound/2:
                L *= 1.1
                eps_L = 2*self.spat_dims*max(self.Nbcn,self.Ntrg)*(self.s**(-self.spat_dims/2.0))*np.exp(-(L*L)/(4*self.s))/(1.0 - np.exp(-(2*L*L)/self.s))
    
        # determine error from finitenss of Q
        if self.prev_Q == None:
            min_prev_sum_axial_fft = (np.sqrt(self.Nbcn/2.0) + np.sqrt(self.Ntrg/2.0))
            self.prev_Q = np.ceil(2.0*L/np.sqrt(self.s))
            self.prev_L = np.double(L)
        else:
            bcn_denom = 1.0
            trg_denom = 1.0
            if self.rel_err_bound:
                bcn_denom = self.prev_min_bcn_sumw
                trg_denom = self.prev_min_trg_sumw
            min_prev_sum_axial_fft = np.minimum(np.sum(np.abs(np.real(self.prev_axial_bcn_fft))
                                                              + np.abs(np.imag(self.prev_axial_bcn_fft)), axis = 0)/trg_denom,
                                                   np.sum(np.abs(np.real(self.prev_axial_trg_fft))
                                                              + np.abs(np.imag(self.prev_axial_trg_fft)), axis = 0)/bcn_denom)
            min_prev_sum_axial_fft = np.min(min_prev_sum_axial_fft, axis = 0) # take minimum across dimensions
            
        eps_prev_Q_array = np.multiply(np.multiply(
                                               np.exp(-np.square(np.pi*np.arange(1,int(self.prev_Q)+1)/self.prev_L)*self.s,dtype=np.double),
                                               np.power(1.0 - np.exp(-2*np.arange(1,int(self.prev_Q)+1,dtype=np.double)*self.s*np.square(np.pi/self.prev_L)), -1.0)),
                               (4*np.sqrt(np.pi * (self.s ** (self.spat_dims - 1)))/self.prev_L)*min_prev_sum_axial_fft)
        
        first_acceptable_index = 1
        for eps in eps_prev_Q_array:
            if eps < self.err_bound/2.0:
                break
            first_acceptable_index += 1
        
        Q = np.ceil(first_acceptable_index*L/self.prev_L)
        
        return L, Q, min_x
            
    def calc_grad(self, X):
        # FUnction is called on 1D array of EITHER positions OR eigenvector coefficients (which it is will be known to function based on whether self.seq_evecs is None)
        
        # uei_data is a (Nassoc x 3) numpy array
        # self.prev_axial_fft should be spat_dims x (2 x self.prev_Q + 1)
        Xumi = np.zeros([self.Numi,self.spat_dims+2],dtype=np.float64)
              
        # first Numi elements of X will always be amplification factors 
        if not(self.seq_evecs is None):
            Xumi[:,:self.spat_dims] = self.seq_evecs.transpose().dot(np.reshape(X,[self.seq_evecs.shape[0],self.spat_dims]))
        else:
            # if MLE is not being calculated as spectral projection, then:
            #     first Numi elements of X will correspond to spatial dimension 0
            #     second Numi elements of X will correspond to spatial dimension 1 ... etc
            Xumi[:,:self.spat_dims] = np.reshape(X,[self.Numi,self.spat_dims])
        
        Xumi[:,self.spat_dims] = self.bcn_amp_factors
        Xumi[:,self.spat_dims + 1] = self.trg_amp_factors
        ffgt_internal_count = 0
        corrected_gradients = 0
        
        # Here, we apply the criterion that outlier points (lying >5 x sigma from mean position) are not included in the FFGT calculation
         
        mean_pos = np.zeros(self.spat_dims)
        mean_sq_pos = np.zeros(self.spat_dims)            
        
        umi_incl_ffgt = np.zeros(self.Numi,dtype=np.bool_) # initiate as boolean array
        
        tot_outliers = get_non_outliers(umi_incl_ffgt,Xumi[:,:self.spat_dims],mean_pos,mean_sq_pos,self.Numi,self.spat_dims,5.0)
        umi_incl_ffgt = np.arange(self.Numi,dtype=np.int64)[umi_incl_ffgt] # convert to index array
        
        default_max_Q = 100.0
        has_bcn_arr = ((self.sum_bcn_uei[umi_incl_ffgt]+self.sum_self_bcn_uei[umi_incl_ffgt]) > 0) # crucially, has_bcn_arr and has_trg_arr are boolean arrays referring to the sub-set determined by umi_incl_ffgt
        has_trg_arr = ((self.sum_trg_uei[umi_incl_ffgt]+self.sum_self_trg_uei[umi_incl_ffgt]) > 0)
        
        while True:
            # This loop exists to ensure that values of L and Q are appropriate --> if not, will update accordingly
            
            L, Q, min_x = self.get_ffgt_args(Xumi[umi_incl_ffgt,:])
            
            if ffgt_internal_count == 0:
                Q = np.ceil(1.1*Q) # being conservative with initial estimate
                if self.print_status:
                    sysOps.throw_status('Calling FFGT with L = ' + str(L) + ', Q = ' + str(Q) + ', max_nu = ' + str(self.max_nu))
            elif (ffgt_internal_count > 0 and L <= self.prev_L and Q <= self.prev_Q) or self.prev_Q == default_max_Q:
                if self.print_status:
                    if corrected_gradients > 0:
                        sysOps.throw_status('Corrected ' + str(corrected_gradients) + ' gradients.')
                    sysOps.throw_status(str(tot_outliers) + ' outliers excluded from partition function.')
                break
            elif ffgt_internal_count > 0:
                Q = np.ceil(1.2*self.prev_Q)
                if self.print_status:
                    sysOps.throw_status('On sub-iteration ' + str(ffgt_internal_count) + ' calling with L = ' + str(L) + ', Q = ' + str(Q) + '. ' + str(tot_outliers) + ' outliers excluded from partition function.')
            if Q>default_max_Q:
                if self.print_status:
                    sysOps.throw_status('Over-sized memory requested. Setting Q to max value of ' + str(default_max_Q) + '.')
                Q = float(default_max_Q)

            (min_exp_amp_bcn, min_exp_amp_trg, 
             self.prev_axial_bcn_fft, self.prev_axial_trg_fft, 
             Xbcn_grad_sub, Xtrg_grad_sub) = call_ffgt(Xumi[umi_incl_ffgt,:],
                                                     has_bcn_arr,has_trg_arr,
                                                     self.x_umi_polynom_tuples_buff[umi_incl_ffgt,:],
                                                     min_x,self.glo_indices,L,Q,self.s,self.max_nu,self.spat_dims,True,True)
             
            self.ffgt_call_count += 1
            # Xbcn_grad_sub and Xtrg_grad_sub will both be of size Numi x (spat_dims+1) with the final column containing the gradient with respect to the amplification factor
            
            # assign internal variables for next iteration
            self.prev_Q = float(Q)
            self.prev_L = float(L)
            self.prev_min_bcn_sumw = np.min(Xbcn_grad_sub[has_bcn_arr,self.spat_dims])
            self.prev_min_trg_sumw = np.min(Xtrg_grad_sub[has_trg_arr,self.spat_dims])
            
            min_relevant_weight = min(min_exp_amp_bcn,min_exp_amp_trg) # minimum acceptable weight is assigned to the minimum exponentiated amplification factor 
                                                                       # (on the order of the absolute minimum what we'd expect for any point that is in communication with some other arbitrarily chosen point)
            
            if self.prev_min_bcn_sumw < min_relevant_weight or self.prev_min_trg_sumw < min_relevant_weight:
                corrected_gradients = 0
                
                bcn_corr = np.where(Xbcn_grad_sub[has_bcn_arr,self.spat_dims] < min_relevant_weight)[0]
                corrected_gradients += len(bcn_corr)
                Xbcn_grad_sub[has_bcn_arr,:][bcn_corr,:] = np.multiply(Xbcn_grad_sub[bcn_corr,:],np.random.uniform(-1.0,1.0,[len(bcn_corr),self.spat_dims+1])) 
                # direction of "repulsive" force randomized, since given its extremely small magnitude, it is error-prone
                self.prev_min_bcn_sumw = float(min_relevant_weight) # this memory will be retained for the next time get_ffgt_args() is called
                
                trg_corr = np.where(Xtrg_grad_sub[has_trg_arr,self.spat_dims] < min_relevant_weight)[0]
                corrected_gradients += len(trg_corr)
                Xtrg_grad_sub[has_trg_arr,:][trg_corr,:] = np.multiply(Xtrg_grad_sub[trg_corr,:],np.random.uniform(-1.0,1.0,[len(trg_corr),self.spat_dims+1])) 
                # direction of "repulsive" force randomized, since given its extremely small magnitude, it is error-prone
                self.prev_min_trg_sumw = float(min_relevant_weight)
            
            ffgt_internal_count += 1
            
        sumw_bcn = np.sum(Xbcn_grad_sub[has_bcn_arr,self.spat_dims]) 
        sumw_trg = np.sum(Xtrg_grad_sub[has_trg_arr,self.spat_dims])
        
        if (np.abs(sumw_bcn - sumw_trg)/min(sumw_bcn,sumw_trg)) > self.err_bound:
            sysOps.throw_exception('ERROR: weight-sum inconsistency, (' + str(sumw_bcn) + ',' + str(sumw_trg) + ')')
            sysOps.exitProgram()
        self.sumw = (sumw_bcn + sumw_trg)/2.0            
        
        dXbcn = np.zeros([self.Numi,self.spat_dims],dtype=np.float64)
        dXtrg = np.zeros([self.Numi,self.spat_dims],dtype=np.float64)
        dXbcn[umi_incl_ffgt,:] = np.multiply(-(self.Nuei/self.sumw),Xbcn_grad_sub[:,:self.spat_dims])
        dXtrg[umi_incl_ffgt,:] = np.multiply(-(self.Nuei/self.sumw),Xtrg_grad_sub[:,:self.spat_dims])
        
        # add convex (UEI-data) portion of gradients
        
        log_likelihood = add_convex_components(Xumi,dXbcn,dXtrg,self.uei_data,self.sumw,self.Nuei,self.Nassoc,self.s,self.spat_dims)
            
        if self.print_status:
            sysOps.throw_status('Log-likelihood = ' + str(log_likelihood))
        
        if not(self.seq_evecs is None):
            evec_num = self.seq_evecs.shape[0]
            dX = np.zeros([evec_num,self.spat_dims],dtype=np.float64)
            dX = self.backproj_mat.transpose().dot(np.add(dXbcn,dXtrg))
        else:
            evec_num = self.Numi
            dX = np.add(dXbcn, dXtrg)
        
        # since returned objective function is being minimized, return opposite sign
        return -log_likelihood, -np.reshape(dX,evec_num*self.spat_dims)
    
@jit("int64(bool_[:],float64[:,:],float64[:],float64[:],int64,int64,float64)",nopython=True)
def get_non_outliers(umi_incl_ffgt,Xumi,mean_pos,mean_sq_pos,Numi,spat_dims,fold_value):
    
    # Function is exception handler -- in cases where a point has been ejected for some reason far from the main point cloud, this function prevents
    #     FFGT calls from wasting time on these points
    # Elements of umi_incl_ffgt are switched to True only if they lie within fold_value*np.sqrt(sigma) -- where sigma is the standard deviation of all points -- of the mean position
    
    for d in range(spat_dims):
        mean_pos[d] = 0.0
        mean_sq_pos[d] = 0.0
        
    for n in range(Numi):
        for d in range(spat_dims):
            mean_pos[d] += Xumi[n,d]
            mean_sq_pos[d] += Xumi[n,d]*Xumi[n,d]
    
    sigma = 0.0
    for d in range(spat_dims):
        mean_pos[d] /= Numi
        mean_sq_pos[d] /= Numi
        sigma += mean_sq_pos[d] - mean_pos[d]*mean_pos[d]
        
    sum_sq_pos = 0.0
    tot_outliers = 0
    outlier_dist = fold_value*np.sqrt(sigma)
    for n in range(Numi):
        sum_sq_pos = 0.0
        for d in range(spat_dims):
            sum_sq_pos += (Xumi[n,d] - mean_pos[d])*(Xumi[n,d] - mean_pos[d])
        if np.sqrt(sum_sq_pos) <= outlier_dist:
            umi_incl_ffgt[n] = True
        else:
            tot_outliers += 1
    
    return tot_outliers

@jit("void(int64[:],int64[:],int64[:],int64[:],float64[:],int64[:,:],int64,int64,bool_)",nopython=True)
def get_normalized_sparse_matrix(sum_bcn_uei,sum_trg_uei,
                                 row_indices,col_indices,norm_uei_data,uei_data,
                                 Nassoc,Numi,symm):
    # Output of get_normalized_sparse_matrix() depends on the presence of overlapping index-identities between beacon and targets
    # This is determined by bcn_trg_same_indices flag which indicates if indices in first and second columns of uei_data are referring to the same points 
    
    for n in range(Numi): # add in on-diagonal entries
        row_indices[n] = n
        col_indices[n] = n
        norm_uei_data[n] = -1.0
    
    for i in range(Nassoc): # add in off-diagonal entries
        row_indices[Numi + (2*i)] = int(uei_data[i,0])
        col_indices[Numi + (2*i)] = int(uei_data[i,1])
        row_indices[Numi + (2*i + 1)] = int(uei_data[i,1])
        col_indices[Numi + (2*i + 1)] = int(uei_data[i,0])
        
        if symm: # normalize as symmetric graph-laplacian: product of square roots of column-sum and row-sum
            normfactor = np.sqrt(float((sum_bcn_uei[uei_data[i,0]]+sum_trg_uei[uei_data[i,0]])*(sum_bcn_uei[uei_data[i,1]]+sum_trg_uei[uei_data[i,1]])))
            norm_uei_data[Numi + (2*i)] = uei_data[i,2]/normfactor
            norm_uei_data[Numi + (2*i + 1)] = uei_data[i,2]/normfactor
        else: # non-symmetrix graph-laplacian: normalize only by row-sum
            norm_uei_data[Numi + (2*i)] = uei_data[i,2]/float(sum_bcn_uei[uei_data[i,0]]+sum_trg_uei[uei_data[i,0]])
            norm_uei_data[Numi + (2*i + 1)] = uei_data[i,2]/float(sum_bcn_uei[uei_data[i,1]]+sum_trg_uei[uei_data[i,1]])
        
@jit("void(int64[:],int64[:],int64[:,:],int64)",nopython=True)
def min_contig_edges(index_link_array,dataset_index_array,uei_data,Nassoc):
    # Function is used for single-linkage clustering of UMIs (to identify which sets are contiguous and which are not)
    # Inputs:
    #    1. index_link_array: indices for individual UMIs
    #    2. dataset_index_array: belonging to the same set is a requirement for two UMIs to be examined for linkage -- subsets of the data that have different values in dataset_index_array will not be merged
     
    min_index_links_changed = 1 # initiate flag to enter while-loop
    
    while min_index_links_changed > 0:
        min_index_links_changed = 0
        for i in range(Nassoc):
            if dataset_index_array[uei_data[i,0]] == dataset_index_array[uei_data[i,1]]:
                if index_link_array[uei_data[i,0]] > index_link_array[uei_data[i,1]]:
                    index_link_array[uei_data[i,0]] = index_link_array[uei_data[i,1]]
                    min_index_links_changed += 1
                if index_link_array[uei_data[i,1]] > index_link_array[uei_data[i,0]]:
                    index_link_array[uei_data[i,1]] = index_link_array[uei_data[i,0]]
                    min_index_links_changed += 1
                
    return

@jit("float64(float64[:,:],float64[:,:],float64[:,:],int64[:,:],float64,int64,int64,float64,int64)",nopython=True)
def add_convex_components(Xumi,dXbcn,dXtrg,uei_data,sumw,Nuei,Nassoc,s,spat_dims):
    # Function adds sum of squared differences to likelihood, and their gradient to the total likelihood gradient
    log_likelihood = -Nuei*np.log(sumw)
    for i in range(Nassoc):
        pos_diff = 0.0
        sum_sq = 0.0
        for d in range(spat_dims):
            pos_diff = Xumi[uei_data[i,0],d] - Xumi[uei_data[i,1],d]
            dXbcn[uei_data[i,0],d] += -2*uei_data[i,2]*pos_diff/s
            dXtrg[uei_data[i,1],d] += +2*uei_data[i,2]*pos_diff/s
            sum_sq += pos_diff*pos_diff
            
        log_likelihood += uei_data[i,2]*(-sum_sq/s + Xumi[uei_data[i,0],spat_dims] + Xumi[uei_data[i,1],spat_dims+1])
        
    return log_likelihood

@jit("void(float64[:,:],float64[:,:],float64[:],float64[:],int64[:,:],float64[:],int64)",nopython=True)
def add_xtuples_to_grid2d(bcn_spat_field,trg_spat_field,bcn_amp_factors,trg_amp_factors,x_umi_grid_indices,x_polynom_tuples,Numi):
    for n in range(Numi):
        bcn_spat_field[x_umi_grid_indices[n,0],x_umi_grid_indices[n,1]] += bcn_amp_factors[n]*x_polynom_tuples[n]
        trg_spat_field[x_umi_grid_indices[n,0],x_umi_grid_indices[n,1]] += trg_amp_factors[n]*x_polynom_tuples[n]

@jit("void(float64[:],float64[:],float64[:,:],float64[:,:],float64[:],float64[:],int64[:,:],float64[:],int64)",nopython=True)
def add_xtuples_to_vector2d(bcn_x_grad,trg_x_grad,bcn_spat_field,trg_spat_field,bcn_amp_factors,trg_amp_factors,x_umi_grid_indices,x_polynom_tuples,Numi):
    for n in range(Numi):
        bcn_x_grad[n] += x_polynom_tuples[n]*bcn_amp_factors[n]*trg_spat_field[x_umi_grid_indices[n,0],x_umi_grid_indices[n,1]]
        trg_x_grad[n] += x_polynom_tuples[n]*trg_amp_factors[n]*bcn_spat_field[x_umi_grid_indices[n,0],x_umi_grid_indices[n,1]]

def obj_glo_loop(q_polynom_tuples, 
                 bcn_amp_factors, trg_amp_factors,
                 bcn_spat_field, trg_spat_field, 
                 bcn_freq_field, trg_freq_field, Numi, x_umi_polynom_tuples, x_umi_vec_from_ctrs, x_umi_grid_indices, q_grid, glo_indices, L, max_nu, spat_dims):
    diag_term_indices = np.zeros(spat_dims,dtype=np.int64)
    
    bcn_spat_field[:] = 0.0
    trg_spat_field[:] = 0.0
    
    if spat_dims == 2:
        add_xtuples_to_grid2d(bcn_spat_field.real,trg_spat_field.real,bcn_amp_factors,trg_amp_factors,x_umi_grid_indices,x_umi_polynom_tuples[:,0],Numi)
        bcn_freq_field += np.fft.fft2(bcn_spat_field)
        trg_freq_field += np.fft.fft2(trg_spat_field)
    else:
        sysOps.exitProgram()
        
    prev_order_final_index = 0
 
    on_glo_index = 1
    for i in range(1,max_nu+1):
        for d in range(spat_dims):
            tmp_diag_index = int(on_glo_index)
            for j in range(diag_term_indices[d],prev_order_final_index+1):
                x_umi_polynom_tuples[:,on_glo_index] = np.multiply(x_umi_polynom_tuples[:,j],-x_umi_vec_from_ctrs[:,d])
                bcn_spat_field[:] = 0.0
                trg_spat_field[:] = 0.0
                if spat_dims == 2:
                    q_polynom_tuples[:,:,on_glo_index] = np.multiply(q_polynom_tuples[:,:,j],np.multiply(q_grid[d],1j*2*np.pi/(L*glo_indices[on_glo_index,d])))
                    add_xtuples_to_grid2d(bcn_spat_field.real,trg_spat_field.real,bcn_amp_factors,trg_amp_factors,x_umi_grid_indices,x_umi_polynom_tuples[:,on_glo_index],Numi)
                    bcn_freq_field += np.multiply(np.fft.fft2(bcn_spat_field),q_polynom_tuples[:,:,on_glo_index])
                    trg_freq_field += np.multiply(np.fft.fft2(trg_spat_field),q_polynom_tuples[:,:,on_glo_index])
                else:
                    sysOps.exitProgram()
                
                on_glo_index+=1
            
            diag_term_indices[d] = int(tmp_diag_index)
        
        prev_order_final_index = on_glo_index-1
  
def src_glo_loop(bcn_umi_grad, trg_umi_grad, 
                 bcn_amp_factors,trg_amp_factors,
                 q_polynom_tuples, grad_factor_field, bcn_spat_field, trg_spat_field, bcn_freq_field, trg_freq_field, Numi, x_umi_polynom_tuples_buff, x_umi_vec_from_ctrs, x_umi_grid_indices, q_grid, glo_indices, L, max_nu, spat_dims, positional_grad = True):
     
    diag_term_indices = np.zeros(spat_dims,dtype=np.int64)
    
    if spat_dims == 2:
        if positional_grad:
            for d2 in range(spat_dims):
                add_xtuples_to_vector2d(bcn_umi_grad[:,d2],trg_umi_grad[:,d2],
                                        (np.fft.ifft2(np.multiply(grad_factor_field[d2],np.multiply(bcn_freq_field,q_polynom_tuples[:,:,0])))).real,
                                        (np.fft.ifft2(np.multiply(grad_factor_field[d2],np.multiply(trg_freq_field,q_polynom_tuples[:,:,0])))).real,
                                        bcn_amp_factors,trg_amp_factors, x_umi_grid_indices,x_umi_polynom_tuples_buff[:,0],Numi)
           

        add_xtuples_to_vector2d(bcn_umi_grad[:,spat_dims],trg_umi_grad[:,spat_dims],
                                (np.fft.ifft2(np.multiply(bcn_freq_field,q_polynom_tuples[:,:,0]))).real,
                                (np.fft.ifft2(np.multiply(trg_freq_field,q_polynom_tuples[:,:,0]))).real,
                                bcn_amp_factors,trg_amp_factors, x_umi_grid_indices,x_umi_polynom_tuples_buff[:,0],Numi)
            
    else:
        sysOps.exitProgram()
        
    prev_order_final_index = 0
 
    on_glo_index = 1
    for i in range(1,max_nu+1):
        for d in range(spat_dims):
            tmp_diag_index = int(on_glo_index)
            for j in range(diag_term_indices[d],prev_order_final_index+1):
                x_umi_polynom_tuples_buff[:,on_glo_index] = np.multiply(x_umi_polynom_tuples_buff[:,j],x_umi_vec_from_ctrs[:,d])
            
                if spat_dims == 2:   
                    if positional_grad:
                        for d2 in range(spat_dims):
                            add_xtuples_to_vector2d(bcn_umi_grad[:,d2],trg_umi_grad[:,d2],
                                                    (np.fft.ifft2(np.multiply(grad_factor_field[d2],np.multiply(bcn_freq_field,q_polynom_tuples[:,:,on_glo_index])))).real,
                                                    (np.fft.ifft2(np.multiply(grad_factor_field[d2],np.multiply(trg_freq_field,q_polynom_tuples[:,:,on_glo_index])))).real,
                                                    bcn_amp_factors,trg_amp_factors, x_umi_grid_indices,x_umi_polynom_tuples_buff[:,on_glo_index],Numi)
                            
                    add_xtuples_to_vector2d(bcn_umi_grad[:,spat_dims],trg_umi_grad[:,spat_dims],
                                            (np.fft.ifft2(np.multiply(bcn_freq_field,q_polynom_tuples[:,:,on_glo_index]))).real,
                                            (np.fft.ifft2(np.multiply(trg_freq_field,q_polynom_tuples[:,:,on_glo_index]))).real,
                                            bcn_amp_factors,trg_amp_factors, x_umi_grid_indices,x_umi_polynom_tuples_buff[:,on_glo_index],Numi)
                        
                else:
                    sysOps.exitProgram()
                    
                on_glo_index+=1
            
            diag_term_indices[d] = int(tmp_diag_index)
        
        prev_order_final_index = on_glo_index-1

def call_ffgt(x_umi,has_bcn_arr,has_trg_arr,x_umi_polynom_tuples_buff,min_x,glo_indices,L,Q,s,max_nu,spat_dims,
              do_exponentiate=True,positional_grad=True):
    # variables:
    #    Q,r
    #    function call requires x_bcn and x_trg are:
    #        N_bcn x spat_dims+1 and N_trg x spat_dims+1 double arrays
    # Will return only real part of Gaussian sum
        
    x_umi = np.array(x_umi) #make copy locally
    # exponentiate amplification factors
    if do_exponentiate:
        x_umi[has_bcn_arr,spat_dims] = np.exp(x_umi[has_bcn_arr,spat_dims])
        x_umi[has_trg_arr,spat_dims+1] = np.exp(x_umi[has_trg_arr,spat_dims+1])
    x_umi[~has_bcn_arr,spat_dims] = 0.0
    x_umi[~has_trg_arr,spat_dims+1] = 0.0
    
    Numi = x_umi.shape[0]
    
    tot_fgtcells_side = 2*int(Q) + 1
    sector_width = L/np.double(tot_fgtcells_side)
    
    delta = np.zeros(spat_dims,dtype=np.float64)
    # randomize window orientation to eliminate error bias
    for d in range(spat_dims):
        delta[d] = -min(sector_width,(L/4.0))*np.random.rand()
        x_umi[:,d] += (-min_x[d] + (L/4.0) + delta[d]) # changed 12/15/17
    
    x_umi_grid_indices = np.int64(np.round(np.divide(x_umi[:,:spat_dims],sector_width)))
    
    num_terms = len(glo_indices) # number of polynomial-tuples
    
    bcn_spat_field = np.zeros(list([tot_fgtcells_side])*spat_dims,dtype=np.complex)
    trg_spat_field = np.zeros(list([tot_fgtcells_side])*spat_dims,dtype=np.complex)
    bcn_freq_field = np.zeros(list([tot_fgtcells_side])*spat_dims,dtype=np.complex)
    trg_freq_field = np.zeros(list([tot_fgtcells_side])*spat_dims,dtype=np.complex)
    
    gridside_size_vec = list([tot_fgtcells_side])*spat_dims
    gridside_size_vec.append(num_terms)
    gridside_size_vec = np.array(gridside_size_vec,dtype=np.int64)
    
    # In the below, the function ifftshift is used to bring q-indices in line with what the output of fft will be later
    if spat_dims == 2:
        q_grid = np.fft.ifftshift(np.meshgrid(np.arange(-Q,Q+1,dtype=np.double),np.arange(-Q,Q+1,dtype=np.double)))
    else:
        sysOps.throw_exception('ERROR: spat_dims = ' + str(spat_dims))
    
    x_umi_vec_from_ctrs = np.subtract(x_umi[:,:spat_dims],np.multiply(np.double(x_umi_grid_indices),sector_width))
    #Note that the center of the first sector remains at (0,...0)

    x_umi_polynom_tuples_buff[:] = 0.0
    x_umi_polynom_tuples_buff[:,0] = 1.0
    q_polynom_tuples = np.ones(gridside_size_vec,dtype=np.complex)
    
    # np's fft is defined as \sum_{m=0}^{M-1} a_{mn} exp(-2 pi (m q / M))
    # upon fft multiplication by q_polynom_tuples, fft will become \sum_{m=0}^{2M-1} exp(-pi * i * (m q/M))
    # further \sum_{m=0}^{M-1} a_{mn} exp(-i pi (m (q + M) / M)) =  \sum_{m=0}^{M/2 -1} a_{2m,n} exp(-i pi (2m q / M)) -  exp(-i pi q / M) \sum_{m=0}^{M/2 - 1} a_{2m + 1,n} exp(-i pi (2m q / M))
    
    # "nu" index iteration
    
    # For the zeroth order addition, effectively just adding amplification factor weights to FFGT sectors
    obj_glo_loop(q_polynom_tuples, 
                 x_umi[:,spat_dims],x_umi[:,spat_dims+1],
                 bcn_spat_field, trg_spat_field, 
                 bcn_freq_field, trg_freq_field, Numi, 
                 x_umi_polynom_tuples_buff, 
                 x_umi_vec_from_ctrs, 
                 x_umi_grid_indices, q_grid, glo_indices, L, max_nu, spat_dims)
    
    axial_bcn_fft = np.zeros([spat_dims,int(Q)],dtype=np.complex)
    axial_trg_fft = np.zeros([spat_dims,int(Q)],dtype=np.complex)
    
    # can clear large memory usage at this point and proceed to multiplying freq_field
    
    # store axial FFT's -- FFT output (without using fftshift or ifftshift) will have the first Q rows/columns be ascending non-negative frequencies (starting at 0)
    if spat_dims == 2:
        axial_bcn_fft[0,:] = bcn_freq_field[0,1:int(Q+1)]
        axial_bcn_fft[1,:] = bcn_freq_field[1:int(Q+1),0]
        axial_trg_fft[0,:] = trg_freq_field[0,1:int(Q+1)]
        axial_trg_fft[1,:] = trg_freq_field[1:int(Q+1),0]
        bcn_freq_field = np.multiply(bcn_freq_field,np.exp(-np.multiply(np.square(q_grid[0])+np.square(q_grid[1]),np.square(np.pi/L)*s)))
        trg_freq_field = np.multiply(trg_freq_field,np.exp(-np.multiply(np.square(q_grid[0])+np.square(q_grid[1]),np.square(np.pi/L)*s)))
    else:
        sysOps.exitProgram()
    
    bcn_spat_field[:] = 0.0 # now repurposed from storing trg information to storing bcn information
    trg_spat_field[:] = 0.0 
    grad_factor_field = np.multiply(q_grid,1j*2*np.pi/L)
    
    bcn_umi_grad = np.zeros([Numi,spat_dims + 1],dtype=np.double) #will store summed gradient information
    trg_umi_grad = np.zeros([Numi,spat_dims + 1],dtype=np.double)
    src_glo_loop(bcn_umi_grad, trg_umi_grad, x_umi[:,spat_dims],x_umi[:,spat_dims+1], q_polynom_tuples, grad_factor_field, 
                 bcn_spat_field, trg_spat_field, bcn_freq_field, trg_freq_field, Numi, x_umi_polynom_tuples_buff, x_umi_vec_from_ctrs, x_umi_grid_indices, q_grid, glo_indices, L, max_nu, spat_dims,positional_grad)
        
    return (np.min(x_umi[has_bcn_arr,spat_dims]), np.min(x_umi[has_trg_arr,spat_dims+1]), 
            axial_bcn_fft, axial_trg_fft, 
            np.multiply(bcn_umi_grad, np.power(np.sqrt(np.pi)/sector_width,np.double(spat_dims))), 
            np.multiply(trg_umi_grad, np.power(np.sqrt(np.pi)/sector_width,np.double(spat_dims))))

