import masterProcesses
import summaryAnalysis
import sysOps
import upstream
import sys

'''
GLOBAL VARIABLES
statuslogfilename -- logfile used to print status. Default is statuslog.csv
globaldatapath
'''
global statuslogfilename
global globaldatapath

if __name__ == '__main__':
    
    # Calls sub-routines
    
    #optimOps.test_ffgt()
    
    sys.argv[len(sys.argv)-1] = sys.argv[len(sys.argv)-1].strip('\r')
    sysOps.initiate_runpath('')
    sysOps.initiate_statusfilename('',make_file = False)
    sys.argv = sys.argv[1:] #remove first argument (script call)
    sysOps.throw_status('sys.argv = ' + str(sys.argv))
    
    if len(sys.argv)>0 and sys.argv[0][(len(sys.argv[0])-2):] == '//':
        #if first argument is a directory, use this directory as the data directory for all subsequent operations
        sysOps.initiate_runpath(sys.argv[0]) #initiate data run path
        sys.argv = sys.argv[1:] #remove directory from argument list
    
    sysOps.globalmasterProcess = masterProcesses.masterProcess([])
    
    if len(sys.argv)==0 or sys.argv[0]=='data_layout.csv':
        sysOps.globalmasterProcess.generate_uxi_library()
    elif sys.argv[0].endswith('infer'):
        compute_local_solutions_only = False
        if len(sys.argv) > 1 and sys.argv[1] == 'local':
            sysOps.throw_status('Performing local computing function alone.')
            compute_local_solutions_only = True
        if sys.argv[0]=='smle_infer':
            sysOps.globalmasterProcess.dnamic_inference(True, False, False, compute_local_solutions_only)
        elif sys.argv[0]=='msmle_infer':
            sysOps.globalmasterProcess.dnamic_inference(False, True, False, compute_local_solutions_only)
        elif sys.argv[0]=='ptmle_infer':
            sysOps.globalmasterProcess.dnamic_inference(False, False, False, compute_local_solutions_only)
        elif sys.argv[0]=='segment_infer':
            sysOps.globalmasterProcess.dnamic_inference(False, False, True, compute_local_solutions_only)
    elif sys.argv[0] == 'layout':
        upstream.generate_data_layout()
    elif(len(sys.argv)>2 and sys.argv[0] == 'compare'):
        sysOps.globalmasterProcess.crosscomparison_analysis(sys.argv)
    elif(sys.argv[0] == 'stats'):
        summaryAnalysis.gather_rarefaction_data()
        summaryAnalysis.gather_raw_read_stats()
        summaryAnalysis.gather_stats()
        summaryAnalysis.gather_cluster_stats()
    else:
        sysOps.throw_exception('Unrecognized pipeline input: ' + str(sys.argv))

    print "Completed run."
