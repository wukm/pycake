# coding: utf-8
runlog
runlog.keys()
runlog['labels'] = labels
runlog['beta'] = beta
runlog['gamma'] = gamma
log_standard = runlog
with open('output/190225-segmentation_demo_all_semistrict/runlog.json') as f:
    runlog_semistrict = json.load(f)
    
runlog_semistrict
with open('output/190225-segmentation_demo_all_strict/runlog.json') as f:
    runlog_strict = json.load(f)
    
runlog_strict
import xarray as xr
M_standard = M.copy()
P_standard = P.copy()
M_strict = np.array([m[1:] for m in runlog_strict['mccs']])
M_strict
P_strict = np.array([p[1:] for p in runlog_strict['precs']])
P_semistrict = np.array([p[1:] for p in runlog_semistrict['precs']])
M_semistrict = np.array([m[1:] for m in runlog_semistrict['mccs']])
C_standard = np.stack((M_standard, P_standard))
C_standard.shape
C_semistrict = np.stack((M_semistrict, P_semistrict))
C_strict = np.stack((M_strict, P_strict))
labels
C_strict
xr.Dataset({'strict': (['score', 'sample', 'method'], C_strict),
'semistrict': (['score', 'sample', 'method'], C_semistrict),
'standard': (['score', 'sample', 'method'], C_standard)},
coords={'score': ['MCC', 'precision'], 'sample':placentas, 'method':labels})
D = _
placentas
placentas = [strip_ncs_name(pn) for pn in placentas]
placentas
D = xr.Dataset({'strict': (['score', 'sample', 'method'], C_strict),
'semistrict': (['score', 'sample', 'method'], C_semistrict),
'standard': (['score', 'sample', 'method'], C_standard)},
coords={'score': ['MCC', 'precision'], 'sample':placentas, 'method':labels})
D
D['sample']
D['semistrict']
D.dims
arr = D.to_array()
arr
D.coords
arr.shape
arr
D['sample']
A
A = arr
A.values
_.shape
A.dims
A.rename({'variable': 'parametrization'})
A = _
A.dims
A[:,0,:,-1]
A[:,0,:,-1].argmax(axis=0)
a = _
a
a.view()
print(a)
np.ndarray(a)
a.to_masked_array()
A[:,1,:,-1].argmax(axis=0).to_masked_array()
A
AD = A.to_dict()
with open('output/190225-segmentation-all-meta.json', 'w') as f:
    json.dump(AD, f)
    
help(json.dump)
with open('output/190225-segmentation-all-meta.json', 'w') as f:
    json.dump(AD, f, indent=True)
    
    
argmax_p = A[:,1,:,-1].argmax(axis=0).to_masked_array() # parametrization giving highest precision for trough filling algorithm
argmax_m = A[:,0,:,-1].argmax(axis=0).to_masked_array() # parametrization giving highest MCC for trough filling algorithm
max_prec_per_sample = [np.unravel_index(A[:,1,i,:].argmax(), A[:,1,i,:].shape) for i in range(len(placentas))] # coordinates of maximum precision score (parametrization, segmentation method)
max_mcc_per_sample = [np.unravel_index(A[:,0,i,:].argmax(), A[:,0,i,:].shape) for i in range(len(placentas))] # coordinates of maximum precision score (parametrization, segmentation method)
