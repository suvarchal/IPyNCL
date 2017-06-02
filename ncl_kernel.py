from ipykernel.kernelbase import Kernel
from IPython.utils.path import locate_profile
from pexpect import EOF,spawn
import pexpect
from nclreplwrap import REPLWrapper as replw
import signal
import re
import os
from distutils.spawn import find_executable
import cPickle
import sys
__version__ = '0.9.1'
class NCLKernel(Kernel):
    implementation = 'NCL'
    implementation_version = '1.0'
    language = 'ncl'
    language_version = '0.1'
    language_info = {'mimetype': 'text/plain','name':'ncl','file_extension':'.ncl'}
    banner = "NCAR IPython Kernel "

    def __init__(self, **kwargs):
        Kernel.__init__(self, **kwargs)
        os.environ['PAGER']="more"
        self._start_ncl()
        
        self.pexpect_version=pexpect.__version__    
        try:
            self.hist_file = os.path.join(locate_profile(),'ncl_kernel.hist')
        except:
            self.hist_file = None
            self.log.warn('No default profile found, history unavailable')

        self.max_hist_cache = 1000
        self.hist_cache = []
	#self._default_matches=cPickle.load(open('data/inbuilt_list', 'rb'))
        self._default_matches=self.inbuiltlist()
    def get_usage(self):
        return "This is the NCL kernel."
    def _start_ncl(self):
        sig = signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            self._executable = find_executable("ncl")
            #self._executable = self._executable+" -p"
            self._child  = spawn(self._executable,timeout = 300) #env={'TERM':'xterm','PAGER':'more'})
            self._child.log=sys.stdout
            self.nclwrapper = replw(self._child,"ncl \d> ",None)
            self._child.setwinsize(400,500)
        finally:
            signal.signal(signal.SIGINT, sig)
       

    def do_execute(self, code, silent=True, store_history=True, user_expressions=None,
                   allow_stdin=False):
        code   =  code.strip()
	abort_msg = {'status': 'abort',
                     'execution_count': self.execution_count}
        interrupt = False
        av=[]
        #doLine=False
        #below removes print stement
        #if len(code.splitlines()) > 1:
        #    av=[line for line in code.splitlines() if not line.strip().startswith("print(")]
        #   code='\n'.join(av)

        try:
            if code.startswith(';!'):
                #that is these commands are run from ncl
	        cmd    = "system(\""+code.replace(";!","").strip()+"\")"
                output = self.nclwrapper.run_command(cmd,timeout=None)
                output = '\n'.join(output.splitlines()[1::])+'\n'
            elif code.startswith('%debug'):
       	    	#self.nclwrapper.child.send(code+"\n\Q")
                #self.nclwrapper.child.expect(u'ncl \d> ')
                #output=self.nclwrapper.child.before
	    	#indstr=output.find('\n')
                #indend=output.rfind('\n')
                #output=output[indstr+1:indend]
	    	#output = '\n'.join([line for line in output.splitlines()[1::] if line.strip()])+'\n'
                #output='\n'.join(output.splitlines()[1::])+'\n'
                output="Before: \n"+str(self._child.before)+"\n After: \n"+str(self._child.after)
 	    elif code.startswith(';%timeit'):
	        code=code.replace(";%timeit","").strip()
                tstart=time.time()
		output = self.nclwrapper.run_command(code.strip(), timeout=None)
		output = "\nTime: %s seconds.\n" % (time.time() - tstart)
	    else:
  	        #output = self.nclwrapper.run_command(code.strip(), timeout=None)
                #indstr=output.find('\n')
                #indend=output.rfind('\n')
                #output=output[indstr+1:indend] 
                #output = '\n'.join([line for line in output.splitlines()[1::] if line.strip()])+'\n'
                
                #using repl child
                #self.nclwrapper.child.setecho(False)
                #self.nclwrapper.child.waitnoecho(False)
                #for line in code.strip().splitlines():
                #    self.nclwrapper.child.sendline(line)
                #    self.nclwrapper.child.expect(["ncl",pexpect.EOF])
                #output=self.nclwrapper.child.before 
                #output = '\n'.join([line for line in output.splitlines()[1::] if line.strip()])+'\n'
                #self.nclwrapper.child.setecho(True)
                #self.nclwrapper.child.waitnoecho(True)
                #using pexpect child
                #self._child.setecho(False)
                #self._child.waitnoecho(True)
                self.pattern=["ncl"] #,"ncl \d >","\r\n","lines",pexpect.EOF,pexpect.TIMEOUT]
                for line in code.strip().splitlines():
                    self._child.sendline(line)
                    #self._child.expect(["ncl",pexpect.EOF,pexpect.TIMEOUT])
                    i=self._child.expect(self.pattern)
                output=self._child.before
                self.code=code
                self.output=output
                self.outmatch=self.pattern[i]    
                output = '\n'.join([line for line in output.splitlines()[1::] if line.strip()])+'\n'
                  
#2 ways of doing it: 
#---1send bunch of lines and then do expect once 
#---2send expect everytime  
        except KeyboardInterrupt:
            self._child.sendintr()
            output = self._child.before
            if not self._child.isalive():
                 output+='\n killing ncl session and restarting'
                 interrupt = True
                 self._start_ncl()
                 if self._child.isalive():
                     return {'status': 'ok','execution_count': [],'payload': [],'user_expressions': {}}

	except EOF:
            #when would this happen
            output = self._child.before +'\n Reached EOF Restarting NCL'
            if not self._child.isalive():
                 output+='\n killing ncl session and restarting'
                 interrupt = True
                 self._start_ncl()
 	if not silent:
            #is is being used when kernel is being made from jupyter notebook
            stream_content = {'name': 'stdout', 'text': output}
            self.send_response(self.iopub_socket, 'stream', stream_content)
            
            #stream_content = {'name': 'stdout', 'text': "what the fck"}
            #self.send_response(self.iopub_socket, 'stream', stream_content)
            
	    #stream_content = {'data':{'text':l}} #or forl in output.splitlines??
            #self.send_response(self.iopub_socket,'display_data',stream_content)
        if interrupt:
            return {'status': 'abort', 'execution_count': self.execution_count}
        return {'status': 'ok','execution_count': self.execution_count,'payload': [],'user_expressions': {}}
        #return {'status': 'ok','execution_count': 11,'payload': [],'user_expressions': {}}
    def do_complete(self, code, cursor_pos):
        code = code[:cursor_pos]
        default = {'matches': [], 'cursor_start': 0,
                   'cursor_end': cursor_pos, 'metadata': dict(),
                   'status': 'ok'}

        if not code or code[-1] == ' ':
            return default
        
 	tokens = code.split()
        if not tokens:
            return default

        matches = []
        token = tokens[-1]
        start = cursor_pos - len(token)
	lib_search_path="$NCARG_ROOT/lib/ncarg/nclscripts/csm/* $NCARG_ROOT/lib/ncarg/nclscripts/*"
        cmd="system(\"grep -i 'function\|procedure' "+lib_search_path+"|cut -d':' -f2|grep -v '^;'|cut -d' ' -f2|cut -d'(' -f1|grep -v -e '^$'\")"
        #output=self.nclwrapper.run_command(cmd,timeout=None) 
        #matches.extend([e for e in list(set(output.split()[1:])) if not e.startswith(";")])
        matches.extend(self._default_matches)    
        if not matches:
            return default
        matches = [m for m in matches if m.startswith(token)]

        return {'matches': sorted(matches), 'cursor_start': start,
                'cursor_end': cursor_pos, 'metadata': dict(),
                'status': 'ok'}
    def inbuiltlist(self):
        ncl_inbuilt=['acos', 'addfile', 'addfiles', 'all', 'angmom_atm', 'any', 'area_conserve_remap', 'area_hi2lores', 'area_poly_sphere', 'asciiread', 'asciiwrite', 'asin', 'atan', 'atan2', 'attsetvalues', 'avg', 'bin_avg', 'bin_sum', 'bw_bandpass_filter', 'cbinread', 'cbinwrite', 'cd_calendar', 'cd_inv_calendar', 'cdfbin_p', 'cdfbin_pr', 'cdfbin_s', 'cdfbin_xn', 'cdfchi_p', 'cdfchi_x', 'cdfgam_p', 'cdfgam_x', 'cdfnor_p', 'cdfnor_x', 'cdft_p', 'cdft_t', 'ceil', 'center_finite_diff', 'center_finite_diff_n', 'cfftb', 'cfftf', 'cfftf_frq_reorder', 'charactertodouble', 'charactertofloat', 'charactertointeger', 'charactertolong', 'charactertoshort', 'charactertostring', 'chartodouble', 'chartofloat', 'chartoint', 'chartointeger', 'chartolong', 'chartoshort', 'chartostring', 'chiinv', 'cla_sq', 'clear', 'color_index_to_rgba', 'conform', 'conform_dims', 'cos', 'cosh', 'count_unique_values', 'covcorm', 'covcorm_xy', 'craybinnumrec', 'craybinrecread', 'create_graphic', 'csa1', 'csa1d', 'csa1s', 'csa1x', 'csa1xd', 'csa1xs', 'csa2', 'csa2d', 'csa2l', 'csa2ld', 'csa2ls', 'csa2lx', 'csa2lxd', 'csa2lxs', 'csa2s', 'csa2x', 'csa2xd', 'csa2xs', 'csa3', 'csa3d', 'csa3l', 'csa3ld', 'csa3ls', 'csa3lx', 'csa3lxd', 'csa3lxs', 'csa3s', 'csa3x', 'csa3xd', 'csa3xs', 'csc2s', 'csgetp', 'css2c', 'cssetp', 'cssgrid', 'csstri', 'csvoro', 'cumsum', 'cz2ccm', 'day_of_week', 'day_of_year', 'days_in_month', 'default_fillvalue', 'delete', 'depth_to_pres', 'destroy', 'determinant', 'dewtemp_trh', 'dgeevx_lapack', 'dim_acumrun_n', 'dim_avg', 'dim_avg_n', 'dim_avg_wgt', 'dim_avg_wgt_n', 'dim_cumsum', 'dim_cumsum_n', 'dim_gamfit_n', 'dim_gbits', 'dim_max', 'dim_max_n', 'dim_median', 'dim_median_n', 'dim_min', 'dim_min_n', 'dim_num', 'dim_num_n', 'dim_numrun_n', 'dim_pqsort', 'dim_pqsort_n', 'dim_product', 'dim_product_n', 'dim_rmsd', 'dim_rmsd_n', 'dim_rmvmean', 'dim_rmvmean_n', 'dim_rmvmed', 'dim_rmvmed_n', 'dim_spi_n', 'dim_standardize', 'dim_standardize_n', 'dim_stat4', 'dim_stat4_n', 'dim_stddev', 'dim_stddev_n', 'dim_sum', 'dim_sum_n', 'dim_sum_wgt', 'dim_sum_wgt_n', 'dim_variance', 'dim_variance_n', 'dimsizes', 'doubletobyte', 'doubletochar', 'doubletocharacter', 'doubletofloat', 'doubletoint', 'doubletointeger', 'doubletolong', 'doubletoshort', 'dpres_hybrid_ccm', 'dpres_plevel', 'draw', 'draw_color_palette', 'dsgetp', 'dsgrid2', 'dsgrid2d', 'dsgrid2s', 'dsgrid3', 'dsgrid3d', 'dsgrid3s', 'dspnt2', 'dspnt2d', 'dspnt2s', 'dspnt3', 'dspnt3d', 'dspnt3s', 'dssetp', 'dtrend', 'dtrend_msg', 'dtrend_msg_n', 'dtrend_n', 'dtrend_quadratic', 'dtrend_quadratic_msg_n', 'dv2uvF', 'dv2uvf', 'dv2uvG', 'dv2uvg', 'dz_height', 'echo_on', 'eof2data', 'eof_varimax', 'eofcor', 'eofcor_pcmsg', 'eofcor_ts', 'eofcov', 'eofcov_pcmsg', 'eofcov_ts', 'eofunc', 'eofunc_ts', 'eofunc_varimax', 'equiv_sample_size', 'erf', 'erfc', 'esacr', 'esacv', 'esccr', 'esccv', 'escorc', 'escorc_n', 'escovc', 'exit', 'exp', 'exp_tapersh', 'exp_tapersh_wgts', 'exp_tapershC', 'ezfftb', 'ezfftb_n', 'ezfftf', 'ezfftf_n', 'f2foshv', 'f2fsh', 'f2fshv', 'f2gsh', 'f2gshv', 'fabs', 'fbindirread', 'fbindirwrite', 'fbinnumrec', 'fbinread', 'fbinrecread', 'fbinrecwrite', 'fbinwrite', 'fft2db', 'fft2df', 'fftshift', 'fileattdef', 'filechunkdimdef', 'filedimdef', 'fileexists', 'filegrpdef', 'filevarattdef', 'filevarchunkdef', 'filevarcompressleveldef', 'filevardef', 'filevardimsizes', 'filwgts_lancos', 'filwgts_lanczos', 'filwgts_normal', 'floattobyte', 'floattochar', 'floattocharacter', 'floattoint', 'floattointeger', 'floattolong', 'floattoshort', 'floor', 'fluxEddy', 'fo2fsh', 'fo2fshv', 'fourier_info', 'frame', 'fspan', 'ftcurv', 'ftcurvd', 'ftcurvi', 'ftcurvp', 'ftcurvpi', 'ftcurvps', 'ftcurvs', 'ftest', 'ftgetp', 'ftkurv', 'ftkurvd', 'ftkurvp', 'ftkurvpd', 'ftsetp', 'ftsurf', 'g2fshv', 'g2gsh', 'g2gshv', 'gamma', 'gammainc', 'gaus', 'gaus_lobat', 'gaus_lobat_wgt', 'gc_aangle', 'gc_clkwise', 'gc_dangle', 'gc_inout', 'gc_latlon', 'gc_onarc', 'gc_pnt2gc', 'gc_qarea', 'gc_tarea', 'generate_2d_array', 'get_color_rgba', 'get_cpu_time', 'get_isolines', 'get_ncl_version', 'get_script_name', 'get_script_prefix_name', 'get_sphere_radius', 'get_unique_difference', 'get_unique_intersection', 'get_unique_union', 'get_unique_values', 'getbitsone', 'getenv', 'getfiledimsizes', 'getfilegrpnames', 'getfilepath', 'getfilevaratts', 'getfilevarchunkdimsizes', 'getfilevardims', 'getfilevardimsizes', 'getfilevarnames', 'getfilevartypes', 'getvaratts', 'getvardims', 'gradsf', 'gradsg', 'greg2jul', 'grid2triple', 'hsvrgb', 'hydro', 'hyi2hyo', 'igradsf', 'igradsF', 'igradsg', 'igradsG', 'ilapsf', 'ilapsF', 'ilapsg', 'ilapsG', 'ilapvf', 'ilapvg', 'ind', 'ind_resolve', 'int2p', 'int2p_n', 'integertobyte', 'integertochar', 'integertocharacter', 'integertoshort', 'inttobyte', 'inttochar', 'inttoshort', 'inverse_matrix', 'isatt', 'isbigendian', 'isbyte', 'ischar', 'iscoord', 'isdefined', 'isdim', 'isdimnamed', 'isdouble', 'isenumeric', 'isfile', 'isfilepresent', 'isfilevar', 'isfilevaratt', 'isfilevarcoord', 'isfilevardim', 'isfloat', 'isfunc', 'isgraphic', 'isint', 'isint64', 'isinteger', 'isleapyear', 'islogical', 'islong', 'ismissing', 'isnan_ieee', 'isnumeric', 'ispan', 'isproc', 'isscalar', 'isshort', 'issnumeric', 'isstring', 'isubyte', 'isuint', 'isuint64', 'isulong', 'isunlimited', 'isunsigned', 'isushort', 'isvar', 'kolsm2_n', 'kron_product', 'lapsf', 'lapsG', 'lapsg', 'lapvf', 'lapvg', 'latlon2utm', 'lclvl', 'lderuvf', 'lderuvg', 'linint1', 'linint1_n', 'linint2', 'linint2_points', 'linmsg', 'linmsg_n', 'linrood_latwgt', 'linrood_wgt', 'list_files', 'list_filevars', 'list_hlus', 'list_procfuncs', 'list_vars', 'ListAppend', 'ListCount', 'ListGetType', 'ListIndex', 'ListIndexFromName', 'ListPop', 'ListPush', 'ListSetType', 'loadscript', 'local_max', 'local_min', 'log', 'log10', 'longtobyte', 'longtochar', 'longtocharacter', 'longtoint', 'longtointeger', 'longtoshort', 'lspoly', 'lspoly_n', 'max', 'maxind', 'min', 'minind', 'mixed_layer_depth', 'mixhum_ptd', 'mixhum_ptrh', 'mjo_cross_coh2pha', 'mjo_cross_segment', 'moc_globe_atl', 'monthday', 'namedcolor2rgba', 'natgrid', 'natgridd', 'natgrids', 'ncargpath', 'ncargversion', 'ndctodata', 'ndtooned', 'new', 'NewList', 'ngezlogo', 'nggcog', 'nggetp', 'nglogo', 'ngsetp', 'NhlAddAnnotation', 'NhlAddData', 'NhlAddOverlay', 'NhlAddPrimitive', 'NhlAppGetDefaultParentId', 'NhlChangeWorkstation', 'NhlClassName', 'NhlClearWorkstation', 'NhlDataPolygon', 'NhlDataPolyline', 'NhlDataPolymarker', 'NhlDataToNDC', 'NhlDestroy', 'NhlDraw', 'NhlFrame', 'NhlFreeColor', 'NhlGetBB', 'NhlGetClassResources', 'NhlGetErrorObjectId', 'NhlGetNamedColorIndex', 'NhlGetParentId', 'NhlGetParentWorkstation', 'NhlGetWorkspaceObjectId', 'NhlIsAllocatedColor', 'NhlIsApp', 'NhlIsDataComm', 'NhlIsDataItem', 'NhlIsDataSpec', 'NhlIsTransform', 'NhlIsView', 'NhlIsWorkstation', 'NhlName', 'NhlNDCPolygon', 'NhlNDCPolyline', 'NhlNDCPolymarker', 'NhlNDCToData', 'NhlNewColor', 'NhlNewDashPattern', 'NhlNewMarker', 'NhlPalGetDefined', 'NhlRemoveAnnotation', 'NhlRemoveData', 'NhlRemoveOverlay', 'NhlRemovePrimitive', 'NhlSetColor', 'NhlSetDashPattern', 'NhlSetMarker', 'NhlUpdateData', 'NhlUpdateWorkstation', 'nice_mnmxintvl', 'nngetaspectd', 'nngetaspects', 'nngetp', 'nngetsloped', 'nngetslopes', 'nngetwts', 'nngetwtsd', 'nnpnt', 'nnpntd', 'nnpntend', 'nnpntendd', 'nnpntinit', 'nnpntinitd', 'nnpntinits', 'nnpnts', 'nnsetp', 'num', 'omega_ccm', 'onedtond', 'overlay', 'pdfxy_bin', 'poisson_grid_fill', 'pop_remap', 'potmp_insitu_ocn', 'prcwater_dp', 'pres2hybrid', 'pres_hybrid_ccm', 'pres_hybrid_jra55', 'pres_sigma', 'print', 'print_table', 'printFileVarSummary', 'printVarSummary', 'product', 'pslec', 'pslhor', 'pslhyp', 'random_chi', 'random_gamma', 'random_normal', 'random_setallseed', 'random_uniform', 'rcm2points', 'rcm2rgrid', 'rdsstoi', 'read_colormap_file', 'reg_multlin', 'regcoef', 'regCoef', 'regCoef_n', 'regline', 'relhum', 'relhum_ice', 'relhum_water', 'replace_ieeenan', 'reshape', 'reshape_ind', 'rgbhls', 'rgbhsv', 'rgbyiq', 'rgrid2rcm', 'rhomb_trunc', 'rhomb_trunC', 'rip_cape_2d', 'rip_cape_3d', 'round', 'rtest', 'runave', 'runave_n', 'set_sphere_radius', 'setfileoption', 'sfvp2uvf', 'sfvp2uvg', 'shaeC', 'shaec', 'shagC', 'shagc', 'shgetnp', 'shgetp', 'shgrid', 'shorttobyte', 'shorttochar', 'shorttocharacter', 'show_ascii', 'shsec', 'shseC', 'shsetp', 'shsgc', 'shsgC', 'shsgc_R42', 'sigma2hybrid', 'simpeq', 'simpne', 'sin', 'sindex_yrmo', 'sinh', 'sizeof', 'sleep', 'smth9', 'snindex_yrmo', 'solve_linsys', 'span_color_indexes', 'span_color_rgba', 'span_named_colors', 'sparse_matrix_mult', 'spcorr', 'spcorr_n', 'specx_anal', 'specxy_anal', 'speidx', 'sprintf', 'sprinti', 'sqrt', 'sqsort', 'srand', 'stat2', 'stat4', 'stat_medrng', 'stat_trim', 'status_exit', 'stdatmus_p2tdz', 'stdatmus_z2tdp', 'stddev', 'str_capital', 'str_concat', 'str_fields_count', 'str_get_cols', 'str_get_dq', 'str_get_field', 'str_get_nl', 'str_get_sq', 'str_get_tab', 'str_index_of_substr', 'str_insert', 'str_is_blank', 'str_join', 'str_left_strip', 'str_lower', 'str_match', 'str_match_ic', 'str_match_ic_regex', 'str_match_ind', 'str_match_ind_ic', 'str_match_ind_ic_regex', 'str_match_ind_regex', 'str_match_regex', 'str_right_strip', 'str_split', 'str_split_by_length', 'str_split_csv', 'str_squeeze', 'str_strip', 'str_sub_str', 'str_switch', 'str_upper', 'stringtochar', 'stringtocharacter', 'stringtodouble', 'stringtofloat', 'stringtoint', 'stringtointeger', 'stringtolong', 'stringtoshort', 'strlen', 'student_t', 'sum', 'svd_lapack', 'svdcov', 'svdcov_sv', 'svdstd', 'svdstd_sv', 'system', 'systemfunc', 'tanh', 'taper', 'taper_n', 'tdclrs', 'tdctri', 'tdcudp', 'tdcurv', 'tddtri', 'tdez2d', 'tdez3d', 'tdgetp', 'tdgrds', 'tdgrid', 'tdgtrs', 'tdinit', 'tditri', 'tdlbla', 'tdlblp', 'tdlbls', 'tdline', 'tdlndp', 'tdlnpa', 'tdlpdp', 'tdmtri', 'tdotri', 'tdpara', 'tdplch', 'tdprpa', 'tdprpi', 'tdprpt', 'tdsetp', 'tdsort', 'tdstri', 'tdstrs', 'tdttri', 'thornthwaite', 'tobyte', 'tochar', 'todouble', 'tofloat', 'toint', 'toint64', 'tointeger', 'tolong', 'toshort', 'tosigned', 'tostring', 'tostring_with_format', 'totype', 'toubyte', 'touint', 'touint64', 'toulong', 'tounsigned', 'toushort', 'trend_manken', 'tri_trunC', 'tri_trunc', 'triple2grid', 'triple2grid2d', 'trop_wmo', 'ttest', 'typeof', 'unique_string', 'update', 'ushorttoint', 'ut_calendar', 'ut_inv_calendar', 'utm2latlon', 'uv2dv_cfd', 'uv2dvf', 'uv2dvF', 'uv2dvg', 'uv2dvG', 'uv2sfvpF', 'uv2sfvpf', 'uv2sfvpG', 'uv2sfvpg', 'uv2vr_cfd', 'uv2vrdvF', 'uv2vrdvf', 'uv2vrdvG', 'uv2vrdvg', 'uv2vrF', 'uv2vrf', 'uv2vrG', 'uv2vrg', 'v5d_create', 'v5d_setLowLev', 'v5d_setUnits', 'v5d_write', 'v5d_write_var', 'variance', 'vhaeC', 'vhaec', 'vhagC', 'vhagc', 'vhseC', 'vhsec', 'vhsgc', 'vhsgC', 'vibeta', 'vinth2p', 'vinth2p_ecmwf', 'vinth2p_ecmwf_nodes', 'vinth2p_nodes', 'vintp2p_ecmwf', 'vr2uvf', 'vr2uvF', 'vr2uvg', 'vr2uvG', 'vrdv2uvf', 'vrdv2uvF', 'vrdv2uvg', 'vrdv2uvG', 'wavelet_default', 'weibull', 'wgt_area_smooth', 'wgt_areaave', 'wgt_areaave2', 'wgt_arearmse', 'wgt_arearmse2', 'wgt_areasum2', 'wgt_runave', 'wgt_runave_n', 'wgt_vert_avg_beta', 'wgt_volave', 'wgt_volave_ccm', 'wgt_volrmse', 'wgt_volrmse_ccm', 'where', 'wk_smooth121', 'wmbarb', 'wmbarbmap', 'wmdrft', 'wmgetp', 'wmlabs', 'wmsetp', 'wmstnm', 'wmvect', 'wmvectmap', 'wmvlbl', 'wrf_avo', 'wrf_cape_2d', 'wrf_cape_3d', 'wrf_dbz', 'wrf_eth', 'wrf_helicity', 'wrf_ij_to_ll', 'wrf_interp_1d', 'wrf_interp_2d_xy', 'wrf_interp_3d_z', 'wrf_latlon_to_ij', 'wrf_ll_to_ij', 'wrf_omega', 'wrf_pvo', 'wrf_rh', 'wrf_slp', 'wrf_smooth_2d', 'wrf_td', 'wrf_tk', 'wrf_updraft_helicity', 'wrf_uvmet', 'wrf_virtual_temp', 'wrf_wetbulb', 'wrf_wps_close_int', 'wrf_wps_open_int', 'wrf_wps_rddata_int', 'wrf_wps_rdhead_int', 'wrf_wps_read_int', 'wrf_wps_write_int', 'write_matrix', 'write_table', 'zonal_mpsi']
        return ncl_inbuilt
if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(kernel_class=NCLKernel)
