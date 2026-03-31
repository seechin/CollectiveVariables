# Author: Siqin Cao <scao66@wisc.edu>
# Copyright (c) 2024-2026, University of Wisconsin-Madison and the author
# MIT License

import numpy

class KinetcVariables:
    """ Kinetic Variables Class 

    Parameters
    ----------
    kv : list, shape: [n_trajs][n_frames, n_kv]
        The kinetic variables

    ev: list, shape: [>=n_kv], default: []
        The eigenvalues of Koopman matrix
        Will use MSM as kinetic model, and ignore lag_time

    lag_time: int, default: 1
        The lag time of model
        Will use x(0) and x(t) as kinetic model

    _convention_msm: bool, default: False
        Choose to use MSM (True) or Hummer-Szabo (False) convention
        MSM is row-normalized and Hummer-Szabo is column normalized
        Identical if Koopman is symmetric; Hummer-Szabo convention is faster
        The Koopman is not symmetric in rare cases, e.g. Weighted-AmusetTICA

    Note: dynamical model:
    ----------------------
      use MSM       if ev is given; lag_time is ignored
      use x(t).x(0) if lag_time is given; ev is ignored

    """

    def __init__(this, kv, ev: list=[], lag_time: int=1, _convention_msm: bool=False):
      # convention
        this._convention_msm = _convention_msm
      # elements
        this._kv = kv
        this._kv_, this._len_kv_ = _convert_sequences(kv)
        this._n_kv = len(this._kv_)
        this._n_frames_ = len(this._kv_[0])
        this._ev = ev
        this._lagtime = lag_time
      # caches
        this.__cache_A = []
        this.__key_A   = []

    def set_lag_time(this, lag_time: int):
        """
        Change the lag_time
        
        Parameters
        ----------
        lag_time : int
            The lag time of the model

        Return
        ------
        self
        
        """
        this._lagtime = lag_time
        return this

    def set_ev(this, ev: list):
        """
        Change the eigenvalue list
        
        Parameters
        ----------
        ev: list, shape: [>=n_kv], default: []
            The eigenvalues of Koopman matrix

        Return
        ------
        self

        """
        this._ev = ev
        return this

    def regen_ev(this):
        """
        Regenerate eigenvalues (diagonal of Koopman) with lag_time
        This function will assume KVs are always orthorgonal and ignore the
        off-diagonal elements of Koopman matrix

        No parameter

        Return
        ------
        self
        
        """
        if this._lagtime > 0:
            ev, vrt = numpy.linalg.eig(_compute_Koopman(this._kv, lag_time=this._lagtime))
            idx = [numpy.argsort(numpy.abs(vrt[i]))[-1] for i in range(this._n_kv)]
            this._ev = ev[idx]
        return this

    def timescales(this):
        """
        Display or compute the timescales of the KVs

        No parameter

        Return
        ------
        timescales: numpy.array
            A list of all positive timescales (real & positive)
        
        """
        if this._lagtime>0:
            if len(this._ev)>0:
                ev = numpy.array(this._ev)
            else:
                ev, vr = numpy.linalg.eig(_compute_Koopman(this._kv, lag_time=this._lagtime))
                ev = numpy.sort(ev)[::-1][1:]
            return -this._lagtime / numpy.log(ev[numpy.where((ev>0) & (ev<1))[0]])
        else:
            return numpy.array([])

    def compute_TCM(this, states: list, n_kv: int=0):
        """
        Compute transition count matrix (unsymmetrized)

        Parameters
        ----------
        state: list, shape: [n_all_frames] or [n_trajs][n_frames]
            A list of state assignments, begin with 0 

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        Return
        ------
        TCM: numpy.array, shape: [n_state, n_state]
            The unsymmetrized TCM
        """
        _n_kv = n_kv
        if _n_kv<=0:
            _n_kv = this._n_kv
        _kv_ = this._kv_[:_n_kv]
        states_ = numpy.concatenate(states)
      # generate A matrix
        if id(this.__key_A) == id(states):
            A = this.__cache_A
        else:
            n_states = numpy.max(states_).astype(int) + 1
            A = []
            for i in range(n_states):
                A.append(numpy.where(states_==i, 1, 0))
            A = numpy.array(A)
            this.__cache_A = A
            this.__key_A = states
      # compute TCMs
        TCM = []
        if len(this._ev) < _n_kv:
            xindices, yindices = _gen_xy_indices(this._len_kv_, this._lagtime)
            xxTi = numpy.linalg.inv( _kv_[:, xindices] @ _kv_.T[xindices] )
            if this._convention_msm:
                TCM = (A[:,xindices] @ _kv_.T[xindices]) @ xxTi @ (A[:,xindices] @ _kv_.T[yindices]).T 
            else:
                TCM = (A[:,xindices] @ _kv_.T[yindices]) @ xxTi @ (A[:,xindices] @ _kv_.T[xindices]).T 
        else:
            xxTi = numpy.linalg.inv( _kv_ @ _kv_.T )
            TCM = (A @ _kv_.T) @ numpy.diag(this._ev[:_n_kv]) @ xxTi @ (A @ _kv_.T).T 
      # return TCM
        for i in range(len(TCM)):
            for j in range(len(TCM[i])):
                if TCM[i, j]<0:
                    TCM[i, j] = 0
        return TCM

    def compute_TPM(this, states: list, n_kv: int=0, estimator: str="transpose"):
        """
        Compute transition count matrix (unsymmetrized)

        Parameters
        ----------
        state: list, shape: [n_all_frames] or [n_trajs][n_frames]
            A list of state assignments, begin with 0 

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        estimator: str, default: "transpose"
            Estimator of TPM, can be: none, transpose, mle

        Return
        ------
        TPM: numpy.array, shape: [n_state, n_state]
            The transition probability matrix
        """
        TCM = this.compute_TCM(states, n_kv)
        if estimator.lower() == "transpose":
            TPM = TCM + TCM.T
            for i in range(len(TPM)):
                if numpy.sum(TPM[i])>0:
                    TPM[i] /= numpy.sum(TPM[i])
        elif estimator.lower() == "mle":
            sp, TPM = _msm_mle(TCM)
        else:
            TPM = TCM
            for i in range(len(TPM)):
                if numpy.sum(TPM[i])>0:
                    TPM[i] /= numpy.sum(TPM[i])
        return TPM

    def _compute_committor_spectral(this, source: list, target: list, n_kv: int=0):
        """
        Compute the spectral expansion of committors

        Parameters
        ----------
        source: list
            A list of frame indices of sources

        target: list
            A list of frame indices of targets

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        Return
        ------
        alpha: numpy.array, length: n_kv
            The spectral expansion of committors
        """
        _n_kv = n_kv
        if _n_kv<=0:
            _n_kv = this._n_kv
        _kv_ = this._kv_[:_n_kv]

        u = numpy.zeros(this._n_frames_)
        u[target] = 1
        w = numpy.ones(this._n_frames_)
        w[source] = 0
        w[target] = 0

        if len(this._ev) >= _n_kv:
            xxTi = numpy.linalg.inv(_kv_ @ _kv_.T)
            xuT = _kv_ @ u.T
            xxTixuT = xxTi @ xuT
            xwxT = _kv_ * w @ _kv_.T
            pre = xxTi @ xwxT @ numpy.diag(this._ev[:_n_kv])
            identity = numpy.identity(_n_kv)

            alpha = (numpy.linalg.inv(identity - pre) @ xxTixuT).T
            return alpha

        else:
            xindices, yindices = _gen_xy_indices(this._len_kv_, this._lagtime)
            xxTi = numpy.linalg.inv( _kv_[:, xindices] @ _kv_.T[xindices] )
            xuT = _kv_[:, xindices] @ u[xindices].T
            xxTixuT = xxTi @ xuT
            if this._convention_msm:
                xwxT = _kv_[:, xindices] * w[xindices] @ _kv_[:, xindices].T
                xtxT = _kv_[:, yindices] @ _kv_[:, xindices].T
                pre = xxTi @ xwxT @ xxTi @ xtxT
            else:
                xwxTt = _kv_[:, xindices] * w[xindices] @ _kv_[:, yindices].T
                pre = xxTi @ xwxTt
            identity = numpy.identity(_n_kv)

            alpha = (numpy.linalg.inv(identity - pre) @ xxTixuT).T
            return alpha
            
    def _gen_committor_from_spectral(this, alpha: list, kv_: list=[], source: list=[], target: list=[]):
        """
        Compute committors from spectral expansions

        Parameters
        ----------
        alpha: list, length: n_kv
            The spectral expansion of committors

        kv_: list, shape: [>=n_kv, n_frames], default: this._kv_
            The KVs of the frames to compute committors

        source: list, default: []
            A list of frame indices of sources

        target: list, default: []
            A list of frame indices of targets

        Return
        ------
        committor: numpy.array, length: n_frames
            The committors of each frame

        """
        _kv_ = kv_
        if len(_kv_) < len(alpha):
            _kv_ = this._kv_
        comm = alpha @ _kv_[:len(alpha)]
        comm[source] = 0
        comm[target] = 1
        return comm
        

    def compute_committor(this, source: list, target: list, n_kv: int=0, _return_in_trajs: bool=True):
        """
        Compute committors of all frames

        Parameters
        ----------
        source: list
            A list of frame indices of sources

        target: list
            A list of frame indices of targets

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        _return_in_trajs: bool, default: True
            True: return committors of frames of all trajs: [n_trajs][n_traj_frames]
            False: return committors of all frames: [n_all_frames]

        Return
        ------
        committor: numpy.ndarray, shape: [n_trajs][n_frames] or [n_all_frames]
            The committors of each frame

        """
        alpha = this._compute_committor_spectral(source, target, n_kv)
        comm = this._gen_committor_from_spectral(alpha, source=source, target=target)
        if _return_in_trajs:
            return _convert_to_sequences(comm, this._len_kv_) 
        else:
            return comm
        

    def _compute_mfpt_spectral(this, target: list, n_kv: int=0):
        """
        Compute the spectral expansion of MFPTs

        Parameters
        ----------
        target: list
            A list of frame indices of targets

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        Return
        ------
        alpha: numpy.array, length: n_kv
            The spectral expansion of committors
        """
        _n_kv = n_kv
        if _n_kv<=0:
            _n_kv = this._n_kv
        _kv_ = this._kv_[:_n_kv]

        u = numpy.ones(this._n_frames_)
        w = numpy.ones(this._n_frames_)
        w[target] = 0

        if len(this._ev) >= _n_kv:
            xxTi = numpy.linalg.inv(_kv_ @ _kv_.T)
            xuT = _kv_ @ u.T
            xxTixuT = xxTi @ xuT
            xwxT = _kv_ * w @ _kv_.T
            pre = xxTi @ xwxT @ numpy.diag(this._ev[:_n_kv])
            identity = numpy.identity(_n_kv)

            alpha = (numpy.linalg.inv(identity - pre) @ xxTixuT).T
            return alpha

        else:
            xindices, yindices = _gen_xy_indices(this._len_kv_, this._lagtime)
            xxTi = numpy.linalg.inv( _kv_[:, xindices] @ _kv_.T[xindices] )
            xuT = _kv_[:, xindices] @ u[xindices].T
            xxTixuT = xxTi @ xuT
            if this._convention_msm:
                xwxT = _kv_[:, xindices] * w[xindices] @ _kv_[:, xindices].T
                xtxT = _kv_[:, yindices] @ _kv_[:, xindices].T
                pre = xxTi @ xwxT @ xxTi @ xtxT
            else:
                xwxTt = _kv_[:, xindices] * w[xindices] @ _kv_[:, yindices].T
                pre = xxTi @ xwxTt
            identity = numpy.identity(_n_kv)

            alpha = (numpy.linalg.inv(identity - pre) @ xxTixuT).T
            return alpha

    def _gen_mfpt_from_spectral(this, alpha: list, kv_: list=[], target: list=[]):
        """
        Compute MFPTs from spectral expansions

        Parameters
        ----------
        alpha: list, length: n_kv
            The spectral expansion of committors

        kv_: list, shape: [>=n_kv, n_frames], default: this._kv_
            The KVs of the frames to compute committors

        target: list, default: []
            A list of frame indices of targets

        Return
        ------
        committor: numpy.array, length: n_frames
            The committors of each frame

        """
        _kv_ = kv_
        if len(_kv_) < len(alpha):
            _kv_ = this._kv_
        mfpt = alpha @ _kv_[:len(alpha)]
        mfpt[target] = 0
        return mfpt
        
    def compute_mfpt(this, target: list, n_kv: int=0, _return_in_trajs: bool=True):
        """
        Compute MFPT of all frames

        Parameters
        ----------
        target: list
            A list of frame indices of targets

        n_kv: int, default: 0 (use all KVs)
            The number of CVs used to compute TCM

        _return_in_trajs: bool, default: False
            True: return committors of frames of all trajs: [n_trajs][n_traj_frames]
            False: return committors of all frames: [n_all_frames]

        Return
        ------
        committor: numpy.array, length: n_frames
            The committors of each frame

        """
        alpha = this._compute_mfpt_spectral(target, n_kv)
        #return this._gen_mfpt_from_spectral(alpha)
        mfpt = this._gen_mfpt_from_spectral(alpha)
        if _return_in_trajs:
            return _convert_to_sequences(mfpt, this._len_kv_) 
        else:
            return mfpt


# some common tools

def _convert_sequences(sequences):
    """  
    Convert sequences into the matrix

    Parameters
    ----------
    sequences : 2D/3D list, [n_trajs][n_frames] or [n_trajs][n_frames, n_features]
        a list of trajectories

    Returns
    -------
    data_matrix : 2D array, [n_features, n_all_frames]
        the converted sequences

    traj_lens : list, [n_trajs]
        lengths of each trajectory

    """
    data_matrix = [] 
    for k in range(len(sequences)):
        data_matrix.append(sequences[k].T)
    data_matrix = numpy.hstack(data_matrix)

    traj_lens = [] 
    for k in range(len(sequences)):
        traj_lens.extend([len(sequences[k])])

    return data_matrix, traj_lens

def _convert_to_sequences(data, traj_lens):
    """  
    Convert matrix to sequences

    Parameters
    ----------
    data : 1D/2D Array, [n_all_frames] / [n_dim, n_all_frames] / [n_all_frames, n_dim]
        the Amuset sequence

    traj_lens : list, [n_trajs]
        lengths of each trajectory

    Return
    ------
    sequences : 3D list, [n_traj][n_frames, n_dim]
        a list of trajectories
    """

    if not isinstance(data[0], (int, float)):
        if len(data)>len(data[0]):
            data_ = data
        else:
            data_ = data.T
        __is_input_2D = True
    else:
        data_ = data
        __is_input_2D = False
    seq = []
    op = 0
    for it in range(len(traj_lens)):
        if __is_input_2D:
            seq.append(data_[op:op+traj_lens[it], :])
        else:
            seq.append(data_[op:op+traj_lens[it]])
        op += traj_lens[it]
    return seq

def _gen_xy_indices(traj_lens, lag_time):
    x_indices = numpy.array([], dtype=int)
    y_indices = numpy.array([], dtype=int)
    pos = 0 
    for i in range(len(traj_lens)):
        x_indices = numpy.concatenate((x_indices, numpy.arange(pos, pos + traj_lens[i] - lag_time)))
        y_indices = numpy.concatenate((y_indices, numpy.arange(pos + lag_time, pos + traj_lens[i])))
        pos += traj_lens[i]
    return x_indices, y_indices

def _compute_Koopman(kv, lag_time, symmetrize: bool=True):
    kv_, len_kv_ = _convert_sequences(kv)
    xindices, yindices = _gen_xy_indices(len_kv_, lag_time)
    C00 = kv_[:, xindices] @ kv_[:, xindices].T
    C01 = kv_[:, xindices] @ kv_[:, yindices].T
    C11 = kv_[:, yindices] @ kv_[:, yindices].T
    if symmetrize:
        K = numpy.linalg.inv(C00 + C11) @ (C01 + C01.T)
    else:
        K = numpy.linalg.inv(C00) @ C01
    #return numpy.diag(K)
    return K

def _msm_mle(input: list, steps: int=100000, errtol: float=1e-7, initial_guess: list=None, weight: float=1, debug: int=0):
    """
    Use MLE estimator to compute TPM and stationary populations from unsymmetrized TCM

    Parameters:
    -----------
    input : a list of shape [dim, dim] or a two-element list [raw_trajs, lag_time]
        shape [dim, dim] : input = the unsymmetrized transition count matrix
        list [raw_trajs, lag_time] : perform MLE with given trajectories and lag time 
            state_trajs: shape [n_trajs, n_frames]

    steps : int, default: 100000
        The maximum steps of this-consistent iterations

    errtol : float, default: 1e-7
       the error tolerance of MLE
       will use off-diagonal criterion if errtol<=0

    initial_guess : list [dim], default: None
        The initial guess of this-consistent iterations

    weight: float, default: 1
        The step-in weight of this-consistent iterations

    debug: int, default: 1
        the debug level

    Returns:
    --------
    sp : list [dim]
        The stationary populations under the MLE estimator

    TPM : numpy.ndarray [dim, dim]
        The transition probability matrix under the MLE estimator 

    """
    if len(input) == 2 and isinstance(input[1], int):
        _trajs = input[0]
        _lagtime = input[1]
        _dim = numpy.max(_trajs).astype(int)+1
        _TCM = numpy.zeros((_dim, _dim), dtype=int)
        for i in range(len(_trajs)):
            for t in range(len(_trajs[i])-_lagtime):
                _TCM[int(_trajs[i][t]), int(_trajs[i][t+_lagtime])] += 1
        if debug>=2:
            print("# _msm_mle: TCM=:\n"+str(_TCM))
    else:
        _TCM = input
        _dim = len(_TCM)

  # set initial guess
    _sp_count = numpy.sum(_TCM, axis=1)
    _sp_count = _sp_count / numpy.sum(_sp_count)
    _sp = numpy.ones(_dim) / _dim
    if not (initial_guess is None):
        for i in range(_dim):
            _sp[i] = initial_guess[i]
    else:
        for i in range(_dim):
            _sp[i] = _sp_count[i]
    _N = numpy.sum(_TCM, axis=1)

  # MLE this-consistent iterations
    _sp_new = numpy.zeros(_dim) # [0] * _dim
    _TCM_ss = _TCM + numpy.array(_TCM).T
    _last_recover_err = -1
    for istep in range(steps+1):
        _N_over_sp = _N / _sp
        for i in range(len(_sp_new)):
            _sp_new[i] = 0
        for i in range(_dim):
            for j in range(_dim):
                _sp_new[i] += _TCM_ss[i,j] / (_N_over_sp[i] + _N_over_sp[j])
        # _sp_new /= numpy.sum(_sp_new)
        _sum_sp_new = numpy.sum(_sp_new)
        for i in range(_dim):
            _sp_new[i] /= _sum_sp_new

      # check convergence
        _err = numpy.max(numpy.abs(numpy.array(_sp_new) - _sp))
        #_err = numpy.linalg.norm(_sp_new - _sp)
        if errtol <= 0:
          # off-diagonal criterion
            __recovered_TPM = _TCM.T * _sp_new / _sp_count
            _recover_err = numpy.sum(numpy.abs(__recovered_TPM - __recovered_TPM.T))
            if _last_recover_err>0 and _recover_err>_last_recover_err:
                _sp = _sp_new
                break
            _last_recover_err = _recover_err
        else:
          # SCF convergence criterion
            if _err < errtol:
                _sp = _sp_new
                break

      # next step
        if weight>0 and weight!=1:
            for i in range(_dim):
                _sp[i] = _sp[i]*(1-weight) + _sp_new[i]*weight
        else:
            for i in range(_dim):
                _sp[i] = _sp_new[i]
        #_sp = _sp_new
        if debug>=2 and istep%10000==0:
            print("# _msm_mle: step %d err %g"%(istep, _err), end="\r")

    if debug>=1:
        print("# _msm_mle: step %d err %g"%(istep, _err), end="\n")

    _sp /= numpy.sum(_sp)
    _TPM = numpy.zeros((_dim, _dim))
    for i in range(_dim):
        for j in range(_dim):
            _TPM[i][j] = (_TCM_ss[i,j]*_sp[j] / (_N[j]*_sp[i] + _N[i]*_sp[j])) * _sp[i]
    for i in range(_dim):
        _TPM[i] /= numpy.sum(_TPM[i])
    
    return list(_sp), _TPM

