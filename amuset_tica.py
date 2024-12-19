# Author: Siqin Cao <scao66@wisc.edu>
# Copyright (c) 2024, University of Wisconsin-Madison and the author
# All rights reserved.

import numpy
from scipy.linalg import svd
from sklearn.mixture import GaussianMixture

class AmusetTICA:

    """ Tensor-based with Time-lagged Independent Component Analysis

    Parameters
    ----------
    max_rank : int, default: 0
        The maximum dimensions of the Amuset model. <=0: ignored and all
        dimensions will be used. Note: when rank overflows max_rank before
        outer product finishes, transform() will be disabled and only return
        an empty list [].

    Tutorial
    --------
      A standard protocol:
        import amuset_tica
        ttrajs = numpy.load(...)
        sigma = ...
        basis_list = amuset_tica.Basis.find(ttrajs, [2,2,2], sigma)
        ttica = amuset_tica.AmusetTICA()
        fit = ttica.fit(basis_list, ttrajs, lag_time=...)
            # equivalent to:
            # svd_v, traj_lens = ttica.build(basis_list, ttrajs)
            # ttica.covariance(svd_v, traj_lens, lag_time=...)
        print(ttica.timescales(5))
        cvs = ttica.transform(ttrajs, 3)

    """

    def __init__(self, max_rank: int=0):
        self.max_rank = max_rank
        self._basis_list = []
        self._tt_u = []
        self._tt_s = []
        self._tt_indices = []
        self._tt_intra_svd_layers = 0
        self._traj_lens = []
        #self._tt_cvs = []
        self._rank_used = 0
        self._K = []
        self._ev_K = []
        self._vr_K = []
        self._lagtime = 0
        self._timescales = []

    @property
    def rank_(self):
        return self._rank_used

    @property
    def timescales_(self):
        return self._timescales

    @property
    def eigenvalues_(self):
        return self._ev_K

    def save(self, file_name: str=""):
        """
        Save the Amuset model to a dictionary or a file

        Parameters
        ----------
        file_name : str, default empty
            an NPZ file name to save the Amuset model
            will not save to file if file_name is empty

        Return
        ------
        dic : dictionary
            a dictionary containing all element of the Amuset model

        """

        dic = {}
        dic['max_rank']     = self.max_rank
        #dic['basis_list']  = self._basis_list
        dic['n_basis_list'] = len(self._basis_list)
        for i in range(len(self._basis_list)):
            dic['basis_list_'+str(i)] = self._basis_list[i]
        dic['n_tt_layers']  = len(self._tt_u)
        for i in range(len(self._tt_u)):
            dic['tt_u_'+str(i)] = self._tt_u[i]
            dic['tt_s_'+str(i)] = self._tt_s[i]
            dic['tt_indices_'+str(i)] = self._tt_indices[i]
        #dic['tt_u']         = self._tt_u
        #dic['tt_s']         = self._tt_s
        #dic['tt_indices']   = self._tt_indices
        dic['rank']         = self._rank_used
        dic['K']            = self._K
        dic['ev_K']         = self._ev_K
        dic['vr_K']         = self._vr_K
        dic['lagtime']      = self._lagtime
        dic['timescales']   = self._timescales

        if len(file_name)>0:
            numpy.savez_compressed(file_name, **dic)

        return dic

    def load(self, src):
        """
        Load the Amuset model from a dictionary or a file

        Parameters
        ----------
        src : dictionary or string
            a dictionary generated by save() or file name of this dictionary

        Return
        ------
        self : object

        """

        if isinstance(src, str):
            dic = numpy.load(src)
            dic = {key:dic[key] for key in dic.files}
        else:
            dic = src

        self.max_rank       = float(dic['max_rank'])
        self._basis_list = []
        for i in range(dic['n_basis_list']):
            self._basis_list.append(dic['basis_list_'+str(i)])
        self._tt_u = []
        self._tt_s = []
        self._tt_indices = []
        for i in range(dic['n_tt_layers']):
            self._tt_u.append(dic['tt_u_'+str(i)])
            self._tt_s.append(dic['tt_s_'+str(i)])
            self._tt_indices.append(dic['tt_indices_'+str(i)])
        self._rank_used     = int(dic['rank'])
        self._K             = dic['K']
        self._ev_K          = dic['ev_K']
        self._vr_K          = dic['vr_K']
        if 'lagtime' in dic.keys():
            self._lagtime       = int(dic['lagtime'])
        self._timescales    = dic['timescales']


    def does_basis_overflow(self, basis_list):
        """
        Check if basis overflows max_rank when doing outer product
        If this function returns True, then model will be very inaccuratte

        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features
    
        Returns
        -------
            True (will overflow) or False (will not overflow)

        """

        if self.max_rank>0:
            outer_rank_overflow = 1
            for i in range(len(basis_list)-1):
                outer_rank_overflow *= len(basis_list[i])+1
            if outer_rank_overflow > self.max_rank:
                return True
        return False

    def _convert_sequences(self, sequences):
        """
        Convert sequences into the Amuset compatible structure

        Parameters
        ----------
        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of trajectories

        Returns
        -------
        data_matrix : 2D array, [n_features, n_all_frames]
            the converted sequences

        traj_lens : list, [n_trajs]
            lengths of each trajectory

        """
        #data_matrix = []
        #for k in range(len(sequences)):
        #    data_matrix.append(sequences[k].T)
        #data_matrix = numpy.hstack(data_matrix)

        #traj_lens = []
        #for k in range(len(sequences)):
        #    traj_lens.extend([len(sequences[k])])

        #return data_matrix, traj_lens

        return _convert_sequences(sequences)

    def _build_outer_product(self, basis_list, data_matrix, build_model: bool):
        """
        Build the outer product of Amuset
    
        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features
    
        data_matrix : 2D array, [n_features, n_all_frames]
            converted feature sequences

        build_model : bool
            True if building Amuset model, False if transforming Amuset model
    
        Returns
        -------
        outer : 2D array, [n_features, n_all_frames]
            the outer product of all basis
    
        """
      # convert data to Amuset compatible format
        data_length = data_matrix.shape[1]
        i_intra_svd_layer = 0

      # outer product
        outer = [numpy.ones(data_length)]
        for il in range(len(basis_list)):
          # force to use tensor-train structure if max_rank is set
          # dimensionality reduction if number of basis overflows max_rank
            # if self.max_rank>0 and len(outer)>self.max_rank :
            if self.max_rank>0:
                if build_model:
                    u, s, v = svd(outer, full_matrices=False, overwrite_a=True, check_finite=False)
                    indices = numpy.argsort(s)[::-1]
                    v = v[indices, :]
                    outer = v
                    self._tt_u.append(u)
                    self._tt_s.append(s)
                    self._tt_indices.append(indices)
                    #print("build_svd: len(self._tt_s)=%d, shape(self._tt_s[-1])=%s"%(len(self._tt_s), numpy.array(self._tt_s[-1]).shape))
                else:
                    u = self._tt_u[i_intra_svd_layer]
                    s = self._tt_s[i_intra_svd_layer]
                    indices = self._tt_indices[i_intra_svd_layer]
                    transform_v = numpy.matmul(numpy.matmul(numpy.diag(1.0/s), numpy.array(u).T), outer)
                    transform_v = transform_v[indices, :]
                    outer = transform_v
                    #print("apply_svd: s[%d].shape= %s"%(i_intra_svd_layer, numpy.array(s).shape))
                i_intra_svd_layer += 1

            if self.max_rank>0 and len(outer)>self.max_rank:
                outer = outer[:self.max_rank]

            if build_model:
                self._tt_intra_svd_layers = i_intra_svd_layer

          # outer product

            outer_a = outer
    
            outer_b = [numpy.ones(data_length)]
            for i in range(len(basis_list[il])):
                mean  = basis_list[il][i][0]
                sigma = basis_list[il][i][1]
                outer_b.append(numpy.exp(-numpy.array(data_matrix[il]-mean)**2/2/sigma**2))
    
            outer = []
            for i in range(len(outer_a)):
                for j in range(len(outer_b)):
                    outer.append(outer_a[i] * outer_b[j])   

        return outer

    def build(self, basis_list, sequences):
        """
        Build the tensor structure of Amuset
    
        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features
    
        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of feature sequences
    
        Returns
        -------
        outer_product : 2D array, [n_basis, n_all_frames] 
            the outer product of all basis

        traj_lens: list, [n_trajs]
            a list of lengths of all sequences 
    
        """

        self._basis_list = basis_list

        self._tt_u = []
        self._tt_s = []
        self._tt_indices = []

      # convert data to Amuset compatible format
        data_matrix, traj_lens = self._convert_sequences(sequences)
        self._traj_lens = traj_lens 

      # build the outer product
        outer = self._build_outer_product(basis_list, data_matrix, True)

      # build the Amuset
        u, s, cvs = svd(outer, full_matrices=False, overwrite_a=True, check_finite=False)
        indices = numpy.argsort(s)[::-1]
        cvs = cvs[indices, :]

        if self.max_rank>0 and len(cvs[0])>self.max_rank:
            cvs = cvs[: self.max_rank]

        self._tt_u.append(u)
        self._tt_s.append(s)
        self._tt_indices.append(indices)
        #self._tt_cvs = cvs
        self._rank_used = len(cvs)

        return cvs, traj_lens

    def covariance(self, input_data, traj_lens, lag_time: int) :
        """
        Compute time-lagged Koopman matrix
    
        Parameters
        ----------
        input_data : 2D array, [n_basis, n_all_frames]
            the right singular vector of Amuset

        traj_lens : list, [n_trajs]
            a list of lengths of all sequences
    
        lag_time : integer
            the lag time used to compute the time-lagged correlation matrix

        Returns
        -------
        self : object

        """

      # Loop over sequences to get indices arries
        x_indices = numpy.array([], dtype=int)
        y_indices = numpy.array([], dtype=int)
        pos = 0
        for i in range(len(traj_lens)):
            x_indices = numpy.concatenate((x_indices, numpy.arange(pos, pos + traj_lens[i] - lag_time)))
            y_indices = numpy.concatenate((y_indices, numpy.arange(pos + lag_time, pos + traj_lens[i])))
            pos += traj_lens[i]
    
      # compute the Koopman matrix
        x_of_input = input_data[:, x_indices]
        y_of_input = input_data[:, y_indices]
        C00 = numpy.matmul(x_of_input, x_of_input.T)
        C11 = numpy.matmul(y_of_input, y_of_input.T)
        C01 = numpy.matmul(x_of_input, y_of_input.T)
        K = numpy.matmul(numpy.linalg.inv(C00+C11), C01+C01.T)
    
      # eigen decomposition of the Koopman matrix
        evK, vrK = numpy.linalg.eig(K)
        idx = numpy.argsort(evK)[::-1]
        evK = evK[idx]
        vrK = vrK[:,idx]

        self._K = K
        self._ev_K = evK
        self._vr_K = vrK
        self._lagtime = lag_time

      # timescales
        ev_out = []
        for i in range(1, len(self._ev_K)):
            if self._ev_K[i]>0:
                ev_out.extend([self._ev_K[i]])
            else:
                break
        self._timescales = -lag_time/numpy.log(ev_out)

      # done 
        return self 

    def fit(self, basis_list, sequences, lag_time: int) : 
        """
        Build the tensor structure of Amuset

        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of feature sequences

        lag_time : integer
            the lag time used to compute the time-lagged correlation matrix
    
        Returns
        -------
        self : object

        """

        outer_product, traj_lens = self.build(basis_list, sequences)
        return self.covariance(outer_product, traj_lens, lag_time)

    def _transform(self, input_data, traj_lens, cvs_list, use_right_vr: bool, do_amuset_tica: bool) :
        """
        Apply the Amuset model to the input_data
    
        Parameters
        ----------
        input_data : 2D array, [n_basis, n_all_frames]
            the right singular vector of Amuset

        traj_lens : list, [n_trajs]
            a list of lengths of all sequences

        cvs_list : list
            A list of cvs elements to output

        use_right_vr : bool
            True: use the right eigenvector of Koopman matrix
            False: use the left eigenvector of Koopman matrix
    
        do_amuset_tica: bool, default: False
            True: do AmusetTICA, CVs are orthogonalized with eigenvectors of K
            False: do Amuset, CVs are original right singular vector
    
        Returns
        -------
        cvs : 3D array like
            The collective-variable sequences
            shape: [n_trajs, n_frames, len(cvs_list)]
    
        """

        if do_amuset_tica:
            vr_Koopman = self._vr_K
            if use_right_vr:
                all_data = numpy.matmul(vr_Koopman.T, input_data)[cvs_list].T
            else:
                all_data = numpy.matmul(numpy.linalg.inv(vr_Koopman), input_data)[cvs_list].T
        else:
            all_data = input_data[cvs_list].T
    
        cvs = []
        op = 0
        for it in range(len(traj_lens)):
            cvs.append(all_data[op:op+traj_lens[it], :])
            op += traj_lens[it]
    
        return cvs

    def transform(self, sequences, cvs_list, use_right_vr: bool=True, _do_amuset_tica: bool=True) :
        """
        Apply the Amuset model to the input_data
    
        Parameters
        ----------
        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of feature sequences to be transformed

        cvs_list : a list or a integer
            list : a list of cvs elements to output
            integer : the number of cvs elements to output

        use_right_vr : bool, default: True
            True: use the right eigenvector of Koopman matrix
            False: use the left eigenvector of Koopman matrix

        _do_amuset_tica : bool, default: True
            True: CVs are orthogonalized with eigenvectors of K
            False: CVs are not orthogonalized, just top singular vectors
    
        Returns
        -------
        cvs : 3D list, [n_trajs][n_frames, n_cvs]
            The collective-variable sequences
    
        """

        basis_list = self._basis_list

        data_matrix, traj_lens = self._convert_sequences(sequences)
        outer = self._build_outer_product(self._basis_list, data_matrix, False)
        transform_v = numpy.matmul(numpy.matmul(numpy.diag(1.0/self._tt_s[-1]), numpy.array(self._tt_u[-1]).T), outer)
        transform_v = transform_v[self._tt_indices[-1], :]
        if self.max_rank>0 and len(transform_v[0])>self.max_rank:
            transform_v = transform_v[:self.max_rank]

        if isinstance(cvs_list, int):
            cvs_list = range(1, 1+cvs_list)
        #return self._transform(transform_v, traj_lens, cvs_list, use_right_vr, True)
        return self._transform(transform_v, traj_lens, cvs_list, use_right_vr, _do_amuset_tica)


class Basis :
    """ Basis list tools for Amuset """

    @staticmethod
    def size(basis_list):
        """
        Estimate the number of basis in the basis list

        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        Returns
        -------
        size : int
            number of basis in the basis_list

        """
        ret = 1
        for i in range(len(basis_list)):
            ret *= len(basis_list[i]) + 1
        return ret

    @staticmethod
    def _find_by_GMM(sequences, n_basis_list, sigma: float=-1, random_seed: int=0):
        """
        Find the basis list with Gaussian Mixture Model of scipy

        Parameters
        ----------
        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of sequences

        n_basis_list : list
            a list of number of basis of each feature.
            e.g.: [4, 3, 2] : 4 for tic1, 3 for tic2, 2 for tic3

        sigma : float, default -1
            predefined sigma instead of covariances of data
            will use the covariances of Gaussian distribtuions by default

        random_seed : int, default: 0
            the random seed of Gaussian mixture model

        Returns
        -------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        """
      # convert sequences
        data_matrix = []
        for k in range(len(sequences)):
            data_matrix.append(sequences[k].T)
        data_matrix = numpy.hstack(data_matrix)
        traj_lens = []
        for k in range(len(sequences)):
            traj_lens.extend([len(sequences[k])])
    
        mtics = data_matrix
        len_tics = traj_lens
    
      # generate basis list
        basis_list = []
        for itic in range(len(n_basis_list)):
            this_basis_list = []
            gm = GaussianMixture(n_components=n_basis_list[itic], random_state=random_seed)
            gmf = gm.fit(mtics[itic].reshape([mtics[itic].shape[0],1]))
            means = gmf.means_.reshape(n_basis_list[itic])
            sigmas = gmf.covariances_.reshape(n_basis_list[itic])
            for i in range(n_basis_list[itic]):
                if sigma > 0:
                    this_basis_list.append([means[i], sigma])
                else:
                    this_basis_list.append([means[i], sigmas[i]])
            basis_list.append(this_basis_list)
    
        return basis_list

    @staticmethod
    def mix(basis_list):
        """
        Combine basis of all dimensions, and duplicate to all dimensions
        This function will generate a large and good basis set.

        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        Returns
        -------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            a new basis list with mixing of all basis of input

        """
        combined_basis_list = []
        for i in range(len(basis_list)):
            combined_basis_list.extend(basis_list[i])
        ret = []
        for i in range(len(basis_list)):
            ret.append(combined_basis_list)
        return ret

    @staticmethod
    def find(sequences, n_basis_list, sigma: float, random_seed: int=0, mix: bool=True):
        """
        Find the basis list with Gaussian Mixture Model

        Parameters
        ----------
        sequences : 3D list, [n_trajs][n_frames, n_features]
            a list of sequences

        n_basis_list : list
            a list of number of basis of each feature.
            e.g.: [4, 3, 2] : 4 for tic1, 3 for tic2, 2 for tic3

        sigma : float
            predefined sigma instead of covariances of data
            will use the covariances of Gaussian distribtuions if <= 0

        random_seed : int, default: 0
            the random seed of Gaussian mixture model

        mix : bool, default: True
            mix the basis obtained by the Gaussian Mixture Model

        Returns
        -------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        """
        if mix :
            basis_list_base = Basis._find_by_GMM(sequences, n_basis_list, sigma, random_seed)
            basis_list = Basis.mix(basis_list_base)
            return basis_list
        else :
            basis_list = Basis._find_by_GMM(sequences, n_basis_list, sigma, random_seed)
            return basis_list

    @staticmethod
    def scale_sigma(basis_list, sigma: float):
        """
        Scale the covariances of Gaussian basis with given number

        Parameters
        ----------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            list of Gaussian basis for each of the features

        sigma : float
            a number to scale all covariances of Gaussian basis

        Returns
        -------
        basis_list : 3D list, [[[mean,sigma], ...], ... ]
            a new basis list with scaled covariances

        """
        new_basis_list = []
        for i in range(len(basis_list)):
            this_basis_list = []
            for j in range(len(basis_list[i])):
                this_basis_list.append([basis_list[i][j][0], basis_list[i][j][1]*sigma])
            new_basis_list.append(this_basis_list)
        return new_basis_list

def _convert_sequences(sequences):
    """  
    Convert sequences into the Amuset compatible structure

    Parameters
    ----------
    sequences : 3D list, [n_trajs][n_frames, n_features]
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
    Convert Amuset sequence to sequences

    Parameters
    ----------
    data : 2D Array, [n_dim, n_all_frames] or [n_all_frames, n_dim]
        the Amuset sequence

    traj_lens : list, [n_trajs]
        lengths of each trajectory

    Return
    ------
    sequences : 3D list, [n_traj][n_frames, n_dim]
        a list of trajectories
    """

    if len(data)>len(data[0]):
        data_ = data
    else:
        data_ = data.T
    seq = []
    op = 0
    for it in range(len(traj_lens)):
        seq.append(data_[op:op+traj_lens[it], :])
        op += traj_lens[it]
    return seq

