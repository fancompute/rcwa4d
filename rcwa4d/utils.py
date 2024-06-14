import os
import sys
import numpy as np
import itertools
from numpy import cos,sin
from numpy.linalg import solve as bslash ### S4 mentions it is more efficient to FFT the inverted epsilon, but here we are inverting FFT matrix 
from scipy.linalg import block_diag
from copy import copy
from tqdm import tqdm
# import cmath
# import configparser
# import pdb
# import pickle
# import time


class scatter_matrices:
    def A(self, W_layer, Wg, V_layer, Vg): # PLUS SIGN
        '''
        OFFICIAL EMLAB prescription
        inv(W_layer)*W_gap
        :param W_layer: layer E-modes
        :param Wg: gap E-field modes
        :param V_layer: layer H_modes
        :param Vg: gap H-field modes
        # the numbering is just 1 and 2 because the order differs if we're in the structure
        # or outsid eof it
        :return:
        '''
        # assert type(W_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Wg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(V_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Vg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

        #A = np.linalg.inv(W_layer) * Wg + np.linalg.inv(V_layer) * Vg;
        A = bslash(W_layer, Wg) + bslash(V_layer, Vg);

        return A;

    def B(self, W_layer, Wg, V_layer, Vg): #MINUS SIGN
        '''
        :param W_layer: layer E-modes
        :param Wg: gap E-field modes
        :param V_layer: layer H_modes
        :param Vg: gap H-field modes
        # the numbering is just 1 and 2 because the order differs if we're in the structure
        # or outsid eof it
        :return:
        '''

        # assert type(W_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Wg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(V_layer) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Vg) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

        #B = np.linalg.inv(W_layer) * Wg - np.linalg.inv(V_layer) * Vg;
        B = bslash(W_layer,Wg) - bslash(V_layer, Vg);

        return B;


    def A_B_matrices_half_space(self, W_layer, Wg, V_layer, Vg):
        ## this function is needed because for the half-spaces (reflection and transmission
        # spaces, the convention for calculating A and B is DIFFERENT than inside the main layers
        a = self.A(Wg, W_layer, Vg, V_layer);
        b = self.B(Wg, W_layer, Vg, V_layer);
        return a, b;


    def A_B_matrices(self, W_layer, Wg, V_layer, Vg):
        '''
        single function to output the a and b matrices needed for the scatter matrices
        :param W_layer: gap
        :param Wg:
        :param V_layer: gap
        :param Vg:
        :return:
        '''
        a = self.A(W_layer, Wg, V_layer, Vg);
        b = self.B(W_layer, Wg, V_layer, Vg);
        return a, b;

    def S_layer(self, A,B, Li, k0, modes):
        '''
        function to create scatter matrix in the ith layer of the uniform layer structure
        we assume that gap layers are used so we need only one A and one B
        :param A: function A =
        :param B: function B
        :param k0 #free -space wavevector magnitude (normalization constant) in Si Units
        :param Li #length of ith layer (in Si units)
        :param modes, eigenvalue matrix
        :return: S (4x4 scatter matrix) and Sdict, which contains the 2x2 block matrix as a dictionary
        '''
        # assert type(A) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(B) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

        #sign convention (EMLAB is exp(-1i*k\dot r))
        X_i = np.diag(np.exp(-np.diag(modes)*Li*k0)); #never use expm

        #term1 = (A - X_i * B * A.I * X_i * B).I
        # S11 = term1 * (X_i * B * A.I * X_i * A - B);
        # S12 = term1 * (X_i) * (A - B * A.I * B);
        # S22 = S11;
        # S21 = S12;
        term1 = (A - X_i @ B @ bslash(A, X_i) @ B)
        S11 = bslash(term1, (X_i @ B @ bslash(A,X_i) @ A - B));
        S12 = bslash(term1, (X_i) @ (A - B @ bslash(A, B)));
        S22 = S11;
        S21 = S12;

        S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
        S = np.block([[S11, S12], [S21, S22]]);
        return S, S_dict;


    def S_R(self, Ar, Br):
        '''
        function to create scattering matrices in the reflection regions
        different from S_layer because these regions only have one boundary condition to satisfy
        :param Ar:
        :param Br:
        :return:
        '''
        # assert type(Ar) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Br) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

        #
        # S11 = -np.linalg.inv(Ar) * Br;
        # S12 = 2*np.linalg.inv(Ar);
        # S21 = 0.5*(Ar - Br * np.linalg.inv(Ar) * Br);
        # S22 = Br * np.linalg.inv(Ar)

        S11 = -bslash(Ar,Br);
        S12 = 2*np.linalg.inv(Ar);
        S21 = 0.5*(Ar - Br @ bslash(Ar,Br));
        S22 = Br @ np.linalg.inv(Ar)
        S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
        S = np.block([[S11, S12], [S21, S22]]);
        return S, S_dict;

    def S_T(self, At, Bt):
        '''
        function to create scattering matrices in the transmission regions
        different from S_layer because these regions only have one boundary condition to satisfy
        :param At:
        :param Bt:
        :return:
        '''
        # assert type(At) == np.matrixlib.defmatrix.matrix, 'not np.matrix'
        # assert type(Bt) == np.matrixlib.defmatrix.matrix, 'not np.matrix'

        # S11 = (Bt) * np.linalg.inv(At);
        # S21 = 2*np.linalg.inv(At);
        # S12 = 0.5*(At - Bt * np.linalg.inv(At) * Bt);
        # S22 = - np.linalg.inv(At)*Bt
        S11 = (Bt) @ np.linalg.inv(At);
        S21 = 2*np.linalg.inv(At);
        S12 = 0.5*(At - Bt @ bslash(At,Bt));
        S22 = - bslash(At,Bt)
        S_dict = {'S11': S11, 'S22': S22,  'S12': S12,  'S21': S21};
        S = np.block([[S11, S12], [S21, S22]]);
        return S, S_dict;

class eigen_modes:
    def eigen_W(self, Gamma_squared):
        '''
        for the E_field
        use: you would only really want to use this if the media is anisotropic in any way
        :param Gamma: matrix for the scattering formalism
        :return:
        '''
        #could be an issue with how eig sorts eigenvalues in the output
        lambda_squared, W = np.linalg.eig(Gamma_squared);  # LAMBDa is effectively refractive index
        lambda_matrix = np.diag(np.sqrt(lambda_squared.astype('complex'))); ### ??? might need check branch cut here too
        return W, lambda_matrix

    def eigen_V(self, Q, W, lambda_matrix):
        #V = Q*W*(lambda)^-1
        '''
        eigenmodes for the i*eta*H field
        :param Q: Q matrix
        :param W: modes from eigen W
        :param lambda_matrix: eigen values from W
        :return:
        '''
        return Q@W@np.linalg.inv(lambda_matrix);

class rcwa_initial_conditions:
    def delta_vector(self, P, Q):
        '''
            create a vector with a 1 corresponding to the 0th order
            #input P = 2*(num_ord_specified)+1
        '''
        fourier_grid = np.zeros((P,Q))
        fourier_grid[int(P/2), int(Q/2)] = 1;
        # vector = np.zeros((P*Q,));
        #
        # #the index of the (0,0) element requires a conversion using sub2ind
        # index = int(P/2)*P + int(Q/2);
        vector = fourier_grid.flatten();
        return np.matrix(np.reshape(vector, (1,len(vector))));

    def delta_vector_1D(self, P):
        '''
            create a vector with a 1 corresponding to the 0th order
        '''
        vector = np.zeros((P,));

        #the index of the (0,0) element requires a conversion using sub2ind
        index = int(P/2);
        vector[index] = 1
        return vector;

    def initial_conditions_1D(self, K_inc_vector, theta, P):
        '''
        K_inc points only in X and Z plane, so theta is the only specifying angle
        :param K_inc_vector:
        :param theta:
        :param P:
        :return:
        '''
        num_ord = 2*P+1;
        delta = delta_vector_1D(num_ord);
        cinc = delta


    def initial_conditions(self, K_inc_vector, theta, normal_vector, pte, ptm, P, Q):
        '''
        :param K_inc_vector: whether it's normalized or not is not important...
        :param theta: angle of incience
        :param ate_vector:
        :param normal_vector: pointing into z direction
        :param pte: te polarization amplitude
        :param ptm: tm polarization amplitude
        :return:
        calculates the incident E field, cinc, and the polarization fro the initial condition vectors
        '''
        if (theta != 0):
            ate_vector = np.cross(K_inc_vector, normal_vector);
            ate_vector = ate_vector / (np.linalg.norm(ate_vector));
        else:
            ate_vector = np.array([0,1,0]);

        atm_vector = np.cross(ate_vector, K_inc_vector);
        atm_vector = atm_vector / (np.linalg.norm(atm_vector))

        Polarization = pte * ate_vector + ptm * atm_vector; #total E_field incident which is a 3 component vector (ex, ey, ez)
        E_inc = Polarization;
        # go from mode coefficients to FIELDS
        Polarization = np.squeeze(np.array(Polarization));
        delta = self.delta_vector(2*P+1,2*Q+1);

        #cinc
        esrc = np.hstack((Polarization[0]*delta, Polarization[1]*delta));
        esrc = np.matrix(esrc).T; #mode amplitudes of Ex, and Ey

        return E_inc, esrc, Polarization

class homogeneous_layer:
    def homogeneous_module(self, Kx, Ky, e_r = 1, m_r = 1, k0=None):
        '''
        homogeneous layer is much simpler to do, so we will create an isolated module to deal with it
        :return:
        '''
        assert type(Kx) == np.ndarray, 'not np.array'
        assert type(Ky) == np.ndarray, 'not np.array'
        j = 1j;
        N = len(Kx);
        I = np.identity(N);
        P = (e_r**-1)*np.block([[Kx*Ky, e_r*m_r*I-Kx**2], [Ky**2-m_r*e_r*I, -Ky*Kx]])
        Q = (e_r/m_r)*P;
        W = np.identity(2*N)
        # arg = (m_r*e_r*I-Kx**2-Ky**2); #arg is +kz^2
        # arg = arg.astype('complex');
        # Kz = np.conj(np.sqrt(arg)); #conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
        ### previous implementation
        arg = (m_r*e_r*I*k0**2-(Kx*k0)**2-(Ky*k0)**2); #arg is +kz^2
        Kz = arg.astype('complex')
        for i in range(len(Kz)):
            if Kz[i, i].real < 0:
                Kz[i, i] = -1j * np.sqrt(-Kz[i, i])
                # print('caught one!')
            else:
                Kz[i, i] = np.sqrt(Kz[i, i])
                # print('fine')
        Kz/=k0
        ### end of previous implementation
        ### to guarantee analytic continuation, sqrt all Kz^2 to Kz choice below im=re line
        ### since for real frequency, kz will be either positive real or negatiev imaginary
        ### new implementation
        # arg = (m_r*e_r*I*k0**2-(Kx*k0)**2-(Ky*k0)**2); #arg is +kz^2
        # # arg = (m_r*e_r*I-(Kx)**2-(Ky)**2); #arg is +kz^2
        # Kz = arg.astype('complex')
        # Kz1 = np.copy(Kz)
        # Kz2 = np.copy(Kz)
        # # print('before sqrt',np.around(np.diag(Kz),5))
        # for i in range(len(Kz)):
        #     Kz1[i, i] = np.sqrt(Kz[i, i])
        #     if Kz[i, i].real < 0:
        #         Kz2[i, i] = -1j * np.sqrt(-Kz[i, i])
        #         # print('caught one!',Kz[i, i])
        #     else:
        #         Kz2[i, i] = np.sqrt(Kz[i, i])
        #         # print('fine',Kz[i, i])
        #     # if Kz1[i,i] != Kz2[i,i]:
        #     #     print(Kz[i,i], Kz1[i,i], Kz2[i,i])
        # # print('after sqrt',np.around(np.diag(Kz),2))
        # # Kz/=k0
        # Kz=Kz2/k0
        # # print('normalized',np.around(np.diag(Kz),2))
        # # pdb.set_trace()
        ### end of new implementation
        ### !!! above is the key to branching cut choice for solving Smat pole
        ### compare to the old util function!
        # Kz = -np.conj(np.sqrt(arg));  # to agree with benchmark data?!
        eigenvalues = block_diag(j*Kz, j*Kz) #determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
        ### ??? this is where the convention of \pm kz kicks in?!
        #W is just identity matrix
        V = Q@np.linalg.inv(eigenvalues); #eigenvalue order is arbitrary (hard to compare with matlab
        #alternative V with no inverse
        #V = np.matmul(np.linalg.inv(P),np.matmul(Q,W)); apparently, this fails because P is singular

        # N = len(W)//2
        # print(W.shape, V.shape)
        # for i in range(N):
        #     print(i)
        #     print(W[:N,i]*V[N:,i] - W[N:,i]*V[:N,i])
        #     print(np.sum(W[:N,i]*V[N:,i] - W[N:,i]*V[:N,i]))

        return W,V,Kz

    def homogeneous_1D(self, Kx, e_r, m_r = 1):
        '''
        efficient homogeneous 1D module
        :param Kx:
        :param e_r:
        :param m_r:
        :return:
        '''
        j = 1j;

        I = np.identity(len(Kx));
        P = e_r*I - Kx**2;
        Q = I;
        arg = (m_r*e_r*I-Kx**2); #arg is +kz^2
        arg = arg.astype('complex');

        Kz = np.conj(np.sqrt(arg)); #conjugate enforces the negative sign convention (we also have to conjugate er and mur if they are complex)
        eigenvalues = j*Kz #determining the modes of ex, ey... so it appears eigenvalue order MATTERS...
        V = np.matmul(Q,eigenvalues); #eigenvalue order is arbitrary (hard to compare with matlab
        return I,V, Kz



class redheffer_star:
    def dict_to_matrix(self, S_dict):
        '''
        converts dictionary form of scattering matrix to a np.matrix
        :param S_dict:
        :return:
        '''
        return np.block([[S_dict['S11'], S_dict['S12']],[S_dict['S21'], S_dict['S22']]]);


    def RedhefferStar(self, SA,SB): #SA and SB are both 2x2 block matrices;
        '''
        RedhefferStar for arbitrarily sized 2x2 block matrices for RCWA
        :param SA: dictionary containing the four sub-blocks
        :param SB: dictionary containing the four sub-blocks,
        keys are 'S11', 'S12', 'S21', 'S22'
        :return:
        '''

        assert type(SA) == dict, 'not dict'
        assert type(SB) == dict, 'not dict'

        # once we break every thing like this, we should still have matrices
        SA_11 = SA['S11']; SA_12 = SA['S12']; SA_21 = SA['S21']; SA_22 = SA['S22'];
        SB_11 = SB['S11']; SB_12 = SB['S12']; SB_21 = SB['S21']; SB_22 = SB['S22'];
        N = len(SA_11) #SA_11 should be square so length is fine
        I = np.matrix(np.identity(N));

        # D = np.linalg.inv(I-SB_11*SA_22);
        # F = np.linalg.inv(I-SA_22*SB_11);
        #
        # SAB_11 = SA_11 + SA_12*D*SB_11*SA_21;
        # SAB_12 = SA_12*D*SB_12;
        # SAB_21 = SB_21*F*SA_21;
        # SAB_22 = SB_22 + SB_21*F*SA_22*SB_12;

        D = (I-SB_11@SA_22);
        F = (I-SA_22@SB_11);

        SAB_11 = SA_11 + SA_12@bslash(D,SB_11)@SA_21;
        SAB_12 = SA_12@bslash(D,SB_12);
        SAB_21 = SB_21@bslash(F,SA_21);
        SAB_22 = SB_22 + SB_21@bslash(F,SA_22)@SB_12;

        SAB = np.block([[SAB_11, SAB_12],[SAB_21, SAB_22]])
        SAB_dict = {'S11': SAB_11, 'S22': SAB_22,  'S12': SAB_12,  'S21': SAB_21};

        return SAB, SAB_dict;

    def RedhefferStar_with_rotation(self, SA,SB,R): #SA and SB are both 2x2 block matrices;
        '''
        RedhefferStar for arbitrarily sized 2x2 block matrices for RCWA
        :param SA: dictionary containing the four sub-blocks
        :param SB: dictionary containing the four sub-blocks,
        keys are 'S11', 'S12', 'S21', 'S22'
        :return:
        '''

        assert type(SA) == dict, 'not dict'
        assert type(SB) == dict, 'not dict'

        # R-inv and R
        R_inv = np.linalg.inv(R)

        # once we break every thing like this, we should still have matrices
        SA_11 = SA['S11']; SA_12 = SA['S12']; SA_21 = SA['S21']; SA_22 = SA['S22'];
        SB_11 = R_inv@SB['S11']@R; SB_12 = R_inv@SB['S12']; SB_21 = SB['S21']@R; SB_22 = SB['S22'];
        N = len(SA_11) #SA_11 should be square so length is fine
        I = np.matrix(np.identity(N));

        # D = np.linalg.inv(I-SB_11*SA_22);
        # F = np.linalg.inv(I-SA_22*SB_11);
        #
        # SAB_11 = SA_11 + SA_12*D*SB_11*SA_21;
        # SAB_12 = SA_12*D*SB_12;
        # SAB_21 = SB_21*F*SA_21;
        # SAB_22 = SB_22 + SB_21*F*SA_22*SB_12;

        D = (I-SB_11@SA_22);
        F = (I-SA_22@SB_11);

        SAB_11 = SA_11 + SA_12@bslash(D,SB_11)@SA_21;
        SAB_12 = SA_12@bslash(D,SB_12);
        SAB_21 = SB_21@bslash(F,SA_21);
        SAB_22 = SB_22 + SB_21@bslash(F,SA_22)@SB_12;

        SAB = np.block([[SAB_11, SAB_12],[SAB_21, SAB_22]])
        SAB_dict = {'S11': SAB_11, 'S22': SAB_22,  'S12': SAB_12,  'S21': SAB_21};

        return SAB, SAB_dict;


    def construct_global_scatter(self, scatter_list):
        '''
        this function assumes an RCWA implementation where all the scatter matrices are stored in a list
        and the global scatter matrix is constructed at the end
        :param scatter_list: list of scatter matrices of the form [Sr, S1, S2, ... , SN, ST]
        :return:
        '''
        Sr = scatter_list[0];
        Sg = Sr;
        for i in range(1, len(scatter_list)):
            Sg = RedhefferStar(Sg, scatter_list[i]);
        return Sg;


def convmat2D(A, Q, P):
    '''
    :param A: input is currently whatever the real space representation of the structure is
    :param Q: specifies max order in x (so the sum is from -Q to Q
    :param P: Pspecifies max order in y (so the sum is from -P to P
    :return:
    '''
    N = A.shape;

    NH = (2*P+1) * (2*Q+1) ;
    p = list(range(-P, P + 1)); #array of size 2Q+1
    q = list(range(-Q, Q + 1));

    ## do fft
    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fft2(A));
    # natural question is to ask what does Af consist of..., what is the normalization for?

    # central indices marking the (0,0) order
    p0 = int((N[0] / 2)); #Af grid is Nx, Ny
    q0 = int((N[1] / 2)); #no +1 offset or anything needed because the array is orders from -P to P

    C = np.zeros((NH, NH))
    C = C.astype(complex);
    for qrow in range(2*Q+1): #remember indices in the arrary are only POSITIVE
        for prow in range(2*P+1): #outer sum
            # first term locates z plane, 2nd locates y column, prow locates x
            row = (prow) * (2*Q+1) + qrow; #natural indexing
            for qcol in range(2*Q+1): #inner sum
                for pcol in range(2*P+1):
                    col = (pcol) * (2*Q+1) + qcol; #natural indexing
                    pfft = p[prow] - p[pcol]; #get index in Af; #index may be negative.
                    qfft = q[qrow] - q[qcol];
                    C[row, col] = Af[p0 + pfft, q0 + qfft]; #index may be negative.
                    # if abs(pfft)>1 or abs(qfft)>1:
                    #     C[row,col]*=0.01
    # plt.imshow(np.abs(C))
    # plt.savefig('tmp3.png')

    return C;


class K_matrix:
    def K_matrix_cubic_2D(self, beta_x, beta_y, k0, a_x, a_y, N_p, N_q):
        #    K_i = beta_i - pT1i - q T2i - r*T3i
        # but here we apply it only for cubic and tegragonal geometries in 2D
        '''
        :param beta_x: input k_x,inc/k0
        :param beta_y: k_y,inc/k0; #already normalized...k0 is needed to normalize the 2*pi*lambda/a
                however such normalization can cause singular matrices in the homogeneous module (specifically with eigenvalues)
        :param T1:reciprocal lattice vector 1
        :param T2:
        :param T3:
        :return:
        '''
        #(indexing follows (1,1), (1,2), ..., (1,N), (2,1),(2,2),(2,3)...(M,N) ROW MAJOR
        # but in the cubic case, k_x only depends on p and k_y only depends on q
        k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
        k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);

        kx, ky = np.meshgrid(k_x, k_y); #this is the N_p x N_q grid
        # final matrix should be sparse...since it is diagonal at most
        #order in flatten actually doesn't matter in the end
        Kx = np.diag(kx.flatten(order = 'C')); #default is C or column major
        Ky = np.diag(ky.flatten(order = 'C'))

        return Kx, Ky

    def K_matrix_bilayer(self, beta_x, beta_y, k0, a_x, a_y, N ,angle):
        '''
        :param beta_x: input k_x,inc/k0
        :param beta_y: k_y,inc/k0; #already normalized...k0 is needed to normalize the 2*pi*lambda/a
                however such normalization can cause singular matrices in the homogeneous module (specifically with eigenvalues)
        :param N: number of modes for m,n,m',n'
        :param angle: rotation btw two layers
        :param T3:
        :return:
        '''
        #(indexing follows (1,1), (1,2), ..., (1,N), (2,1),(2,2),(2,3)...(M,N) ROW MAJOR
        # but in the cubic case, k_x only depends on p and k_y only depends on q
        N_p=N
        N_q=N
        k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
        k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);
        print(k_x)
        dk_x = k_x*cos(angle)+k_y*sin(angle)
        dk_y = -k_x*sin(angle)+k_y*cos(angle)
        print(dk_x)

        kx, ky = np.meshgrid(k_x, k_y); #this is the N_p x N_q grid
        kx, ky = kx.flatten(order = 'C'),ky.flatten(order = 'C')
        dkx, dky = np.meshgrid(dk_x, dk_y);  # this is the N_p x N_q grid
        dkx, dky = dkx.flatten(order='C'), dky.flatten(order='C')


        NN = N * N
        kx = np.tile(dkx,NN) + np.repeat(kx,NN)
        ky = np.tile(dky,NN) + np.repeat(ky,NN)

        print(np.tile(dkx, NN))
        print(np.repeat(kx, NN))

        # final matrix should be sparse...since it is diagonal at most
        Kx = np.diag(kx); ## should be of shape [NN*NN, NN*NN]
        Ky = np.diag(ky)

        return Kx, Ky


class PQ_matrices:
    def Q_matrix(self, Kx, Ky, e_conv, mu_conv):
        '''
        pressently assuming non-magnetic material so mu_conv = I
        :param Kx: now a matrix (NM x NM)
        :param Ky: now a matrix
        :param e_conv: (NM x NM) matrix containing the 2d convmat
        :return:
        '''

        assert type(Kx) == np.ndarray, 'not array'
        assert type(Ky) == np.ndarray, 'not array'
        assert type(e_conv) == np.ndarray, 'not array'
        # print("econv Ky", bslash(e_conv, Ky))
        # print("econv Kx", bslash(e_conv, Kx))
        return np.block([[Kx @ bslash(mu_conv,Ky),  e_conv - Kx @ bslash(mu_conv, Kx)],
                                             [Ky @ bslash(mu_conv, Ky)  - e_conv, -Ky @ bslash(mu_conv, Kx)]]);


    def P_matrix(self, Kx, Ky, e_conv, mu_conv):
        assert type(Kx) == np.ndarray, 'not array'
        assert type(Ky) == np.ndarray, 'not array'
        assert type(e_conv) == np.ndarray, 'not array'
        # print("econv Ky",bslash(e_conv, Ky))
        # print("econv Kx",bslash(e_conv, Kx))
        P = np.block([[Kx @ bslash(e_conv, Ky),  mu_conv - Kx @ bslash(e_conv,Kx)],
                      [Ky @ bslash(e_conv, Ky) - mu_conv,  -Ky @ bslash(e_conv,Kx)]]);
        return P;



    def P_Q_kz(self, Kx, Ky, e_conv, mu_conv):
        '''
        r is for relative so do not put epsilon_0 or mu_0 here
        :param Kx: NM x NM matrix
        :param Ky:
        :param e_conv: (NM x NM) conv matrix
        :param mu_r:
        :return:
        '''
        argument = e_conv - Kx ** 2 - Ky ** 2 ## ??? implicitly assuming mu_r=1 here? then why pass mu_conv at all lol
        Kz = np.conj(np.sqrt(argument.astype('complex'))); ### ??? branch cut choice here?
        ### ??? we could potentially first invert epsilon then FFT in P_matrix and Q_matrix, instead of inverting e_conv
        q = self.Q_matrix(Kx, Ky, e_conv, mu_conv)
        p = self.P_matrix(Kx, Ky, e_conv, mu_conv)

        return p, q, Kz; ### seems Kz wont be used; so we don bother with details here


def expand_cinc(cinc,NM):
    cinc_new=np.zeros([len(cinc)*NM,1],dtype=complex)
    halfNM=int(NM/2)
    for nm in range(len(cinc)):
        nm_new=nm*NM+halfNM
        cinc_new[nm_new,0]=cinc[nm,0]
    return cinc_new

def expand_W_matrix(Ws,NM,layer=1):
    '''
    This is to get the Wg matrix, which is merely identity matrix;
    To exppand dense Wi or Vi matrices, use the expand_V_matrix instead
    '''
    return np.identity(2*NM*NM,dtype=complex)

def RedhefferStar(SA,SB): #SA and SB are both 2x2 block matrices;
    assert type(SA) == dict, 'not dict'
    assert type(SB) == dict, 'not dict'
    SA_11 = SA['S11']; SA_12 = SA['S12']; SA_21 = SA['S21']; SA_22 = SA['S22'];
    SB_11 = SB['S11']; SB_12 = SB['S12']; SB_21 = SB['S21']; SB_22 = SB['S22'];
    N = len(SA_11) #SA_11 should be square so length is fine
    I = np.matrix(np.identity(N));
    D = (I-SB_11@SA_22);
    F = (I-SA_22@SB_11);
    SAB_11 = SA_11 + SA_12@bslash(D,SB_11)@SA_21;
    SAB_12 = SA_12@bslash(D,SB_12);
    SAB_21 = SB_21@bslash(F,SA_21);
    SAB_22 = SB_22 + SB_21@bslash(F,SA_22)@SB_12;
    SAB = np.block([[SAB_11, SAB_12],[SAB_21, SAB_22]])
    SAB_dict = {'S11': SAB_11, 'S22': SAB_22,  'S12': SAB_12,  'S21': SAB_21};
    return SAB, SAB_dict;

def expand_S_dict(S_dicts,NM,layer=1):
    S_11=np.zeros([2*NM*NM,2*NM*NM],dtype=complex)
    S_12=np.zeros([2*NM*NM,2*NM*NM],dtype=complex)
    S_21=np.zeros([2*NM*NM,2*NM*NM],dtype=complex)
    S_22=np.zeros([2*NM*NM,2*NM*NM],dtype=complex)
    if layer==1:
        for nm in range(NM):
            S_dict=S_dicts[nm]
            for old_r in range(2*NM):
                for old_c in range(2*NM):
                    new_r=old_r*NM+nm
                    new_c=old_c*NM+nm
                    S_11[new_r,new_c]=S_dict["S11"][old_r,old_c]
                    S_12[new_r,new_c]=S_dict["S12"][old_r,old_c]
                    S_21[new_r,new_c]=S_dict["S21"][old_r,old_c]
                    S_22[new_r,new_c]=S_dict["S22"][old_r,old_c]
    elif layer==2:
        for nm in range(NM):
            S_dict=S_dicts[nm]
            for old_r in range(NM):
                for old_c in range(NM):
                    new_r=old_r+nm*NM
                    new_c=old_c+nm*NM
                    S_11[new_r,new_c]=S_dict["S11"][old_r,old_c]
                    S_12[new_r,new_c]=S_dict["S12"][old_r,old_c]
                    S_21[new_r,new_c]=S_dict["S21"][old_r,old_c]
                    S_22[new_r,new_c]=S_dict["S22"][old_r,old_c]
            for old_r in range(NM):
                for old_c in range(NM,2*NM):
                    new_r=old_r+nm*NM
                    new_c=old_c+nm*NM+NM*NM-NM
                    S_11[new_r,new_c]=S_dict["S11"][old_r,old_c]
                    S_12[new_r,new_c]=S_dict["S12"][old_r,old_c]
                    S_21[new_r,new_c]=S_dict["S21"][old_r,old_c]
                    S_22[new_r,new_c]=S_dict["S22"][old_r,old_c]
            for old_r in range(NM,2*NM):
                for old_c in range(NM):
                    new_r=old_r+nm*NM+NM*NM-NM
                    new_c=old_c+nm*NM
                    S_11[new_r,new_c]=S_dict["S11"][old_r,old_c]
                    S_12[new_r,new_c]=S_dict["S12"][old_r,old_c]
                    S_21[new_r,new_c]=S_dict["S21"][old_r,old_c]
                    S_22[new_r,new_c]=S_dict["S22"][old_r,old_c]
            for old_r in range(NM,2*NM):
                for old_c in range(NM,2*NM):
                    new_r=old_r+nm*NM+NM*NM-NM
                    new_c=old_c+nm*NM+NM*NM-NM
                    S_11[new_r,new_c]=S_dict["S11"][old_r,old_c]
                    S_12[new_r,new_c]=S_dict["S12"][old_r,old_c]
                    S_21[new_r,new_c]=S_dict["S21"][old_r,old_c]
                    S_22[new_r,new_c]=S_dict["S22"][old_r,old_c]
    else:
        print("Wasn't planning to implement layer>2")
    return {'S11': S_11, 'S22': S_22,  'S12': S_12,  'S21': S_21}

def expand_lambda(lambdas,NM,layer=1):
    '''
    lambdas: list of lambda matrices (diagonal part) for each layer; can be any other g-related variables
    return interleaved lambdas for layer=1, concatenated lambdas for layer=2
    '''
    # print('lam',len(lambdas[0])//NM,len(lambdas[0]),NM)
    res = []
    for i in range(len(lambdas[0])//NM): ### should be 2, corresponding to x and y polarizations
        # print(i)
        if layer==1:
            res += list(itertools.chain(*zip(*[lam[i*NM:(i+1)*NM] for lam in lambdas])))
        elif layer==2:
            res += list(itertools.chain(*[lam[i*NM:(i+1)*NM] for lam in lambdas]))
        else:
            print("Wasn't planning to implement layer>2")
            return None
    return np.array(res)

def expand_V_matrix(Vs,NM,layer=1):
    '''
    Vs: list of V matrices for each layer; can be any other g-related variables
    return interleaved lambdas for layer=1, concatenated lambdas for layer=2
    '''
    num_pol = len(Vs[0])//NM ### should be 2, corresponding to x and y polarizations
    if layer==2:
        V_each_pol = []
        for i in range(num_pol):
            V_each_pol.append([])
            for j in range(num_pol):
                V_each_pol[-1].append(block_diag(*[t[i*NM:(i+1)*NM,j*NM:(j+1)*NM] for t in Vs]))
        V_combined = np.block(V_each_pol)
    elif layer==1:
        V_combined = np.zeros([num_pol * NM * NM, num_pol * NM * NM],dtype=complex)
        for i in range(num_pol): ### index which polarization block
            for j in range(num_pol): ### index which polarization block
                for it,t in enumerate(Vs): ### index which plane wave basis, length NM
                    V_combined[i*NM*NM+it:(i+1)*NM*NM+it:NM, j*NM*NM+it:(j+1)*NM*NM+it:NM] = t[i*NM:(i+1)*NM,j*NM:(j+1)*NM]
    else:
        print("Wasn't planning to implement layer>2")
        return None
    return V_combined

def pk_to_pte_ptm(px,py,k_inc):
    """
    Convert the given polarization components (px, py) to TE and TM components.
    
    :param px: Polarization component along x
    :param py: Polarization component along y
    :param k_inc: Incident wave vector [kx, ky, kz]
    :return: TE and TM polarization amplitudes
    """
    kx, ky = k_inc
    kz = np.sqrt(1 - kx**2 - ky**2)
    #print(kz)
    #find te,tm components of k_inc
    k_inc_norm = np.array([kx, ky, kz]) / np.linalg.norm([kx, ky, kz])
    te_vector = np.cross(k_inc_norm, [0, 0, 1])
    #te_vector /= np.linalg.norm(te_vector)
    
    tm_vector = np.cross(te_vector, k_inc_norm)
    #tm_vector /= np.linalg.norm(tm_vector)
    
    #take in line component of px,py with te,tm
    p_vector = np.array([px, py, 0])
    pte = np.dot(p_vector, te_vector)
    ptm = np.dot(p_vector, tm_vector)
    #print(pte,ptm)
    
    return pte, ptm


def theta_phi_from_kincs(k_incs):
    ### TODO
    ### k_incs does not have 2π or freq inside
    return


def get_real_space_bases(k0, gxs, gys, real_space_x_grid, real_space_y_grid):
    '''
    gxs, gys: [nG], will be unnormalized by k0
    '''
    phase = -1j*(k0*gxs.reshape(-1,1,1)*np.expand_dims(real_space_x_grid,axis=0) 
            + k0*gys.reshape(-1,1,1)*np.expand_dims(real_space_y_grid,axis=0))
    return np.exp(phase)


def field_fourier_to_real(coefs, real_space_bases): ### 
    '''
    coefs: nk,nG
    real_space_bases: [nG,nX,nY]
    '''
    return np.tensordot(coefs,real_space_bases,axes=([-1],[0]))


class SummedRCWA():
    def __init__(self, obj_ref, freq, k_incs, amps, px=1, py=0,
                    x_min=-10, x_max=10, y_min=-10, y_max=10, num_pts=100):
        '''
        Input:
        k_incs: [nk,2]
        amps: complex, will be normalized
        px: polarization(s) along x
        py: polarization(s) along y
        x_min, x_max, y_min, y_max, num_pts: spatial extent specifications
        '''
        xs =  np.linspace(x_min, x_max, num_pts)
        ys =  np.linspace(y_min, y_max, num_pts)
        self.real_space_x_grid, self.real_space_y_grid = np.meshgrid(xs,ys)
        self.real_space_bases = None
        self.objs = [] #### for all kincs
        self.freq = freq
        self.k_incs = np.array(k_incs)
        self.amps = np.array(amps).flatten()
        assert self.k_incs.shape[0] == self.amps.shape[0], "k_incs and amps should have same length"
        self.kzs = np.sqrt(1-np.sum(k_incs**2,axis=-1)) ### TODO: check this
        self.px = px
        self.py = py
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

        ### to get Gx Gy for normal incidence
        self.obj_ref = obj_ref
        obj_ref.set_freq_k(freq,(0,0))
        self.gxs = np.diag(obj_ref.Kx)
        self.gys = np.diag(obj_ref.Ky)
        for k_inc in k_incs:
            obj = copy(obj_ref)
            pte,ptm = pk_to_pte_ptm(px,py,k_inc)
            obj.set_freq_k(freq, kxy_inc=k_inc)
            self.objs.append(obj)

    def total_RT(self, normalize=True):
        '''
        if normalize, should be btw 0 and 1
        '''
        for k_inc,obj in zip(self.k_incs,self.objs):
            pte,ptm = pk_to_pte_ptm(self.px,self.py,k_inc)
            R,T = obj.get_RT(pte,ptm,storing_intermediate_Smats=True)
        return R,T
    def get_field(self, which_layer=0, z_offset=0, real_space=True):
        ### need to run total_RT first under desired polarization
        ### TODO: extend to internal fields case
        ### without x,y,z phases
        # real_space_bases = get_real_space_bases(self.obj_ref.k0, self.gxs, self.gys, self.real_space_x_grid, self.real_space_y_grid)
        fields = []
        # self.real_space_bases = []
        for obj,k_inc,kz in tqdm(zip(self.objs,self.k_incs,self.kzs)): 
            _, field = obj.get_RT_field()
            field = field.reshape(6,-1) ### [6,nG]
            #print(field)
            if not real_space:    
                fields.append(field) ### [6,nG]
            else:
                real_space_bases = get_real_space_bases(obj.k0, np.diag(obj.Kx), np.diag(obj.Ky), self.real_space_x_grid, self.real_space_y_grid)
                field = field_fourier_to_real(field,real_space_bases) ### [6,nX,nY]
                
                field *= np.exp(1j*obj.k0*kz*z_offset)
                fields.append(field)
                # self.real_space_bases.append(real_space_bases) ### debugging only
        fields = np.array(fields) ### nk,6,nG or nk,6,nX,nY]
        print(self.amps.shape,fields.shape)
        return np.tensordot(self.amps,fields,axes=([-1],[0]))





