from .utils import *
hl=homogeneous_layer()
sm=scatter_matrices()
pq=PQ_matrices()
em=eigen_modes()
ic=rcwa_initial_conditions()
km=K_matrix()
# rs=redheffer_star()



def K_matrix_given_mn(beta_x, beta_y, k0, a_x, a_y, N_x, N_y, n, m, angle, layer=1): ## should be called with opposite angles for two layers
    if layer==1:
        dkx = n*cos(angle) + m*sin(angle)
        dky = -n*sin(angle) + m*cos(angle)
        k_x = beta_x - 2*np.pi*(np.arange(-N_x, N_x+1)+dkx)/(k0*a_x);
        k_y = beta_y - 2*np.pi*(np.arange(-N_y, N_y+1)+dky)/(k0*a_y);

        kx, ky = np.meshgrid(k_x, k_y); #this is the N_p x N_q grid
        # final matrix should be sparse...since it is diagonal at most
        #order in flatten actually doesn't matter in the end
    elif layer==2:
        k_x = -2*np.pi*np.arange(-N_x, N_x+1)/(k0*a_x);
        k_y = -2*np.pi*np.arange(-N_y, N_y+1)/(k0*a_y);
        kx, ky = np.meshgrid(k_x, k_y); 
        kx, ky = kx*cos(angle) + ky*sin(angle), ky*cos(angle) - kx*sin(angle)
        kx += -n*2*np.pi/(k0*a_x) + beta_x
        ky += -m*2*np.pi/(k0*a_y) + beta_y
    else:
        print("error! illegal layer.")

    Kx = np.diag(kx.flatten(order = 'C')); #default is C or column major
    Ky = np.diag(ky.flatten(order = 'C'))
    return Kx, Ky

def K_expanded(beta_x, beta_y, k0, a_x, a_y, N, M, angle, e_r,e_t, m_r=1,m_t=1):
    ### slightly wrong if beta_x,beta_y are not zero: they should be rotated when considering layer2
    N_p=N
    N_q=M
    k_x = beta_x - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
    k_y = beta_y - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);
    dk_x = - 2*np.pi*np.arange(-N_p, N_p+1)/(k0*a_x);
    dk_y = - 2*np.pi*np.arange(-N_q, N_q+1)/(k0*a_y);
#     dk_x, dk_y = dk_x*cos(angle)+dk_y*sin(angle), -dk_x*sin(angle)+dk_y*cos(angle)

    kx, ky = np.meshgrid(k_x, k_y); #this is the N_p x N_q grid
    kx, ky = kx.flatten(order = 'C'),ky.flatten(order = 'C')
    dkx, dky = np.meshgrid(dk_x, dk_y);  # this is the N_p x N_q grid
    dkx, dky = dkx*cos(angle)+dky*sin(angle), -dkx*sin(angle)+dky*cos(angle)
    dkx, dky = dkx.flatten(order='C'), dky.flatten(order='C')

    NM = (2*N+1) * (2*M+1)
    kx = np.tile(dkx,NM) + np.repeat(kx,NM)
    ky = np.tile(dky,NM) + np.repeat(ky,NM)
    # final matrix should be sparse...since it is diagonal at most
    Kx = np.diag(kx); ## should be of shape [NM*NM, NM*NM]
    Ky = np.diag(ky);
    N = len(Kx);
    I = np.identity(N);
    arg = (m_r*e_r*I-Kx**2-Ky**2); #arg is +kz^2
    arg = arg.astype('complex');
    Kzr = np.conj(np.sqrt(arg));
    arg2 = (m_t*e_t*I-Kx**2-Ky**2); #arg is +kz^2
    arg2 = arg2.astype('complex');
    Kzt = np.conj(np.sqrt(arg2));
    return Kx,Ky,Kzr,Kzt



class rcwa:
    def __init__(self, epsr_list, thickness_list, orientation_list,
                 mu_list=None, twist=0, gap_layer_indices=[], N=1, M=1,
                 a=1.0, ax=None, ay=None, e_r=1, e_t=1, m_r=1, m_t=1, verbose=1):
        '''
        orientation_list: 1 or 2, for untwisted coor and twisted coor
        N: G_max in x direction, G_x \in [-N,N]
        M: G_max in y direction, G_y \in [-M,M]
        should use a_x, a_y seprarately later
        '''
        self.N, self.M, self.NM = N, M, (2*N+1)*(2*M+1)
        self.ER = [convmat2D(i, N, M) if i is not None else None for i in epsr_list]
        if not mu_list:
            self.UR = [np.eye(self.NM)] * len(self.ER)
        else:
            self.UR = [convmat2D(i, N, M) if i is not None else None for i in mu_list]
        self.layer_thicknesses = thickness_list
        self.orientations = orientation_list
        self.twist = twist
        self.gap_layer_indices = gap_layer_indices
        self.a = a ### the unit length to normalize frequency with
        if ax is not None:
            self.ax = ax
        else:
            self.ax = a
        if ay is not None:
            self.ay = ay
        else:
            self.ay = a
        self.e_r, self.e_t, self.m_r, self.m_t = e_r, e_t, m_r, m_t
        self.verbose = verbose
        ### some default initializations:
        self.Sg = None ### the total scattering matrix, depending on freq and k
        ### container for intermediate variables needed for internal field reconstruction
        self.internal_Smats = [] ### will be storing Smats for internal field reconstruction
        self.internal_lambdas = [] ### will be storing propagation coefficients for internal field reconstruction
        self.internal_Ws = [] ### will be storing eigenmodes for internal field construction
        self.internal_Vs = [] ### will be storing eigenmodes for internal field construction
    
    def set_freq_k(self, freq, theta_phi=None, kxy_inc = None):
        '''
        if theta_phi is provided, will calculate k_inc accordingly;
        if k_inc is provided, will 
        '''
        if self.verbose:
            print('setting freq k...')
        self.Sg = None ### reset scattering matrix; should recalculate after setting freq and k
        ### get incident freq
        self.k0 = 2*np.pi * freq / self.a
        ### get incident k
        n_i =  np.sqrt(self.e_r*self.m_r)
        if theta_phi is not None: ### usually this is the case
            theta, phi = theta_phi
            self.kx_inc = n_i * np.sin(theta) * np.cos(phi)
            self.ky_inc = n_i * np.sin(theta) * np.sin(phi)
            self.kz_inc = np.sqrt(n_i**2 - self.kx_inc ** 2 - self.ky_inc ** 2)
            self.theta, self.phi = theta, phi
        else:
            self.kx_inc, self.ky_inc = kxy_inc
            self.kz_inc = np.sqrt(n_i**2 - self.kx_inc ** 2 - self.ky_inc ** 2)
            self.theta, self.phi = np.arccos(self.kz_inc/n_i), np.arctan2(self.ky_inc, self.kx_inc)
        ### if twisted, will need to expand the plane wave bases
        ### would be faster if we store some highly reused matrices
        if self.twist!=0:
            ### run the homogeneous calculations once for all, for time efficiency
            self.Kxs, self.Kys, self.Kzgs = [{1:{},2:{}} for _ in range(3)]
            self.Wgs, self.Vgs = None, {1:{},2:{}} ### Wgs are all identity matrices
            for m in range(-self.M,self.M+1):
                for n in range(-self.N, self.N+1):
                    for orientation in [1,2]:
                        Kx, Ky = K_matrix_given_mn(self.kx_inc, self.ky_inc, self.k0, self.ax, self.ay, self.N, self.M, n, m, angle=self.twist, layer=orientation)
                        Wg, Vg, Kzg = hl.homogeneous_module(Kx, Ky, k0=self.k0) ### if the dummy gap layer is not vacuum, set eps and mu here
                        self.Kxs[orientation][m,n] = Kx
                        self.Kys[orientation][m,n] = Ky
                        self.Kzgs[orientation][m,n] = Kzg
                        ### Wgs are just identity matrices
                        self.Vgs[orientation][m,n] = Vg
            self.Kx, self.Ky, self.kzr, self.kzt = K_expanded(self.kx_inc, self.ky_inc, self.k0, self.ax, self.ay, self.N, self.M, self.twist, self.e_r, self.e_t)
        else:
            ### no need to store anything for untwisted case
            ### but to be fair let's store Kx, Ky as well
            self.Kx, self.Ky = km.K_matrix_cubic_2D(self.kx_inc, self.ky_inc, self.k0, self.ax, self.ay, self.N, self.M); #Kx and Ky are diagonal matrices


    def solve_Smat_4D(self):
        if self.verbose:
            print('solving Smat 4D...')
        Wg = np.eye(self.NM*2)
        ### Sr: scattering matrix for reflection region
        Sr_dicts = []
        Wrs = []
        for m in range(-self.M,self.M+1):
            for n in range(-self.N, self.N+1):
                ## =============== K Matrices for gap medium =========================
                ## get Kx, Ky for different m_other, n_other
                Kx, Ky, Kzg = self.Kxs[1][m,n], self.Kys[1][m,n], self.Kzgs[1][m,n]
                Vg = self.Vgs[1][m,n]
                if self.e_r == 1 and self.m_r == 1:
                    Wr, Vr = Wg, Vg
                else:
                    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r=self.e_r, m_r=self.m_r, k0=self.k0)
                    Wrs.append(Wr)
                Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr)
                S_ref, Sr_dict = sm.S_R(Ar, Br);
                Sr_dicts.append(Sr_dict)
        Sr_dict_expanded = expand_S_dict(Sr_dicts,self.NM,layer=1)
        if len(Wrs)>0:
            self.Wr_expanded = expand_W_matrix(Wrs,self.NM,layer=1)
        else:
            self.Wr_expanded = None ### would be just identity
        
        ### St: scattering matrix for transmission region
        St_dicts = []
        Wts = []
        for m in range(-self.M,self.M+1):
            for n in range(-self.N,self.N+1):
                Kx, Ky, Kzg = self.Kxs[2][m,n], self.Kys[2][m,n], self.Kzgs[2][m,n]
                Vg = self.Vgs[2][m,n]
                if self.e_t == 1 and self.m_t==1:
                    Wt, Vt = Wg, Vg
                else:
                    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, e_r=self.e_t, m_r=self.m_t, k0=self.k0)
                    Wts.append(Wt)
                At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)
                S_trans, St_dict = sm.S_T(At, Bt)
                St_dicts.append(St_dict)
        # print('time_finishStNM',time_finishStNM - time_finishSl_redheffer)
        St_dict_expanded = expand_S_dict(St_dicts,self.NM,layer=2)
        if len(Wts)>0:
            self.Wt_expanded = expand_W_matrix(Wts,self.NM,layer=2)
        else:
            self.Wt_expanded = None ### would be just identity

        ### Sg: grand scattering matrix, combining each layer
        Sg=Sr_dict_expanded
        ### Sl: scattering matrix for each slab layer
        for i, e_conv in enumerate(self.ER):
            S_dicts = []
            if i in self.gap_layer_indices: ### gap layer can be calculated faster
                for m in range(-self.M,self.M+1):
                    for n in range(-self.N, self.N+1):
                        Kx, Ky, Kzg = self.Kxs[self.orientations[i]][m,n], self.Kys[self.orientations[i]][m,n], self.Kzgs[self.orientations[i]][m,n]
                        Vg = self.Vgs[self.orientations[i]][m,n]
                        W_i, V_i, Kzi = Wg, Vg, Kzg
                        ### if the gap layer is not vacuum, set the epsr and mu here
                        # W_i, V_i, Kzi = hl.homogeneous_module(Kx, Ky, e_h, k0=k0)
                        lambda_matrix = np.block([[1j * Kzi,np.zeros_like(Kzi)],[np.zeros_like(Kzi),1j * Kzi]])
                        Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg);
                        S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
                        S_dicts.append(Sl_dict)
            else:
                mu_conv = self.UR[i]
                for m in range(-self.M,self.M+1):
                    for n in range(-self.N,self.N+1):
                        Kx, Ky, Kzg = self.Kxs[self.orientations[i]][m,n], self.Kys[self.orientations[i]][m,n], self.Kzgs[self.orientations[i]][m,n]
                        Vg = self.Vgs[self.orientations[i]][m,n]
                        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
                        Gamma_squared = P@Q
                        W_i, lambda_matrix = em.eigen_W(Gamma_squared)
                        V_i = em.eigen_V(Q, W_i, lambda_matrix)
                        Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg)
                        S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
                        S_dicts.append(Sl_dict)
                # print('time_finishSlNM',time_finishSlNM - time_finishSr_expanded)
            S_dict_expanded=expand_S_dict(S_dicts,self.NM,layer=self.orientations[i])
            _, Sg = RedhefferStar(Sg,S_dict_expanded)
            # self.Stmp = S_dict_expanded ### for debugging
            
        _, Sg = RedhefferStar(Sg, St_dict_expanded)

        self.Sg = Sg

    def solve_Smat_4D_store_intermediate(self):
        '''
        save the Scattering matrices up to ith layer for i in range(num_layers)
        will be useful for internal field construction
        '''
        if self.verbose:
            print('solving Smat 4D...')
        Wg = np.eye(self.NM*2)
        ### Sr: scattering matrix for reflection region
        Sr_dicts = []
        Wrs = []
        for m in range(-self.M,self.M+1):
            for n in range(-self.N, self.N+1):
                ## =============== K Matrices for gap medium =========================
                ## get Kx, Ky for different m_other, n_other
                Kx, Ky, Kzg = self.Kxs[1][m,n], self.Kys[1][m,n], self.Kzgs[1][m,n]
                Vg = self.Vgs[1][m,n]
                if self.e_r == 1 and self.m_r == 1:
                    Wr, Vr = Wg, Vg
                else:
                    Wr, Vr, kzr = hl.homogeneous_module(Kx, Ky, e_r=self.e_r, m_r=self.m_r, k0=self.k0)
                    Wrs.append(Wr)
                Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr)
                S_ref, Sr_dict = sm.S_R(Ar, Br);
                Sr_dicts.append(Sr_dict)
        Sr_dict_expanded = expand_S_dict(Sr_dicts,self.NM,layer=1)
        if len(Wrs)>0:
            self.Wr_expanded = expand_W_matrix(Wrs,self.NM,layer=1)
        else:
            self.Wr_expanded = None ### would be just identity
        
        ### St: scattering matrix for transmission region
        St_dicts = []
        Wts = []
        for m in range(-self.M,self.M+1):
            for n in range(-self.N,self.N+1):
                Kx, Ky, Kzg = self.Kxs[2][m,n], self.Kys[2][m,n], self.Kzgs[2][m,n]
                Vg = self.Vgs[2][m,n]
                if self.e_t == 1 and self.m_t==1:
                    Wt, Vt = Wg, Vg
                else:
                    Wt, Vt, kz_trans = hl.homogeneous_module(Kx, Ky, e_r=self.e_t, m_r=self.m_t, k0=self.k0)
                    Wts.append(Wt)
                At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)
                S_trans, St_dict = sm.S_T(At, Bt)
                St_dicts.append(St_dict)
        # print('time_finishStNM',time_finishStNM - time_finishSl_redheffer)
        St_dict_expanded = expand_S_dict(St_dicts,self.NM,layer=2)
        if len(Wts)>0:
            self.Wt_expanded = expand_W_matrix(Wts,self.NM,layer=2)
        else:
            self.Wt_expanded = None ### would be just identity

        ### will be used for internal field reconstruction
        self.internal_Wg, self.internal_Vg, Kzg = hl.homogeneous_module(self.Kx, self.Ky, k0=self.k0) 
        
        ### Sg: grand scattering matrix, combining each layer
        Sg=Sr_dict_expanded
        ### Sl: scattering matrix for each slab layer
        for i, e_conv in enumerate(self.ER):
            self.internal_Smats.append(Sg) ### saving Smat up to (before) each layer
            S_dicts = []
            ### following will be concatenated for field reconstruction
            lambda_matrices = []
            Wis = []
            Vis = []
            if i in self.gap_layer_indices: ### gap layer can be calculated faster
                for m in range(-self.M,self.M+1):
                    for n in range(-self.N, self.N+1):
                        Kx, Ky, Kzg = self.Kxs[self.orientations[i]][m,n], self.Kys[self.orientations[i]][m,n], self.Kzgs[self.orientations[i]][m,n]
                        Vg = self.Vgs[self.orientations[i]][m,n]
                        W_i, V_i, Kzi = Wg, Vg, Kzg
                        ### if the gap layer is not vacuum, set the epsr and mu here
                        # W_i, V_i, Kzi = hl.homogeneous_module(Kx, Ky, e_h, k0=k0)
                        lambda_matrix = np.block([[1j * Kzi,np.zeros_like(Kzi)],[np.zeros_like(Kzi),1j * Kzi]])
                        Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg);
                        S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
                        S_dicts.append(Sl_dict)
                        lambda_matrices.append(np.diag(lambda_matrix))
                        Wis.append(W_i)
                        Vis.append(V_i)
            else:
                mu_conv = self.UR[i]
                for m in range(-self.M,self.M+1):
                    for n in range(-self.N,self.N+1):
                        Kx, Ky, Kzg = self.Kxs[self.orientations[i]][m,n], self.Kys[self.orientations[i]][m,n], self.Kzgs[self.orientations[i]][m,n]
                        Vg = self.Vgs[self.orientations[i]][m,n]
                        P, Q, kzl = pq.P_Q_kz(Kx, Ky, e_conv, mu_conv)
                        Gamma_squared = P@Q
                        W_i, lambda_matrix = em.eigen_W(Gamma_squared)
                        V_i = em.eigen_V(Q, W_i, lambda_matrix)
                        Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg)
                        S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
                        S_dicts.append(Sl_dict)
                        lambda_matrices.append(np.diag(lambda_matrix))
                        Wis.append(W_i)
                        Vis.append(V_i)
                # print('time_finishSlNM',time_finishSlNM - time_finishSr_expanded)
            S_dict_expanded=expand_S_dict(S_dicts,self.NM,layer=self.orientations[i])
            _, Sg = RedhefferStar(Sg,S_dict_expanded)
            # self.Stmp = S_dict_expanded ### for debugging
            self.internal_lambdas.append(expand_lambda(lambda_matrices, self.NM, layer=self.orientations[i])) ### saving lambda matrix for each layer
            self.internal_Ws.append(expand_V_matrix(Wis, self.NM, layer=self.orientations[i])) ### saving W matrix for each layer
            self.internal_Vs.append(expand_V_matrix(Vis, self.NM, layer=self.orientations[i])) ### saving V matrix for each layer
        _, Sg = RedhefferStar(Sg, St_dict_expanded)

        self.Sg = Sg
    
    def solve_Smat_2D(self):
        if self.verbose:
            print('solving Smat 2D...')
        Wg, Vg, Kzg = hl.homogeneous_module(self.Kx, self.Ky, k0=self.k0) ### if the dummy gap layer is not vacuum, set eps and mu here
        ### Sr
        if self.e_r == 1 and self.m_r == 1:
            Wr, Vr = Wg, Vg
            self.Wr_expanded = None ### no need to store
            self.kzr = Kzg
        else:
            Wr, Vr, kzr = hl.homogeneous_module(self.Kx, self.Ky, self.e_r)
            self.Wr_expanded = Wr
            self.kzr = kzr
        Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr)
        S_ref, Sr_dict = sm.S_R(Ar, Br); #scatter matrix for the reflection region
        ### St
        if self.e_t == 1 and self.m_t == 1:
            Wt, Vt = Wg, Vg
            self.Wt_expanded = None ### no need to store
            self.kzt = Kzg
        else:
            Wt, Vt, kzt = hl.homogeneous_module(self.Kx, self.Ky, self.e_t)
            self.Wt_expanded = Wt
            self.kzr = kzt
        At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)
        S_trans, St_dict = sm.S_T(At, Bt)
        ### Sg: grand scattering matrix from reflection region to transmission region
        Sg = Sr_dict
        ### Sl: scattering matrix for each slab layer
        for i in range(len(self.ER)):
            ### not giving special treatment to gap layers, since it is fast enough anyways
            ### ok we should; next step is to ensure branch cut choice in P_Q_kz
            if i in self.gap_layer_indices: ### gap layer can be calculated faster
                W_i, V_i, Kzi = Wg, Vg, Kzg
                lambda_matrix = np.block([[1j * Kzi,np.zeros_like(Kzi)],[np.zeros_like(Kzi),1j * Kzi]])
                Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg);
                S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
            else:
                #ith layer material parameters
                e_conv = self.ER[i]
                mu_conv = self.UR[i]
                #longitudinal k_vector
                P, Q, kzl = pq.P_Q_kz(self.Kx, self.Ky, e_conv, mu_conv)
                self.P, self.Q = P, Q
        #             print("condition",cond(P),cond(Q))
                # kz_storage.append(kzl)
                Gamma_squared = P@Q;
                ## E-field modes that can propagate in the medium, these are well-conditioned
                W_i, lambda_matrix = em.eigen_W(Gamma_squared);
                V_i = em.eigen_V(Q, W_i, lambda_matrix);
                #now defIne A and B, slightly worse conditoined than W and V
                A,B = sm.A_B_matrices(W_i, Wg, V_i, Vg); #ORDER HERE MATTERS A LOT because W_i is not diagonal
        #             print("condition",cond(A),cond(B))
                #calculate scattering matrix
                Li = self.layer_thicknesses[i]
                S_layer, Sl_dict = sm.S_layer(A, B, Li, self.k0, lambda_matrix)
            _, Sg = RedhefferStar(Sg, Sl_dict)
            # if i==1:
            #     S_layer.dump('test_S.pkl')
        _, Sg = RedhefferStar(Sg, St_dict)
        self.Sg = Sg

    def solve_Smat_2D_store_intermediate(self):
        '''
        save_internal_Smat: whether to save the Scattering matrices up to ith layer for i in range(num_layers); will be useful for internal field construction
        '''
        if self.verbose:
            print('solving Smat 2D...')
        Wg, Vg, Kzg = hl.homogeneous_module(self.Kx, self.Ky, k0=self.k0) ### if the dummy gap layer is not vacuum, set eps and mu here
        self.internal_Wg = Wg ### saving Wg matrix for field construction
        self.internal_Vg = Vg ### saving Vg matrix for field construction
        ### Sr
        if self.e_r == 1 and self.m_r == 1:
            Wr, Vr = Wg, Vg
            self.Wr_expanded = None ### no need to store
            self.kzr = Kzg
        else:
            Wr, Vr, kzr = hl.homogeneous_module(self.Kx, self.Ky, self.e_r)
            self.Wr_expanded = Wr
            self.kzr = kzr
        Ar, Br = sm.A_B_matrices(Wg, Wr, Vg, Vr)
        S_ref, Sr_dict = sm.S_R(Ar, Br); #scatter matrix for the reflection region
        ### St
        if self.e_t == 1 and self.m_t == 1:
            Wt, Vt = Wg, Vg
            self.Wt_expanded = None ### no need to store
            self.kzt = Kzg
        else:
            Wt, Vt, kzt = hl.homogeneous_module(self.Kx, self.Ky, self.e_t)
            self.Wt_expanded = Wt
            self.kzr = kzt
        At, Bt = sm.A_B_matrices(Wg, Wt, Vg, Vt)
        S_trans, St_dict = sm.S_T(At, Bt)
        ### Sg: grand scattering matrix from reflection region to transmission region
        Sg = Sr_dict
        ### Sl: scattering matrix for each slab layer
        for i in range(len(self.ER)):
            self.internal_Smats.append(Sg) ### saving Smat before each layer
            if i in self.gap_layer_indices: ### gap layer can be calculated faster
                W_i, V_i, Kzi = Wg, Vg, Kzg
                lambda_matrix = np.block([[1j * Kzi,np.zeros_like(Kzi)],[np.zeros_like(Kzi),1j * Kzi]])
                Al,Bl = sm.A_B_matrices(W_i, Wg, V_i, Vg);
                S_layer, Sl_dict = sm.S_layer(Al, Bl, self.layer_thicknesses[i], self.k0, lambda_matrix)
            else:
                ### not giving special treatment to gap layers, since it is fast enough anyways
                #ith layer material parameters
                e_conv = self.ER[i]
                mu_conv = self.UR[i]
                #longitudinal k_vector
                P, Q, kzl = pq.P_Q_kz(self.Kx, self.Ky, e_conv, mu_conv)
                self.P, self.Q = P, Q
        #             print("condition",cond(P),cond(Q))
                # kz_storage.append(kzl)
                Gamma_squared = P@Q;
                ## E-field modes that can propagate in the medium, these are well-conditioned
                W_i, lambda_matrix = em.eigen_W(Gamma_squared);
                V_i = em.eigen_V(Q, W_i, lambda_matrix);
                #now defIne A and B, slightly worse conditoined than W and V
                A,B = sm.A_B_matrices(W_i, Wg, V_i, Vg); #ORDER HERE MATTERS A LOT because W_i is not diagonal
        #             print("condition",cond(A),cond(B))
                #calculate scattering matrix
                Li = self.layer_thicknesses[i]
                S_layer, Sl_dict = sm.S_layer(A, B, Li, self.k0, lambda_matrix)
            _, Sg = RedhefferStar(Sg, Sl_dict)
            self.internal_lambdas.append(np.diag(lambda_matrix)) ### saving lambda matrix for each layer
            self.internal_Ws.append(W_i) ### saving W matrix for each layer
            self.internal_Vs.append(V_i) ### saving V matrix for each layer
        _, Sg = RedhefferStar(Sg, St_dict)
        self.Sg = Sg
        

    def get_RT(self, pte, ptm, storing_all_orders = True, storing_intermediate_Smats = False, cinc_overwrite = None):
        if self.twist != 0:
            if self.Sg is None:
                if storing_intermediate_Smats:
                    self.solve_Smat_4D_store_intermediate()
                else:
                    self.solve_Smat_4D()
            whether_to_expand = True
            num_diffractions = self.NM*self.NM
        else:
            if self.Sg is None:
                if storing_intermediate_Smats:
                    self.solve_Smat_2D_store_intermediate()
                else:
                    self.solve_Smat_2D()
            whether_to_expand = False
            num_diffractions = self.NM
            
        normal_vector = np.array([0, 0, -1]) #positive z points down;
        ate_vector = np.array([0, 1, 0]); # TE E-field in y direction
        n_i =  np.sqrt(self.e_r*self.m_r) # incident beam is in reflection medium, assuming mu_r=1
        NM = self.NM # shorter name
        # for pte, ptm in zip(ptes, ptms):
        if cinc_overwrite is None: ### assuming just incidence in 0th-order diffraction channel
            K_inc_vector = np.array([self.kx_inc, self.ky_inc, self.kz_inc])
            E_inc, cinc, Polarization = ic.initial_conditions(K_inc_vector, self.theta,  normal_vector, pte, ptm, self.N, self.M)
            if whether_to_expand: ### trivially expand for twisted case
                if self.verbose:
                    print('expanding cinc')
                cinc = expand_cinc(cinc,self.NM)
        else: ### overwrite cinc, of shape [2*NM*NM]; Einc and Polarization are not adjusted since they are not used anyways
            cinc = cinc_overwrite
            if whether_to_expand:
                cinc = cinc.reshape(2*self.NM*self.NM,1)
            else:
                cinc = cinc.reshape(2*self.NM,1)
        
        self.cinc = np.array(cinc) ### storing for field connstruction reference

        if self.Wr_expanded:
            Wr_expanded = self.Wr_expanded
            Wr_expanded_inv = np.linalg.inv(Wr_expanded)
        else:
            Wr_expanded = np.eye(2*num_diffractions)
            Wr_expanded_inv = Wr_expanded
        if self.Wt_expanded:
            Wt_expanded=self.Wt_expanded
        else:
            Wt_expanded = np.eye(2*num_diffractions)
        cinc = Wr_expanded_inv@cinc
        reflected = Wr_expanded@self.Sg['S11']@cinc
        transmitted = Wt_expanded@self.Sg['S21']@cinc
        rx = reflected[0:num_diffractions, :]; # rx is the Ex component.
        ry = reflected[num_diffractions:, :];  #
        tx = transmitted[0:num_diffractions,:];
        ty = transmitted[num_diffractions:, :];
        # longitudinal components; should be 0
        rz = np.linalg.inv(self.kzr) @ (self.Kx @ rx + self.Ky @ ry)
        tz = np.linalg.inv(self.kzt) @ (self.Kx @ tx + self.Ky @ ty)
        r_sq = np.square(np.abs(rx)) +  np.square(np.abs(ry))+ np.square(np.abs(rz))
        t_sq = np.square(np.abs(tx)) +  np.square(np.abs(ty))+ np.square(np.abs(tz))
        ### diffraction efficiency in each order
        R = np.real(self.kzr) @ r_sq / np.real(self.kz_inc)
        T = np.real(self.kzt) @ t_sq / (np.real(self.kz_inc))
        ### total diffraction efficiency
        Rtot = np.sum(R)
        Ttot = np.sum(T)
        if self.verbose:
            print('got R,T,R+T',Rtot,Ttot,Rtot+Ttot)
        if storing_all_orders:
            self.reflected, self.transmitted = np.array(reflected), np.array(transmitted)
            self.rz, self.tz = np.array(rz), np.array(tz)
            # self.reflected, self.transmitted = np.vstack([reflected,rz]), np.vstack([transmitted,tz])
        ### returning two tuples: total ref/trans, 0th ref/trans Exyz
        return (Rtot, Ttot),\
             (np.array([rx[num_diffractions//2,0],ry[num_diffractions//2,0],rz[num_diffractions//2,0]]),
             np.array([tx[num_diffractions//2,0],ty[num_diffractions//2,0],tz[num_diffractions//2,0]]))

    def get_internal_field(self, which_layers = None, offsets=None):
        '''
        run after self.get_RT (where incidence is set)
        which_layers: list of 0-indexed layer indices, for which we want field to be constructed
        offsets: corresponding offsets in each layer, where the field should be constructed
        '''
        if which_layers is None:
            which_layers = list(range(len(self.layer_thicknesses)))
            offsets = [0]*len(which_layers)

        fields = [] ### to record the Fourier coefficients in which_layers at offsets
        econv_invs = {} ### to store the inverse of e_conv, to avoid repetive calculations, for Ez and Hz calculations
        mconv_invs = {} ### to store the inverse of mu_conv, to avoid repetive calculations, for Ez and Hz calculations
        for l, z in zip(which_layers, offsets):
            if self.verbose:
                print(f'getting internal field for layer {l} offset {z}')
            Smat = self.internal_Smats[l]
            ### cl1_minus, cl1_plus are mode coefficients immediately before the layer in gap medium
            cl1_minus = np.linalg.solve(Smat['S12'], self.reflected - Smat['S11'].dot(self.cinc)) ### backward wave before l-th layer
            cl1_plus = Smat['S21'].dot(self.cinc) + Smat['S22'].dot(cl1_minus)
            ### mode_coeff are mode coefficients in the layer at offset=0
            mode_to_fourier_l = np.block([[self.internal_Ws[l], self.internal_Ws[l]],[-self.internal_Vs[l], self.internal_Vs[l]]])
            mode_to_fourier_g = np.block([[self.internal_Wg, self.internal_Wg],[-self.internal_Vg, self.internal_Vg]])
            mode_coeff = np.linalg.solve(mode_to_fourier_l, mode_to_fourier_g.dot(np.concatenate([cl1_plus, cl1_minus])))
            phase = np.diag(np.exp(np.concatenate([-self.internal_lambdas[l]*self.k0*z, self.internal_lambdas[l]*self.k0*z])))
            field_xy = mode_to_fourier_l.dot(phase.dot(mode_coeff)) ### len([E,H]) * len([x,y]) * num_Gs
            self.field_xy = field_xy ### ??? debugging
            ### need to invert e_conv and mu_conv to get Ez and Hz
            if self.twist!=0:
                sx, sy, ux, uy = field_xy.reshape(4,self.NM**2)
            else:
                sx, sy, ux, uy = np.array(field_xy).reshape(4,self.NM)

            if l in econv_invs and l in mconv_invs:
                econv_inv = econv_invs[l]
                mconv_inv = mconv_invs[l]
            else:
                ### no need to calculate if is gap layer
                if self.ER[l] is None:
                    econv_inv = np.eye(self.NM)
                else:
                    econv_inv = np.linalg.inv(self.ER[l])
                if self.UR[l] is None:
                    mconv_inv = np.eye(self.NM)
                else:
                    mconv_inv = np.linalg.inv(self.UR[l])
                ### stitch into expanded k space if twisted
                if self.twist!=0:
                    if self.orientations[l]==1:
                        econv_inv = np.kron(econv_inv,np.eye(self.NM))
                        mconv_inv = np.kron(mconv_inv,np.eye(self.NM))
                    elif self.orientations[l]==2:
                        econv_inv = np.kron(np.eye(self.NM), econv_inv)
                        mconv_inv = np.kron(np.eye(self.NM), mconv_inv)
                    else:
                        raise ValueError("Only support up to 2 twist orientations for now")
                econv_invs[l] = econv_inv
                mconv_invs[l] = mconv_inv
                self.econv = econv_inv

            sz = -1j * econv_inv @ (self.Kx@uy - self.Ky@ux)
            uz = -1j * mconv_inv @ (self.Kx@sy - self.Ky@sx)
            fields.append(np.concatenate([sx,sy,sz,ux,uy,uz]))
        return fields
                            

    def get_RT_field(self):
        '''
        should run after get RT
        '''
        # if self.twist!=0:
        #     num_diffractions = self.NM**2
        # else:
        #     num_diffractions = self.NM
        W, V = self.internal_Wg, self.internal_Vg
        cinc = np.linalg.inv(W)@self.cinc
        mode_to_fourier = np.vstack([W,V])
        reflected = mode_to_fourier@self.Sg['S11']@cinc
        transmitted = mode_to_fourier@self.Sg['S21']@cinc
        erx,ery,hrx,hry = reflected.reshape(4,-1)
        etx,ety,htx,hty = transmitted.reshape(4,-1)

        erz = np.linalg.inv(self.kzr) @ (self.Kx @ erx + self.Ky @ ery)
        hrz = np.linalg.inv(self.kzr) @ (self.Kx @ hrx + self.Ky @ hry)
        etz = np.linalg.inv(self.kzt) @ (self.Kx @ etx + self.Ky @ ety)
        htz = np.linalg.inv(self.kzt) @ (self.Kx @ htx + self.Ky @ hty)
        return np.concatenate([erx,ery,erz,hrx,hry,hrz]),np.concatenate([etx,ety,etz,htx,hty,htz])

    def get_Stress_tensor(self, which_layers=None, offsets=None):
        ### !!! now only works in air layers
        ### TODO: make it compatible with patterned layer (just need to calc D and B fields)
        ### TODO: deal with the case of external field and tensor too 
        fields = self.get_internal_field(which_layers, offsets)
        tensors = []
        for field in fields:
            sx,sy,sz,ux,uy,uz = field.reshape(6,-1)
            Tx = np.real(sx*np.conj(sz) + ux*np.conj(uz))
            Ty = np.real(sy*np.conj(sz) + uy*np.conj(uz))
            Tz = 0.5*np.real(sz*np.conj(sz) + uz*np.conj(uz)
                         - np.abs(sx)**2 - np.abs(sy)**2 
                         - np.abs(ux)**2 - np.abs(uy)**2)
            # tensors.append([Tx,Ty,Tz])        
            tensors.append([np.sum(Tx),np.sum(Ty),np.sum(Tz)])        
        return tensors

