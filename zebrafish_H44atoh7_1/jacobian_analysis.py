import numpy as np
import matplotlib.pyplot as plt
import dynamo as dyn
import pandas as pd
import os
import seaborn as sns
import anndata as Anndata

from sklearn.neighbors import NearestNeighbors

# type set
from typing import Dict, List, Any, Tuple
from pandas import DataFrame

dyn.dynamo_logger.main_silence()

###
# # ----------------------------------------------------------------------------------------------
### define some useful transform and calculation functions class
class UseFuns:
    def __init__(self) -> None:
        pass

        # trajectory related
    def pca_to_expr(self,X,PCs, mean=0, func=None):
        # reverse project from PCA back to raw expression space
        if PCs.shape[1] == X.shape[1]:
            exprs = X @ PCs.T + mean
            if func is not None:
                exprs = func(exprs)
        else:
            raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[1], X.shape[1]))
        return exprs

    def expr_to_pca(self,expr, PCs, mean=0, func=None):
        # project from raw expression space to PCA
        if PCs.shape[0] == expr.shape[1]:
            X = (expr - mean) @ PCs
            if func is not None:
                X = func(X)
        else:
            raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[0], expr.shape[1]))
        return X
    
    def transJacPca_to_Ori(self,J_pca,Q):
        """
        Jacobian matrix is inversely transformed back to high dimension. Note that this should also be possible for reduced UMAP space and will be supported shortly.
        """
        return Q @ J_pca @ Q.T

    def get2dmatDet(self,J_mat:np.array):
        """
        calculate determination (mutiplication of eigenvalues) for a given 2-by-2 matrix `J_mat`
        """
        return J_mat[0,0]*J_mat[1,1]-J_mat[0,1]*J_mat[1,0]
    
    def getTrace(self,J_mat:np.array):
        """
        calculate trace (sum of eigenvalues)  for the matrix `J_mat`
        """
        return J_mat.trace()
    
    def getListsIntersect(self,list1:List,list2:List) -> List:
        """
        get 2 lists instersection.
        """
        return list(set(list1).intersection(set(list2)))
    
    def nearest_neighbors(self,coord:np.array, coords:np.array,n_neighbors:int=5,radius:float=1.0,algorithm:str='auto') -> np.array:
        """
        Unsupervised learner for implementing neighbor searches.

        Parameters
        ----------
        coord: {`array-like`, sparse matrix}, shape (n_queries, n_features), or (n_queries, n_indexed)
            The query point or points. If not provided, neighbors of each indexed point are returned. In this case, the query point is not considered its own neighbor.
        coords:{array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
            Training data.
        n_neighbors: `int`, default=5
            Number of neighbors to use by default for kneighbors queries.
        radius: `float`, default=1.0
            Range of parameter space to use by default for radius_neighbors queries.
        algorithm: `str`, {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
            Algorithm used to compute the nearest neighbors:

            `ball_tree` will use BallTree

            `kd_tree` will use KDTree

            `brute` will use a brute-force search.

            `auto` will attempt to decide the most appropriate algorithm based on the values passed to fit method.

            Note: fitting on sparse input will override the setting of this parameter, using `brute force`.
        
        Returns
        ----------
            neighs: ndarray of shape (n_queries, n_neighbors)
                Indices of the nearest points in the population matrix.
        """
        nbrs = NearestNeighbors(n_neighbors=n_neighbors,radius=radius,algorithm=algorithm).fit(coords)
        neighs = nbrs.kneighbors(np.atleast_2d(coord),return_distance=False)

        return neighs
    
    # file related
    def mkdir(self,path:str):
        """
        creat folder for given path
        """
        folder = os.path.exists(path)

        # check if the folder exists, if not ,create the folder.
        if not folder:
            #  `makedirs` create the path if it does not exist when creating a foler
            os.makedirs(path)
            print("---  new folder...  ---")
            print("---  OK  ---")

        else:
            print("---  There is this folder!  ---")
    


# # ---------------------------------------
### define measures class of calculating the confidence of fixed points 
class FPMeasures(UseFuns):
    def __init__(self,adata:Anndata) -> None:
        self.adata = adata
        pass
    def getConf_neighVel(self,neighb_nums:int=30,epsilon:float=1e-1,basis='umap',normalize:bool=False) -> np.array:
        """
        calculate confidence of fixed points according to the velociteis of neighbouring cells
        Parameters
        ----------
        neighb_nums: `int` (default: `30`)
            the number of the nearest neighboring cells for a given fixed point
        epsilon: `float` (default: `1e-1`)
            the threshold to judge the velocity of the cells.
        basis: `str` (default: `umap`)
            which velocities are used to judge.
        normalize: `bool` (default: `True`)
            whether normalize the confidence of fixed points.
        Returns
        ----------
        conf_fp: `np.array`
            confidence of the fixed points.
        """
        from dynamo.tools.utils import nearest_neighbors
        coordfp = self.adata.uns['VecFld_umap']['Xss']
        neighbCells = nearest_neighbors(coordfp, self.adata.obsm["X_umap"],k=neighb_nums)
        conf_fp = []
        for i in range(len(neighbCells)):
            velMagn = np.sum(self.adata.obsm['velocity_'+basis][neighbCells[i]]**2,axis=1)
            conf_fp.append(np.sum(velMagn<epsilon)/len(velMagn))
        conf_fp = np.array(conf_fp)
        if normalize:
            conf_vel /= conf_vel.max()

        return conf_fp

    def plot_fpConfidence_distri(self,conf_list:List,label_list:List, color_list:str, bins:int=20, density:bool=True,save_or_return:str='return',save_path:str='',alpha:float=0.8):
        """
        Compute and plot a histogram. Plot the distribution of conficence of the fxied points.

        Parameters
        ----------
        conf_list: `list[list or array]`
            conficence list of the fxied points.
        label_list: `list`
            label list of the fxied points.
        color_list: `list[str]`
            color list of the histgram.
        bins: `int`
            defines the number of equal-width bins in the range.
        density: `bool`
            If True, draw and return a probability density: each bin will display the bin's raw count divided by the total number of counts and the bin width (density = counts / (sum(counts) * np.diff(bins))), so that the area under the histogram integrates to 1 (np.sum(density * np.diff(bins)) == 1).
        Returns
        ----------
        Nonthing but a plot.
        """
        plt.figure()
        for i,conf_tmp in enumerate(conf_list):
            plt.hist(conf_tmp,bins=bins,density=density,color=color_list[i],label=label_list[i],alpha=alpha)
        plt.title('Confidence of the Fixed Points', fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.xlabel('confidence',fontsize=12)
        plt.ylabel('normalized number',fontsize=12)
        plt.legend(fontsize=12)
        if save_or_return=='save':
            plt.savefig(save_path+'fpConfiDistr_hist_1.png')

        plt.show()

    # ---------------------------------------------------------------------------------------------------
    def get_VecFldfun(self,X:np.array,basis:str='pca'):
        # get vector field function in adata.uns['VecFld_'+basis]
        """
        get vector field function in adata.uns['VecFld_'+basis]
        There is other method to obtain vector filed function, i.e.,
        from dynamo.vectorfield.utils import con_K
        Xc = self.adata.uns['VecFld_'+basis]['X_ctrl']
        beta = self.adata.uns['VecFld_'+basis]['beta']
        need_utility_time_measure = False
        K = con_K(X, Xc, beta, timeit=need_utility_time_measure)
        C = self.adata.uns['VecFld_'+basis]['C']
        V = K.dot(C)
        return V

        Parameters
        ----------
            X:
                the states used to predict.
            basis:
                basis used to get vector field function 

        Returns
        -------
           the predicted velocity.
        """
        from dynamo.vectorfield.scVectorField import DifferentiableVectorField
        dvf=DifferentiableVectorField()
        dvf.from_adata(self.adata,basis=basis)
        # from dynamo.vectorfield.utils import vecfld_from_adata
        # vf_dict, func = vecfld_from_adata(self.adata, basis=basis, vf_key="VecFld")
        V_pred=dvf.func(X)

        return V_pred

    def get_Jacfun(self,basis:str='umap',method:str="analytical",):
        """
        Get the Jacobian of the vector field function.
        If method is 'analytical':
        The analytical Jacobian will be returned and it always
        take row vectors as input no matter what input_vector_convention is.

        No matter the method and input vector convention, the returned Jacobian is of the
        following format:
                df_1/dx_1   df_1/dx_2   df_1/dx_3   ...
                df_2/dx_1   df_2/dx_2   df_2/dx_3   ...
                df_3/dx_1   df_3/dx_2   df_3/dx_3   ...
                ...         ...         ...         ...

        Parameters
        ----------
            basis: `str` (default: `umap`)
                basis used to reconstruct vector field
            method: `str` (default: `analytical`)
                method to obatin the Jacobian matrix, `analytical` means `J=-2w(C^{T}KD)`.
        Returns
        ----------
            f_jac: 'function class'
                Jacobian matrix function.
        """

        from dynamo.vectorfield.scVectorField import SvcVectorField
        vector_field_class = SvcVectorField()
        vector_field_class.from_adata(self.adata, basis=basis)
        f_jac = vector_field_class.get_Jacobian(method=method)

        return f_jac

    def calculate_Jacfun(self, X:np.array, basis:str='umap'):
        """
        Another method to calculate Jacobian matrix, which is faster than `get_Jacfun()`.
        analytical Jacobian for RKHS vector field functions with Gaussian kernel.

        Arguments
        ---------
        X: :class:`~numpy.ndarray`
            Coordinates where the Jacobian is evaluated.
        basis: `str` (default: `umap`)
                basis used to reconstruct vector field

        Returns
        -------
        J: :class:`~numpy.ndarray`
            Jacobian matrices stored as d-by-d-by-n numpy arrays evaluated at x.
            d is the number of dimensions and n the number of coordinates in x.
    """
        from dynamo.vectorfield.utils import con_K
        Xc = self.adata.uns['VecFld_'+basis]['X_ctrl']
        beta = self.adata.uns['VecFld_'+basis]['beta']
        K, D = con_K(X, Xc, beta, return_d=True)
        K, D = np.atleast_2d(K,D)
        C = self.adata.uns['VecFld_'+basis]['C']  
        n, d = X.shape
        J = np.zeros((d,d,n))
        for i in range(n):       
            J[:,:,i] = (C.T * K[i]) @ D[i].T
        return -2 * beta * J

    def compute_acceleration(self,X,basis):
        """
        compute acceleration.
        """
        from dynamo.vectorfield.scVectorField import DifferentiableVectorField
        dvf=DifferentiableVectorField()
        dvf.from_adata(self.adata,basis=basis)
        acc = dvf.compute_acceleration(X=X,method='analytical')
        return acc
    
    def get_fpInfo(self,J:np.array,) -> DataFrame:
        """
        get the information of fixed points in `umap` space. \n
        J = [[J_{11},J_{12}],[J_{21},J_{22}]], \n
        Det(J) = lambda_1*lambda_2 = J_{11}*J_{22}-J_{12}J_{21}, \n
        Tr(J) = lambda_1*lambda_2 = J_{11}*J_{22}-J_{12}J_{21}, \n
        soucer: Det(J)>0, Tr(J)>0. \n
        attractor: Det(J)>0, Tr(J)<0. \n
        saddel: Det(J)<0.

        Parameters
        ----------
            J: `np.array`, 2-by-2-by-n matrix.
                Jacobian matrix of the `n` fixed points in `umap` space.

        Returns
        ----------
            df_fp: `DataFrame`
                Information of the fixed points.
        """

        df_fp = pd.DataFrame(data={ 
            'umap0': self.adata.uns['VecFld_umap']['Xss'][:,0],
            'umap1': self.adata.uns['VecFld_umap']['Xss'][:,1],
            'ftype': self.adata.uns['VecFld_umap']['ftype']
            })
        DetJ = []
        TrJ = []
        for i in range(J.shape[2]):
            Ji = J[:,:,i]
            DetJ.append(self.get2dmatDet(Ji))
            TrJ.append(self.getTrace(Ji))
        df_fp['fptype'] = ['']*len(df_fp)
        df_fp['fptype'][df_fp['ftype']==0] = 'saddle'
        df_fp['fptype'][df_fp['ftype']==1] = 'source'
        df_fp['fptype'][df_fp['ftype']==-1] = 'attractor'
        df_fp['determinant'] = DetJ
        df_fp['trace'] = TrJ
        
        return df_fp
# # ---------------------------------------
### define jacobian class to obtain Jacobian information.
class Jacobian(FPMeasures):
    def __init__(self,adata:Anndata) -> None:
        self.adata = adata[:,adata.var['use_for_pca']]
        pass
    
    def get_JacInfo(self,query_dict:dict,n_neighbors:int=5,genes_dict:dict={},basis:str='pca',row_cluster:bool=True,col_cluster:bool=True,save_res:bool=False,save_fig:bool=False,save_path:str='./',TFs_dict:dict={},rowsQuantile:float=0.5,colsQuantile:float=0.5) -> Dict:
        """
        get Jacobian matrix in one cell.

        Parameters
        ----------
            query_dict: `dict`
                dictionary of query coords, `key` is cell type, `value` is query coords array with shape of(n_queries, n_features).
            n_neighbors: `int`, default=5
                Number of neighbors to use by default for kneighbors queries.
            genes_dict: `dict`, default is `{}`.
                genes dictionary used to extract subset of Jacobian matrix. `key` is cell type, `value` is list of interested genes. Here we consider the interested genes may be different for the different cell type.
            basis: str, default is `pca`.
                the basis to obtain the jacobian matrix. At present, only support 'pca', 'umap' may be supported in the future. 
            {row,col}_cluster: `bool`, optional
                If `True`, cluster the {rows, columns} when plot the heatmap of Jacobian matrix.
            save_res: `bool`, default=False
                whether to save the clustered/non-clustered Jacobian matrix.
            save_fig: `bool`, default is `False`
                whether to save the heatmap of the clustered/non-clustered Jacobian matrix.
            save_path: `str`, default is `./`
                the path used to save the heatmap and Jacobian matrix.
            TFs_dict: `dict`, default is `{}`
                transcription factors (TFs) dictionary used to extract subset of Jacobian matrix. `key` is cell type, `value` is list of interested TFs. Here we consider the interested TFs may be different for the different cell type.
                Note: if `TFs_dict` is offered, then use the intersection beweet `genes_dict` and `TFs_dict` to  extract subset of Jacobian matrix. 
            {rowsQuantile, colsQuantile}: `float`, default is `0.5`
                for the calculated Jacobian matrix, keep values at the top quantile over requested {rows, columns}, based on the the sum of the absolute values of {rows, colums}  .

        Returns
        ----------
            Jac_clustered: `dict`
                the clustered/non-clustered Jacobian matrix. `key` is cell type, `value` is `DataFrame` of the Jacobian matrix.
        """

        PCs = self.adata.uns['PCs']
        # pca_mean = self.adata.uns['pca_mean']
        Jac_dict={}
        for cell_type in query_dict:
            self.mkdir(path=save_path+cell_type)
            adata_new = self.adata[self.adata.obs['Cell_type'].isin([cell_type])]
            neighs = self.nearest_neighbors(query_dict[cell_type],adata_new.obsm['X_umap'],n_neighbors=n_neighbors,algorithm='auto')
            for i,neigh in enumerate(neighs):
                print('For '+cell_type+'_'+str(i)+', found '+str(len(neigh))+' neighbors.')

            # intersection between genes and TFs
            if TFs_dict and len(TFs_dict[cell_type])>0:
                print('------------------------------------------------------------')
                print('calculate the intersection betweeen genes and TFs.')
                print('------------------------------------------------------------')
                genes_dict[cell_type] = self.getListsIntersect(genes_dict[cell_type],TFs_dict[cell_type])

            
            for i,neigh in enumerate(neighs):
                J_arr = self.calculate_Jacfun(adata_new.obsm['X_pca'][neigh],basis='pca')
                # f_jac = self.get_Jacfun(basis='pca')
                # J_arr1 = f_jac(adata_new.obsm['X_pca'][neigh],)
                # np.allclose(J_arr,J_arr1)
                Js_tmp = 0
                for j in range(J_arr.shape[2]):
                    Js_tmp += self.transJacPca_to_Ori(J_arr[:,:,j],PCs)
                Js_tmp /= n_neighbors
                Js_tmp = pd.DataFrame(data=Js_tmp,index=self.adata.var_names.tolist(),columns=self.adata.var_names.tolist())
                Js_tmp = Js_tmp.loc[genes_dict[cell_type],genes_dict[cell_type]]
                Js_tmp.loc['row_abs_sum'] = (Js_tmp.apply(abs)).sum(axis=0)
                Js_tmp['col_abs_sum'] = (Js_tmp.apply(abs)).sum(axis=1)
                # sort values
                # sort by columns
                # Js_tmp.sort_values(by=['row_abs_sum'],axis=1,ascending=False,inplace=True)
                # sort by index
                # Js_tmp.sort_values(by=['col_abs_sum'],axis=0,ascending=False,inplace=True)
                # extract values via given {rows, cols} quantile
                drop_colsTh = Js_tmp['col_abs_sum'].quantile(1-colsQuantile)
                # drop some rows satisfied ['col_abs_sum']<drop_colsTh
                Js_tmp = Js_tmp[Js_tmp['col_abs_sum']>=drop_colsTh]
                # drop some columns satisfied ['row_abs_sum']<drop_rowsTh
                drop_rowsTh = Js_tmp.loc['row_abs_sum'].quantile(1-rowsQuantile)
                Js_tmp = Js_tmp.loc[:,Js_tmp.loc['row_abs_sum']>=drop_rowsTh]
                Js_tmp.drop(index=['row_abs_sum'],inplace=True)
                Js_tmp.drop(columns=['col_abs_sum'],inplace=True)

                Jac_dict[cell_type+'_'+str(i)] = Js_tmp
            
        Jac_clustered = self.plot_JacInfo(Jac_dict=Jac_dict,row_cluster=row_cluster,col_cluster=col_cluster,save_fig=save_fig,save_path=save_path,save_fmt='.pdf')

        if save_res:
            self.save_JacInfo(save_path,Jac_clustered,row_cluster=row_cluster,col_cluster=col_cluster)

        return Jac_clustered


    def plot_JacInfo(self,Jac_dict:dict,row_cluster:bool,col_cluster:bool,color_map:str='bwr',save_fig:bool=False,save_path:str='./',save_fmt:str='.pdf') -> None:
        """
        Plot the Jacobian matries as the hierarchically-clustered heatmaps.
        This function requires scipy to be available.

        Parameters
        ----------
            Jac_dict: `dict`
                the non-clustered Jacobian matrix. `key` is cell type, `value` is `DataFrame` of the Jacobian matrix.
            {row,col}_cluster: `bool`, optional
                    If `True`, cluster the {rows, columns} when plot the heatmap of Jacobian matrix.
            color_map: matplotlib colormap name or object, or list of colors, optional, default is `bwr`
                The mapping from data values to color space. 
            save_fig: bool, default is `False`
                whether to save the heatmap of the clustered/non-clustered Jacobian matrix.
            save_path: `str`, default is `./`
                the path used to save the heatmap of Jacobian matrix.
            save_fmt: str='.png'
                the format to save the heatmap of Jacobian matrix.
        Returns
        ----------
            Jac_clustered: `dict`
                the clustered Jacobian matrix. `key` is cell type, `value` is `DataFrame` of the Jacobian matrix.
        """
        import seaborn as sns
        heatmap_kwargs = {'xticklabels': True, 'yticklabels': 1, 'row_colors': None, 'col_colors': None, 'row_linkage': None, 'col_linkage': None, 'method': 'average', 'metric': 'euclidean', 'z_score': None, 'standard_scale': None}
        Jac_clustered = {}
        for cell_info in Jac_dict:
            
            sns_heatmap = sns.clustermap(Jac_dict[cell_info],row_cluster=row_cluster,col_cluster=col_cluster,cmap=color_map,figsize=(8,8),center=0,**heatmap_kwargs)
            sns_heatmap.fig.suptitle('jacobian of '+cell_info,fontsize=12)
            # sns_heatmap.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
            # sns_heatmap.ax_col_dendrogram.set_visible(False) #suppress column dendrogram
            if save_fig:
                if row_cluster==True and row_cluster==True:
                    sns_heatmap.savefig(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_rowcol_clus_1'+save_fmt)
                elif row_cluster==True:
                    sns_heatmap.savefig(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_row_clus_1'+save_fmt)
                elif col_cluster==True:
                    sns_heatmap.savefig(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_col_clus_1'+save_fmt)
                
            Jac_clustered[cell_info] = sns_heatmap.data2d
        plt.show()

        return Jac_clustered

    def save_JacInfo(self,save_path:str,Jac_dict:dict,row_cluster:bool,col_cluster:bool):
        """
        save the clustered/non-clustered Jacobian matrix. `key` is cell type, `value` is `DataFrame` of the Jacobian matrix.

        Parameters
        ----------
            save_path: `str`, default is `./`
                the path used to save the results of Jacobian matrix.
            Jac_dict: `dict`
                the clustered/non-clustered Jacobian matrix. `key` is cell type, `value` is `DataFrame` of the Jacobian matrix.
            {row,col}_cluster: `bool`, optional
                    If `True`, annotate the {rows, columns} when save the results.
        Returns
        ----------
            Nothing.
        """

        for cell_info in Jac_dict:
            if row_cluster==True and row_cluster==True:
                Jac_dict[cell_info].to_csv(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_rowclo_clus_1.csv')
            elif row_cluster==True:
                Jac_dict[cell_info].to_csv(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_row_clus_1.csv')
            elif col_cluster==True:
                Jac_dict[cell_info].to_csv(save_path+cell_info.split('_')[0]+'/'+'cell_Jac_'+cell_info+'_col_clus_1.csv')
