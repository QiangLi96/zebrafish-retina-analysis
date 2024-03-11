import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# import Scribe as sb
import sys
import os

# import scanpy as sc
import dynamo as dyn
dyn.dynamo_logger.main_silence()

class PerturbInfo():
    ### calculate some useful informations after perturbing genes, to select representative patterns
    def __init__(self,adata) -> None:
        self.adata = adata
        pass
    def calAngle_vel(self,x,y):
        """
        calculate the angle between two vectors `x` and `y`
        """
        return math.acos(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    
    def cal_cosAngle(self,data_ori,data_new):
        """
        calculate the cosine similarities between two data `data_ori` and `data_new`.
        each row in `data_ori` or `data_new` is considered as a vector.

        Parameters
        ----------
            data_ori: `np.array`
                the original data used to calculate cosine similarity, each row is considered as a vector.                
            data_new: `np.array`
                the new data used to calculate cosine similarity, each row is considered as a vector. 

        Returns
        ----------
        cosAngle: `np.array`
            cosine similarity between `data_ori` and `data_new`.

        """

        # normalize data matrix
        data_ori = data_ori/np.linalg.norm(data_ori,axis=1,keepdims=True)
        data_new = data_new/np.linalg.norm(data_new,axis=1,keepdims=True)
        cosAngle = np.sum(np.multiply(data_ori,data_new),axis=1)
        
        return cosAngle
    
    def calAngleChange(self,basis:str='umap',cell_type:list=[]):
        """
        code in this part is not finished, especially for len(cell_type)>0.
        cell_type:
            list of cell types, must like [[c1],[c2]].
        """
        if basis == 'umap':
            data_ori = self.adata.obsm['velocity_umap']
            data_new = self.adata.obsm['velocity_umap_perturbation']
        else:
            data_ori = self.adata.obsm['velocity_pca']
            data_new = self.adata.obsm['j_delta_x_perturbation']
        if len(cell_type)==0 :
            angle_arr = self.calAngleArr(data_ori=data_ori,data_new=data_new)
        else:
            # get cell position
            n_obs = data_ori.shape[0]
            angle_tmp = np.zeros(n_obs)
            cell_loc = [self.adata.obs_names.get_loc(i) for i in self.adata[self.adata.obs['Cell_type'].isin(cell_type)].obs_names]
            for loc in cell_loc:
                angle_tmp[loc] = angle_arr[loc]
            angle_arr = angle_tmp
            

        return angle_arr
    
    def calSimilar(self,ref_gene:list=[],ref_exps:list=[],test_genes=[],test_exps=[],basis:str='umap',cell_type:list=[],cell_range:list=None,filterTh:float=0.1,save_path:str='',fig_ext:str='.png'):
        """
        calculate the cosine similarity  between pre-pertured and post_perturbed velocity data in the given cell_type.

        Parameters
        ----------
            ref_gene: `list`
                the reference gene used to calculate similarity with `test_genes`. Length of `ref_gene` must be `1`.
            ref_exps: `list`
                the corresponding expression of the reference gene `ref_gene`.
            test_genges: `list`
                the selected test genes used to calculate similarity with `ref_genes`.
            test_exps: `list`
                the corresponding expression of `test_gene`.
            basis: `str`, default is `umap`.
                the basis used to calculte the similarity. can be set as `pca`.
            cell_type: `list`
                cell types used to calculate similarity.
            cell_range: `list[list[]]`
                boundaries of the rectangle (x_left,x_right,y_up,y_down), to select cells within the specified range of the rectangle.
            filterTh: `float`
                the threshold used to select cells. for the reference or the test genes, if the sum of the squares of the genes within the cell is greater than the threshold `filterTh`, we calculate the similarity. Otherwise, the similarity is set to 0. 
            save_path: `str`
                path used to save figure.
            fig_ext: `str`, default is `.png`     
                save format of figure.

        Returns
        ----------
            similar_mat: `np.array`
                The resulted similarity between `ref_gene` and `test_gene` in each cell.
            cell_loc: `list`
                The recorded cell index for given cell types `cell_type`.  
        """
        if len(ref_gene) != 1:
            raise ValueError('`ref_gene` is required to be of type `list` or `np.array`, and its length must be exactly 1.')
        if len(test_genes) != len(test_exps):
            raise ValueError('list `test_genes` and list `test_exps` must have the same length')
        
        self.mkdir(path=save_path+ref_gene[0])
        save_path = save_path+ref_gene[0]+'/'
        dyn.pd.perturbation(self.adata, ref_gene[0],expression=ref_exps,emb_basis="umap",projection_method='cosine')
        save_kwargs = {"path": save_path, "prefix":  ref_gene[0]+'_perturb', "dpi": None, "ext": fig_ext[1:], "transparent": True, "close": True, "verbose": True}
        dyn.pl.streamline_plot(self.adata, color=['Cell_type'], basis='umap_perturbation',save_show_or_return='save',save_kwargs=save_kwargs)
        if basis == 'umap':
            data_ref = self.adata.obsm['velocity_umap_perturbation']
        else:
            data_ref = self.adata.obsm['j_delta_x_perturbation']
        
        # get cell position
        if len(cell_type) == 0:
            cell_loc0 = np.arange(self.adata.n_obs)
        else:
            cell_loc0 = [self.adata.obs_names.get_loc(i) for i in self.adata[self.adata.obs['Cell_type'].isin(cell_type)].obs_names]
        if cell_range!=None:
            cell_loc1 = self.cells_rectangle(cell_range=cell_range)
            cell_loc = list(set(cell_loc0).intersection(set(cell_loc1)))
        else:
            cell_loc = cell_loc0
        
        data_ref = data_ref[cell_loc]

        data_test = []
        for i, gene in enumerate(test_genes):
            dyn.pd.perturbation(self.adata, gene,expression=test_exps[i],emb_basis="umap",projection_method='cosine')
            save_kwargs = {"path": save_path, "prefix":  gene+'_perturb', "dpi": None, "ext": fig_ext[1:], "transparent": True, "close": True, "verbose": True}
            dyn.pl.streamline_plot(self.adata, color=['Cell_type'], basis='umap_perturbation',save_show_or_return='save',save_kwargs=save_kwargs)
            if len(cell_type) == 0 and cell_range==None:
                if basis == 'umap':
                    data_test.append(self.adata.obsm['velocity_umap_perturbation'])
                else:
                    data_test.append(self.adata.obsm['j_delta_x_perturbation'])
            else:
                if basis == 'umap':
                    data_test.append(self.adata.obsm['velocity_umap_perturbation'][cell_loc])
                else:
                    data_test.append(self.adata.obsm['j_delta_x_perturbation'][cell_loc])
            
        # calculate cosine similarity
        # first, we should obtain angle difference between reference and test velocities
        # second, similiarity = np.cos(angle difference)
        similar_mat = []
        # filter data
        data_ref, id_ref = self.filter_data(x_arr=data_ref,thres=filterTh,basis=basis)
        for i in range(len(test_genes)):
            test_tmp,id_test = self.filter_data(x_arr=data_test[i],thres=filterTh)
            id_intersec = list(set(id_ref).intersection(set(id_test)))
            similar_tmp = np.zeros(len(data_ref))
            similar_tmp[id_intersec] = self.cal_cosAngle(data_ori=data_ref[id_intersec],data_new=test_tmp[id_intersec])
            similar_mat.append(similar_tmp)
        similar_mat = np.array(similar_mat)

        # # rank genes about similarity
        # rank_id = self.rank_genes(similarMat=similar_mat,method=rank_method)
        # rank_genes = np.array(test_genes)[rank_id]
        # similar_mat = similar_mat[rank_id]

        # plot correlations between reference gene and test genes
        # self.scatterPlot(similar_tmp=similar_mat,test_genes=test_genes,ref_gene=ref_gene[0],cell_loc=cell_loc,cmap=cmap,alpha=alpha,save_fig=save_fig,show_fig=show_fig, save_path=save_path,fig_ext=fig_ext)

        return similar_mat, cell_loc
         
    def filter_data(self,x_arr:np.array,thres:float=0.1,basis:str='umap'):
        """
        Filter small values (sum of the squares of each row < `thres`) in x_arr by setting them as `0`.

        Parameters
        ----------
            x_arr: `np.array`
                the data used to filter.
            thres: `float`, default is `0.1`
                Threshold used to filter data.
            basis: `str`
                Basis of the data `x_arr`, can be 'pca'.

        Returns
        ----------
            the filtered data and the row index of `x_arr` >= `thres`.
        """
        x_len = np.sum(x_arr**2,axis=1)
        filterLen = np.sum(x_len<thres)
        if basis == 'umap':
            dim2 = 2
        else:
            dim2 = 30
        x_arr[x_len<thres] = np.zeros((filterLen,dim2))

        return x_arr, np.where(x_len>=thres)[0]
    
    def cells_rectangle(self,cell_range):
        """
        select cells in some rectangles.

        Parameters
        ----------
        cell_range: `list[list[]]`
            boundaries of the rectangle (x_left,x_right,y_up,y_down), to select cells within the specified range of the rectangle.

        Returns
        ----------   
            indexes of the cells in the given rectangles.
        """
        X_umap = self.adata.obsm['X_umap']
        xy_range = cell_range[0]
        x0,x1,y0,y1 = xy_range
        bool_x = np.logical_and(x0<=X_umap[:,0],X_umap[:,0]<=x1)
        bool_y = np.logical_and(y0<=X_umap[:,1], X_umap[:,1]<=y1)
        bool_xy = np.logical_and(bool_x,bool_y)

        for xy_range in cell_range[1:]:
            x0,x1,y0,y1 = xy_range
            # umap_1 = X_umap[:,0][np.logical_and(x0<=X_umap[:,0],X_umap[:,0]<=x1)]
            # umap_2 = X_umap[:,1][np.logical_and(y0<=X_umap[:,1], X_umap[:,1]<=y1)]
            # X_umap[np.logical_and(x0<=X_umap[:,0],X_umap[:,0]<=x1),np.logical_and(y0<=X_umap[:,1], X_umap[:,1]<=y1)]
            bool_x = np.logical_and(x0<=X_umap[:,0],X_umap[:,0]<=x1)
            bool_y = np.logical_and(y0<=X_umap[:,1], X_umap[:,1]<=y1)
            bool_xy = np.logical_or(bool_xy,np.logical_and(bool_x,bool_y)) 
        
        return np.where(bool_xy==True)[0]
    
    def rank_genes(self,similarMat,method:str='abs_sum'):
        """
        Rank genes about similarity using give method.

        Parameters
        ----------
            similarMat: `np.array`
                the simiartiy array, each element represents the similarity between pre-perturbed and post-perturbed genes in each cell.
            method: `str`, default is `abs_sum`.
                the method used to calculate similarity, can be `sum`. if set as `abs_sum`, we sum all the absolute values of columns in each row then rank genes. if set as `sum`, we sum all columns in each row then rank genes. 
        Returns
            rankid: 1-d `np.array`
                the index of the descending genes.
        ----------

        """
        if method == 'abs_sum':
            similarMat = np.abs(similarMat)
        elif method == 'sum':
            similarMat = similarMat.copy()
        # obtain the length of non-zero in each row for similarMat
        similar_len = []
        for similar_tmp in similarMat:
            similar_len.append(sum(abs(similar_tmp)>0))
        similarSum = similarMat.sum(axis=1)/np.array(similar_len)
        rank_id = np.argsort(-similarSum)

        return rank_id
        
    def scatterPlot(self,similar_tmp,test_genes,ref_gene:str,cell_loc:list,cmap:str='viridis',mark_size:float=10,alpha:float=0.8,similarTh:float=0.1,vmin=None,vmax=None,show_simliarHist:bool=True,bins:int=50,save_fig:bool=True,show_fig:bool=False, save_path:str='',fig_name:list=[],fig_ext:str='.png'):
        """
        plot the similarity results, each dot represent one cell.

        Parameters
        ----------
            similar_tmp: `np.array`
                the simiartiy array, each element represents the similarity between pre-perturbed and post-perturbed genes in each cell.
            test_genes: `str`
                the genes used to calculate `similar_tmp`. one gene is corresponding to one row in `similar_tmp`.
            ref_gene: `str`
                the reference gene used to calculate similarity with `test_genes`.
            cell_loc: list
                The recorded cell index for calculating similarities.  
            cmap: `str`
                The Colormap instance or registered colormap name used to map scalar data to colors.
            mark_size: `float`
                size of marker used to plot scatters.
            show_simliarHist: `bool`, default is `True`
                whether show the histgram of the similarity (with values != 0) distribution.
            bins: `int`, default is `20`
                the number of bins for `plt.hist`.
            alpha: `float`
                The alpha blending value, between 0 (transparent) and 1 (opaque).
            similarTh: `float`
                Threshold used to filter data. We plot the data with similarity >= `similarTh`.
            vmin, vmaxfloat, optional
                vmin and vmax define the data range that the colormap covers. By default, the colormap covers the complete value range of the supplied data. 
            save_fig: `bool`, default is `True`
                Whether save figure.
            show_fig: `bool`, default is `False`
                Whether show figure.
            save_path: `str`
                path used to save figure.
            fig_name: `list[str]`, default `[]`
                To make the order of the plotted figure same as the ranked genes, we offer the user can set the file name of the figure.
                if fig_name is not inputted, the default file name is 'similarity_'+ref_gene+'_'+test_gene[i].
            fig_ext: `str`, default is `.png`  
                figure format.

        Returns
        ----------   
            Nothing but plot a figure.
        """

        similar_mat = np.zeros((similar_tmp.shape[0],self.adata.n_obs))
        similar_mat[:,cell_loc] = similar_tmp
        cell_pos = self.adata.obsm['X_umap']
        for i,gene in enumerate(test_genes):
            if show_simliarHist:
                plt.figure(figsize=[6.4*2,4.8])
                plt.subplot(1,2,1)
                id_similarTh = np.where(np.abs(similar_mat[i])>=similarTh)[0]
                # id_similarTh = np.where(similar_mat[i]>=similarTh)[0]
                # id_similarTh = np.where(similar_mat[i]<=similarTh)[0]
                plt.suptitle(ref_gene+' and '+gene,fontsize=15)
                plt.scatter(cell_pos[:,0],cell_pos[:,1],s=10,c=similar_mat[i],cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.01)
                fig = plt.scatter(cell_pos[:,0][id_similarTh],cell_pos[:,1][id_similarTh],s=mark_size,c=similar_mat[i][id_similarTh],cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
                cbar = plt.colorbar(fig)
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label('Similarity',fontsize=15)
                #show grid
                plt.grid(alpha=0.4)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel(r'$\mathrm{umap}_1$',fontsize=15)
                plt.ylabel(r'$\mathrm{umap}_2$',fontsize=15)

                plt.subplot(1,2,2)
                similar_tmp = similar_mat[i]
                plt.hist(x=similar_tmp[abs(similar_tmp)>0],bins=bins,density=False,alpha=alpha)
                plt.xticks([-1.0,-0.5,0,0.5,1.0],fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel(r'similarity',fontsize=15)
                plt.ylabel(r'number',fontsize=15)
                # to solve overlap between y-axes or y_labels for showing figure
                plt.tight_layout()
                
            else:
                plt.figure(figsize=[6.4,4.8])
                id_similarTh = np.where(np.abs(similar_mat[i])>=similarTh)[0]
                # id_similarTh = np.where(similar_mat[i]>=similarTh)[0]
                # id_similarTh = np.where(similar_mat[i]<=similarTh)[0]
                plt.scatter(cell_pos[:,0],cell_pos[:,1],s=10,c=similar_mat[i],cmap=cmap,vmin=vmin,vmax=vmax,alpha=0.01)
                fig = plt.scatter(cell_pos[:,0][id_similarTh],cell_pos[:,1][id_similarTh],s=mark_size,c=similar_mat[i][id_similarTh],cmap=cmap,vmin=vmin,vmax=vmax,alpha=alpha)
                cbar = plt.colorbar(fig)
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label('Similarity',fontsize=15)
                plt.title(ref_gene+' and '+gene,fontsize=15)
                plt.xticks(fontsize=15)
                plt.yticks(fontsize=15)
                plt.xlabel(r'$\mathrm{umap}_1$',fontsize=15)
                plt.ylabel(r'$\mathrm{umap}_2$',fontsize=15)


            if save_fig:
                if len(fig_name)>0:
                    if len(test_genes) != len(fig_name):
                        raise ValueError('list `test_genes` and list `fig_name` must have the same length.')
                    plt.savefig(save_path+fig_name[i]+fig_ext)
                else:
                    plt.savefig(save_path+'similarity_'+ref_gene+'_'+gene+fig_ext)
        if show_fig:
            plt.show()
        # # # or plot correlation results via the package of dynamo,
        # from dynamo.plot.scatters import scatters
        # for i,gene in enumerate(rank_genes):
        #     scatters(self.adata,basis='umap',values=similar_mat[i],)

    
    def plotSimilar_dist(self,df_tmp:pd.DataFrame,alpha:float=0.8):
        for gene in df_tmp.columns:
            plt.figure(figsize=[6.4,4.8])
            pass
    
    def mkdir(self,path):
        """
        creat folder for given path
        """
        folder = os.path.exists(path)

        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
            print("---  new folder...  ---")
            print("---  OK  ---")

        else:
            print("---  There is this folder!  ---")
    
        
# # ----------------------------------------------------------------------------------------------
# ###
# # load data 
adata = dyn.read_h5ad("zebrafish_H44atoh7_1/data/H44DL1208_pres_processed_data.h5ad")

# gene = 'atoh7'
# exps = [100]
# `projection_method` can be `fp`, `cosine`, `pearson`
# dyn.pd.perturbation(adata, gene,expression=exps,emb_basis='umap',projection_method='cosine')
# # # perturb specified genes in given cell types.
# # dyn.pd.perturbation(adata, gene,expression=exps, cells=[],emb_basis='umap',projection_method='cosine')
# ## in `perturbation.py` of dyn package, 
# # adata.obsm['X_umap_perturbation'] = adata.obsm['X_' + 'umap'].copy()
# # adata.obsm['velocity_umap_perturbation'] has some problems
# save_path = 'F:/python_work_PyCharm/work_7_20220317/zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/genes_perturb/'
# save_kwargs = {"path": save_path, "prefix": gene+'_perturb', "dpi": None, "ext": 'png', "transparent": True, "close": True, "verbose": True}
# dyn.pl.streamline_plot(adata, color=['Cell_type', gene], basis='umap_perturbation',save_show_or_return='save',save_kwargs=save_kwargs)


# from dynamo.plot.scatters import docstrings, scatters

### calculate angles' change of the velocity from pre-perturbation to post-perturbation
perturb_info = PerturbInfo(adata=adata)
ref_gene = ['atoh7']
ref_exps = [100]




### set test_genes
test_genes = ['abhd6a','hey1']
# test_genes = ['abhd6a','hey1','otx2b','actb1','meis2a','sox6','lhx4']
# ### read external genes, merge with `adata.var['use_for_pca']` genes
# data_path = 'F:/python_work_PyCharm/work_7_20220317/zebrafish_H44atoh7_1/'
# TF_list = pd.read_csv(data_path+'/TF_Iist_daniorerio.txt', sep=" ",names=['all'])
# print('Loadinge file, length of genes is '+str(len(TF_list)))
# TF_genes = set(TF_list['all'])
# print('Remove duplicates, length of genes is '+str(len(TF_genes)))
# pca_genes = set(adata.var_names[adata.var['use_for_pca']])
# print('`adata.var[use_for_pca]`, length of genes is '+str(len(pca_genes)))
# test_genes = list(pca_genes.intersection(TF_genes))
# print('test_genes, length of intersected genes is '+str(len(test_genes)))
# # set test expression
# test_exps = [100]*len(test_genes)
# test_exps = [100,1000,-1000,100,100,100,100]
test_exps = [100,1000]

cell_type = ['pre1','PRpre','RGCs']
cell_range=[[-6,-4,2.5,5],[-2,0,0,5],[-4,-2,7.5,10]]

save_path = 'F:/python_work_PyCharm/work_7_20220317/zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/genes_perturb/'

# default cell_range is 'None', length of cell_range is arbitrary, not same as cell_types
similar_arr,cell_loc = perturb_info.calSimilar(ref_gene=ref_gene,ref_exps=ref_exps,test_genes=test_genes,test_exps=test_exps,basis='umap',cell_type=cell_type,cell_range=cell_range,filterTh=0.1,save_path=save_path,fig_ext='.png')
save_path = 'F:/python_work_PyCharm/work_7_20220317/zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/genes_perturb/'+ref_gene[0]+'/'
df_similar = pd.DataFrame(data=similar_arr.T,columns=test_genes)
df_similar.to_csv(save_path+'genes_similar_1.csv',index=False)
# save the indices for given cell types `cell_type`
np.save(save_path+'cells_location_1.npy',cell_loc)

# obtain the filtered data number
# df_similar = pd.read_csv(save_path+'genes_similar_1.csv')
print('the original cell number')
print(df_similar.shape[0])
df_tmp = df_similar.abs()>0
print('the filtered cell number')
# print(df_tmp.sum(axis=0))
print(df_tmp.columns.values)
print(df_tmp.sum(axis=0).values)


# `rank_method` can be 'sum' or 'abs_sum' 
print('--------------------------------------------------------------------------')
print('rank genes according to sum the abosulte values of the cosine similarities')
id_absSum = perturb_info.rank_genes(similarMat=similar_arr,method='abs_sum')
genes_absSum = np.array(test_genes)[id_absSum]
print(genes_absSum)
df_absSum = pd.DataFrame(data=similar_arr[id_absSum].T,columns=genes_absSum)
df_absSum.to_csv(save_path+'genes_rank_absSum_similar_1.csv',index=False)
# plot the ranked results
# perturb_info.scatterPlot(similar_tmp=similar_arr[id_absSum],test_genes=genes_absSum,ref_gene=ref_gene[0],cell_loc=cell_loc,cmap='bwr',alpha=0.8,mark_size=10,similarTh=0.1,save_fig=True,show_fig=False,fig_name=fig_name,save_path=save_path,fig_ext='.png')

print('--------------------------------------------------------------------------')
print('rank genes according to sum the cosine similarities:')
id_sum = perturb_info.rank_genes(similarMat=similar_arr,method='sum')
genes_sum = np.array(test_genes)[id_sum]
print(genes_sum)
df_sum = pd.DataFrame(data=similar_arr[id_sum].T,columns=genes_sum)
df_sum.to_csv(save_path+'genes_rank_sum_similar_1.csv',index=False)
fig_name = ['similar_'+str(i+1)+'_'+gene for i,gene in enumerate(genes_sum)]
# plot the ranked results
perturb_info.scatterPlot(similar_tmp=similar_arr[id_sum],test_genes=genes_sum,ref_gene=ref_gene[0],cell_loc=cell_loc,cmap='bwr',alpha=0.8,vmin=None,vmax=None,mark_size=5,similarTh=0.1,show_simliarHist=True,bins=50,save_fig=True,show_fig=False,fig_name=fig_name,save_path=save_path,fig_ext='.png')




### calculate changes of velocity angles from pre-perturbed data to post-perturbed data
### for given cell types, the code is not correct, shuld be modified
# angle_arr  = perturb_info.calAngleChange(basis='umap')
# # adata.obsm['vel_change_perturb_umap'] = angle_arr
# adata.obs['vel_change_perturb_umap'] = angle_arr
# # dyn.pl.umap(adata, color='Cell_type', show_legend='on data')
# dyn.pl.umap(adata, color='vel_change_perturb_umap', show_legend='on data')
# scatters(adata=adata,basis='umap',values=angle_arr,cmap='bwr')

# ### transfer pca data to original space
# class Transpca():
#     # trajectory related
#     def pca_to_expr(self,X,PCs, mean=0, func=None):
#         # reverse project from PCA back to raw expression space
#         if PCs.shape[1] == X.shape[1]:
#             exprs = X @ PCs.T + mean
#             if func is not None:
#                 exprs = func(exprs)
#         else:
#             raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[1], X.shape[1]))
#         return exprs
    
#     def expr_to_pca(self,expr, PCs, mean=0, func=None):
#         # project from raw expression space to PCA
#         if PCs.shape[0] == expr.shape[1]:
#             X = (expr - mean) @ PCs
#             if func is not None:
#                 X = func(X)
#         else:
#             raise Exception("PCs dim 1 (%d) does not match X dim 1 (%d)." % (PCs.shape[0], expr.shape[1]))
#         return X

# tans_pca = Transpca()
# Pcs = adata.uns['PCs']
# adata.uns['pca_mean']
# X_pca = adata.obsm['X_pca']
# X_ori = tans_pca.pca_to_expr(X=X_pca,PCs=Pcs,mean=adata.uns['pca_mean'])

print()


