import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
# import dynamo as dyn
import pandas as pd
import anndata as Anndata
# type set
from typing import Dict, List, Any, Tuple
from pandas import DataFrame
# dyn.dynamo_logger.main_silence()

"""
1. For acceleation in lap, there are two ways to cluster genes:
    (1) calculate correlation.
    (2) using Hilbert transformation to obtain magnitude and phase, then calculate the order parameter |r|.
# 2. Can we use the genes kinetic heatmap via observing the cluters in figure to culster the genes?
"""
class ClusterGenes():
    def __init__(self,genes_expr:np.array,genes_rank:List[str]) -> None:
        """
        cluster genes according to given cluster keys, e.g., `acceleration` or `velocity`.

        Parameters
        ----------
            genes_expr: `np.array`
                the ranked genes expression along LAP, obtained by the code `genes = gtraj.select_gene(ranking['all'])`.
            genes_rank: `List[str]`
                the ranked gene names.
        Returns
        ----------
            Nothing.
        """
        self.genes = genes_expr.copy()
        self.genes_rank = genes_rank
        pass

    def cal_plot_clusterGenes(self,top_genes:int=100,cluster_key:str='velocity', methods:str='spearman',weigh:float=0.5,thres:float=5e-3,thres_method:str='auto',top_quantile:float=0.5,save_fig:bool=False,save_path='./',fig_fmt:str='.png'):
        """
        calculate the correlations beteeween genes acoording `cluster_key` along the LAP.

        Parameters
        ----------
            top_genes: `int` (default: `100`)
                the number of selected top ranked genes used to cluster.
            cluster_key: `str` (default: `velocity`)
                the measure used to cluster. `cluster_key` can be `acceleration`, `velocity` or `expression`.
            methods: `str` (default: `spearman`)
                the correlation method used to cluster genes. `methods` can be `spearman` or `pearson`. For the large size data, `spearman` is faster than `pearson`.
            weigh: `float` (default: `0.5`)
                wight of the coorelation matrix obtained by `cluster_key`, use `X` represents the values of `cluster_key`, then `1-weigh` is the wight of the coorelation matrix obtained by `dX`. For `cluster_key='velocity'`, `dX` is `acceleration`. For `cluster_key='acceleration'`, `dX` is `acceleration`.
            thres: `float`, (default: `5e-3`)
                Threshold is used to judge abs(dx)=0.
            thres_method: `str`, (default: `auto`)
                the method to obtain the threshold, supprt `auto` and `input`. If set `thres_method` as `auto`, then this can override the `thres` input by user.
            top_quantile: `float`, (default: 0.5)
                select the top quantile absolute peaks and valleys of velocity.
            save_fig: bool, default is `False`
                whether to save the heatmap of the clustered/non-clustered Jacobian matrix.
            save_path: `str`, default is `./`
                the path used to save the heatmap of Jacobian matrix.
            fig_fmt: str='.png'
                the format to save the heatmap of Jacobian matrix.


        Returns
        ----------
            corre_clustered: `DataFrame`
                the clustered correlation matrix.
        """

        self.X = self.genes.copy()[:top_genes,:]
        self.genes_rank = self.genes_rank[:top_genes]
        if cluster_key == 'expression':
            pass

        elif cluster_key == 'velocity':
            self.X = self.X[:,1:]-self.X[:,:-1]
            self.dX = self.X[:,1:]-self.X[:,:-1]
            # calculate coorelation via acceleration=0, meanwhile, selece top abs velocities. we don't consider the marginal time points.
            corre_dX,corre_id, genes_selec = self.calCorreDx(dX=self.dX,X=self.X,thres=thres,thres_method=thres_method,top_quantile=top_quantile,cluster_key=cluster_key)

        elif cluster_key == 'acceleration':
            self.X = self.X[:,2:]-2*self.X[:,1:-1]+self.X[:,:-2]
            # calculate coorelation via acceleration=0
            corre_dX,corre_id, genes_selec = self.calCorreDx(dX=self.X,thres=thres)

        corre_mat = np.zeros((top_genes,top_genes))
        if methods == 'pearson':
            for i,x in enumerate(self.X):
                for j,y in enumerate(self.X):
                    corre_mat[i,j] = self.getPearsonr(x,y) 
        elif methods == 'spearman':
            corre_mat = self.getSpearmanr(axis=1)
        corre_mat -= np.diag(np.ones(len(corre_mat)))

        if cluster_key == 'expression':
            corre_mat = pd.DataFrame(data=corre_mat,index=self.genes_rank,columns=self.genes_rank)
        else:
            corre_mat = corre_mat[corre_id][:,corre_id]
            corre_mat = weigh*corre_mat+(1-weigh)*corre_dX
            corre_mat = pd.DataFrame(data=corre_mat,index=genes_selec,columns=genes_selec)

        corre_clustered = self.plotClustermap(data=corre_mat,row_cluster=True,col_cluster=True,cluster_key=cluster_key,methods=methods,save_fig=save_fig,save_path=save_path,fig_fmt=fig_fmt)

        return corre_clustered

    def plotClustermap(self,data:DataFrame,row_cluster:bool=True,col_cluster:bool=True,cluster_key:str='acceleration',methods:str='spearman',color_map:str='bwr',save_fig:bool=False,save_path='./',fig_fmt:str='.png') -> None:
        """
        Plot a matrix dataset as a hierarchically-clustered heatmap.
        This function requires scipy to be available.

        Parameters
        ----------
            data: 2D array-like
                Rectangular data for clustering. Cannot contain NAs.
            {row,col}_cluster: bool, optional
                If True, cluster the {rows, columns}.
            cluster_key: `str`, (default: `acceleration`)
                string used to figure title.
            method: `str`, (default: `spearman`)
                string used to figure title.
            cmap: matplotlib colormap name or object, or list of colors, optional
                The mapping from data values to color space. the default is 'bwr'.
            save_fig: `bool`, default is `False`
                whether to save the heatmap of the clustered/non-clustered Jacobian matrix.
            save_path: `str`, default is `./`
                the path used to save the heatmap of Jacobian matrix.
            fig_fmt: str='.png'
                the format to save the heatmap of Jacobian matrix.
        
        Returns
        ----------
            sns_heatmap.data2d: `DataFrame`
                the clustered correlation matrix.
        """
        import seaborn as sns
        heatmap_kwargs = {'xticklabels': True, 'yticklabels': 1, 'row_colors': None, 'col_colors': None, 'row_linkage': None, 'col_linkage': None, 'method': 'average', 'metric': 'euclidean', 'z_score': None, 'standard_scale': None}
        sns_heatmap = sns.clustermap(data,row_cluster=row_cluster,col_cluster=col_cluster,cmap=color_map,figsize=(10,10),center=0,**heatmap_kwargs)
        sns_heatmap.fig.suptitle('cluster key is "'+cluster_key+'", method is "'+methods+'"',fontsize=12)
        # sns_heatmap.ax_row_dendrogram.set_visible(False) #suppress row dendrogram
        # sns_heatmap.ax_col_dendrogram.set_visible(False) #suppress column dendrogram
        if save_fig:
            sns_heatmap.savefig(save_path+'cluster_gs_LAP_'+cluster_key[:3]+'_'+methods[:4]+'_rowcol_clus_1'+fig_fmt)
        plt.show()
        return sns_heatmap.data2d
        

    def getPearsonr(self,x:np.array,y:np.array):
        """
        Pearson correlation coefficient.

        Parameters
        ----------
            x : (N,) array_like
                Input array.
            y : (N,) array_like
                Input array.

        Returns
        -------
            Pearson product-moment correlation coefficient.
        """
        return stats.pearsonr(x, y)[0]
        

    def getSpearmanr(self,axis:int=1):
        """
        Calculate a Spearman correlation coefficient.

        Parameters
        ----------
            axis : int or None, optional
                If axis=0 (default), then each column represents a variable, with
                observations in the rows. If axis=1, the relationship is transposed:
                each row represents a variable, while the columns contain observations.
                If axis=None, then both arrays will be raveled.
        Return
        ----------
            Spearman correlation matrix.
        """
        return stats.spearmanr(self.X,axis=axis)[0]
    def calCorreDx(self,dX:np.array,X:np.array=[],thres:float=5e-3,thres_method:str='auto',top_quantile:float=0.5,cluster_key:str='velocity'):
        """
        calculate correlation between the times with dX=0. Smaller time diffences with dX=0 yield heiger correlation. 
        Supposing two genes `gi` and `gj`, with accelerations=0 corresponding to `ti` and `tj`, respectively. 

        First, calculating distance, `dij = abs(ti-tj)`, and if dij <= 1, then setting dij=0. Letting i,j are traversal, we obtain the distance matrix dist_mat. For d_ij=0 and the corresponding velocities `vi` and `vj` are opposite extremes (i.e., the corresponding time indexes to accelerations `ai=0` and `aj=0` are same), we set d_ij = max_thres.

        Second, normalizing the distance matrix via dist_mat = dist_mat/dist_mat.max(), so the values in dist_mat satisfy $d_{ij} \in [0,1]$. 

        Third, the correlation is calculated by $c_{ij} = 1-d_{ij}$, and $c_{ij} \in [0,1]$. Using the transformation $c_{ij} = 1-2*c_{ij}$ such that $c_{ij} \in [-1,1]$.

        Parameters
        ----------
            dX: `np.array`
                The derivative used to obatain peaks or valleys of the data `X`, via dX=0.
                For `velocity`, dX and X should be provided meanwhile.
                Fro `acceleration`, only dX is needed.
            X: `np.array`, (default: `[]`)
                If the correlation is calculated based on the peaks and valleys of velocity, and should select the top quantile absolute velocities, this needs to be provided. 
            thres: `float`, (default: `1e-3`)
                Threshold is used to judge abs(dx)=0.
            thres_method: `str`, (default: `auto`)
                the method to obtain the threshold, supprt `auto` and `input`. If set `thres_method` as `auto`, then this can override the `thres` input by user.
            top_quantile: `float`, (default: 0.5)
                select the top quantile absolute peaks and valleys of velocity.
            cluster_key: `str` (default: `velocity`)
                the measure used to cluster. `cluster_key` can be `acceleration`, `velocity` or `expression`.

        Return
        ----------
            correDx: `np.array`
                The resulted correlation matrix.
            row_uniq[top_id]: 1-d array
                The selceted rows in `dX`.
            genes_selec: `List[str]`
                The selcted genes in the process of calculating the correlation matrix.
        """

        y_id = []
        # for y in Yabs:
        #     y_id = np.where(y<thres)[0]
        if thres_method == 'auto':
            thres = self.get_thres(key='median')
        rows,cols = np.where(np.abs(dX)<thres)
        row_uniq = np.unique(rows)
        genes_selec = []
        for row in row_uniq:
            ir = np.where(rows==row)[0]
            y_id.append(cols[ir])
            genes_selec.append(self.genes_rank[row])
        y_id = self.deRepeat(y_id)
        margin_thres = 2
        # y_id = self.make_dict(data=y_id,keys=genes_selec)
        top_id = np.arange(len(row_uniq))
        if len(X)>0:
            print('------------------------------------------------------------')
            print('find maximum absolute velocities satisfied acceleration is 0.')
            print('------------------------------------------------------------')
            X_selec = X[row_uniq]
            dX_selec = dX[row_uniq]
            max_arr = self.find_maxabsvel(X=np.abs(X_selec),max_id=y_id)
            max_quant = np.quantile(max_arr,1-top_quantile)
            top_id = np.where(max_arr>=max_quant)[0]
            X_selec = X_selec[top_id]
            dX_selec = dX_selec[top_id]
            genes_selec = [genes_selec[i] for i in top_id]
            y_id = [y_id[i] for i in top_id]
            y_id = self.removeMargins(x_list=y_id,end=dX.shape[1]-1,init=0,margin_thres=margin_thres)
            print('------------------------------------------------------------')
            print('finding maximum absolute velocities is finished.')
            print('------------------------------------------------------------')
        
        for y in y_id:
            assert len(y)>0, "try larger threshold or smaller top_quantile."

        ly = len(y_id)
        correDx = np.zeros((ly,ly))
        for i in range(ly):
            for j in range(i+1,ly):
                correDx[i,j] = self.cal_mindist(y_id[i],y_id[j],x_selec=[X_selec[i],X_selec[j]],max_thres=(dX.shape[1]-2*margin_thres)*2,cluster_key=cluster_key)
        if correDx.max()>0:
            correDx = correDx/correDx.max()
        correDx = correDx + correDx.T
        correDx = 1 - 2*correDx
        correDx -= np.diag(np.ones(len(correDx)))

        return correDx, row_uniq[top_id], genes_selec
        
    def get_thres(self,key:str='median'):
        """
        get threshold used to judge derivative of the data (abs(dx)==0).

        Parameters
        ----------
            key: `str`, (default: `median`)
                the key used to generate the threshold, support `median` and `mean`.

        Return
        ----------
            thres: `float`
            threshold.

        """
        if key=='median':
            thres = np.median(abs(self.dX))
        elif key=='mean':
            thres = np.mean(abs(self.dX))

        return np.around(thres*0.8,5)
    
    def deRepeat(self,x_list):
        """
        remove the repeated values (satify abs(x[j]-x[j-1])<=2) in `x_list`. For our process to obatain the time with `dx=0`, we think continus time is the same one.

        Parameters
        ----------
            x_list: List[1-d array]
                the time (integers) with `dx=0`.

        Return
        ----------
            x_list: List[1-d array]
                the time (integers) with removed repeated values to `dx=0`.
        
        """
        x_len = []
        for i,x in enumerate(x_list):
            if len(x)>1:
                for j in range(1,len(x)):
                    if abs(x[j]-x[j-1])<=2:
                        x[j] = x[j-1]
            x = np.unique(x) 
            x_len.append(len(x))
            x_list[i] = x
        
        return x_list
    
    def removeMargins(self,x_list:List,end:int,init:int=0,margin_thres:int=2):
        """
        remove marginal values. For our velocity data, use abs(acc)=0 to obtain time index. Some time indexes is near to the start or end time points, so we should reomve those time. The removed time indexes `ti` stasify `end-ti>margin_thres` and `ti-init<margin_thres`.

        Parameters
        ----------
            x_list: List[1-d array]
                the time (integers) with `dx=0`.
            end: `int`
                the last time index. For 2-d data matrix, we recommend to use `end = data.shape[1]`
            init: `init`, (default: 0)
                the initial time index.
            margin_thres: `int`, (default: 2)
                the marginal threshold used to remove time.
        Return
            new_list: List[1-d array]
                the time (integers) with removed marginal values to `dx=0`.
        ----------
        """
        new_list = []
        for x in x_list:
            x = x[x>=init+margin_thres]
            new_list.append(x[x<=end-margin_thres])
        return new_list

    
    def make_dict(self,data:List[np.array],keys:List[str]):
        """
        generate dictionary for given data and keys.
        Parameters
        ----------
            data: List[1-d array]
                the list of data.
            keys:List[str]
                the list of keys.
        Return
        ----------
            data_dict: dict
                the generated dictonary.
        """
        data_dict = {}
        for i,x in enumerate(data):
            data_dict[keys[i]] = x

        return data_dict
    
    def find_maxabsvel(self,X:np.array,max_id:List):
        """
        find maximum absolute velocities satisfied acceleration is 0.
        Parameters
        ----------
            X: np.array
                velocity matrix.
            max_id: List[1-d array]
                the time index list with acceleration = 0.
        Return
        ----------
            max_arr:
                the array of maximal velocities.
        """
        max_arr = []
        for i,x_id in enumerate(max_id):
            max_arr.append(np.max(X[i][x_id]))
        return np.array(max_arr)
    def cal_mindist(self,arr1:np.array,arr2:np.array,x_selec:List=[],max_thres:float=30,cluster_key:str='velocity'):
        """
        calculate minimal distance between two arrays.
        For d_ij<=1, we set d_ij=0.
        For d_ij=0 and the corresponding velocities `vi` and `vj` are opposite extremes, we set d_ij = max_thres.
        Parameters
        ----------
            arr1: 1-d array
                the first 1-d array.
            arr2: 1-d array
                the second 1-d array.
            x_selec: List[1-d array], (default: [])
                the data (e.g., velocity) used to judge the type of extremes.
            max_thres: `float`, (default: 30)
                for d_ij=0 and the corresponding velocities `vi` and `vj` are opposite extremes (i.e., the corresponding time indexes to accelerations `ai=0` and `aj=0` are same), we set d_ij = max_thres.
            cluster_key: `str` (default: `velocity`)
                the measure used to cluster.

        Return
        ----------
            minimal distance between `arr1` and `arr2`. For d_ij=0 and the corresponding velocities `vi` and `vj` are opposite extremes, return `max_thres`.
        """
        diff_mat = abs(arr1[:,None]-arr2[None,:])
        diff_mat[diff_mat==1] = 0
        if cluster_key=='velocity' and len(x_selec)>0:
            diff_mat = self.deal_extremes(diff_mat=diff_mat,arr1=arr1,arr2=arr2,x_list=x_selec,max_thres=max_thres)
        if diff_mat.min()==0:
            return 0
        elif diff_mat.max()==max_thres:
            return max_thres
        else:
            return np.min(diff_mat)

    def deal_extremes(self,diff_mat:np.array,arr1:np.array,arr2:np.array,x_list:List,max_thres:float=30):
        """
        deal with the extremes in velocities judged by acc=0.
        for two the extremes with same type ('max' or 'min'), `diff_mat` is not changed.
        for two the extremes with opposite type ('max' or 'min'), set `diff_mat[i,j]=max_thres`.

        Parameters
        ----------
            diff_mat: np.array
                the original distance matrix.
            arr1: 1-d array
                the first 1-d array.
            arr2: 1-d array
                the second 1-d array.
            x_selec: List[1-d array], (default: [])
                the data (e.g., velocity) used to judge the type of extremes.
            max_thres: `float`, (default: 30)
                for d_ij=0 and the corresponding velocities `vi` and `vj` are opposite extremes (i.e., the corresponding time indexes to accelerations `ai=0` and `aj=0` are same), we set d_ij = max_thres.

        Return
        ----------
            diff_mat: np.array
            the corrected distance matrix.
        """
        # id1, id2 corresponds to the index of gene1 (i.e.,arr1), and gene2 (i.e.,arr1) respectively. len(id1)==len(id2)
        id1, id2 = np.where(diff_mat==0)
        for i in range(len(id1)):
            max_min1 = self.judge_maxmin(x_list[0],arr1[id1[i]])
            max_min2 = self.judge_maxmin(x_list[1],arr2[id2[i]])
            if max_min1 != max_min2:
                diff_mat[id1,id2] = max_thres
        return diff_mat

    def judge_maxmin(self,x_arr:np.array,extrem_id:int):
        """
        judge the type ('max' or 'min') of extreme.

        Parameters
        ----------
            x_arr: 1-d array
                data (e.g., velocity) array.
            extreme_id: int
                the index of the extreme.
        Return
        ----------
            the type ('max' or 'min')  of extreme.
        """
        extrem_flag = True
        if extrem_id == 0:
            extrem_flag = x_arr[extrem_id]>np.mean(x_arr[extrem_id+1:extrem_id+3])
        elif extrem_id == 1 or extrem_id == len(x_arr)-2: 
            extrem_flag = x_arr[extrem_id]>(x_arr[extrem_id-1]+x_arr[extrem_id+1])/2
        elif extrem_id == len(x_arr)-1:
            extrem_flag = x_arr[extrem_id]>np.mean(x_arr[extrem_id-2:extrem_id])
        else:
            extrem_flag = x_arr[extrem_id]>(x_arr[extrem_id-2]+x_arr[extrem_id-1]+x_arr[extrem_id+1]+x_arr[extrem_id+2])/4
        if extrem_flag:
            return 'max'
        else:
            return 'min'