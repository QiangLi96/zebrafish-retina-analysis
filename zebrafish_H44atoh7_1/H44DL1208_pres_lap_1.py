import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dynamo as dyn
import os
import anndata as Anndata

# type set
from typing import Dict, List, Any, Tuple
from pandas import DataFrame

dyn.dynamo_logger.main_silence()
###
# ----------------------------------------------------------------------------------------------


# ## Data Cleaning
# adata: Anndata = dyn.read("zebrafish_H44atoh7_1/data/H44_H48_H60_H72_DL_1208_pres.h5ad")
# # umap1: DataFrame = adata.obs['umap_1']
# # umap2: DataFrame = adata.obs['umap_2']
# adata.layers['spliced'] = adata.layers['spliced'].astype('int64')
# adata.layers['unspliced'] = adata.layers['unspliced'].astype('int64')
# adata.X = adata.layers['spliced']
# adata.X = adata.X.astype('float32')
# # adata.obs['timebatch'] = adata.obs['timebatch'].astype('str')
# # adata.obs['timebatch'] = adata.obs['timebatch'].astype('category')
# del adata.layers['ambiguous']
# del adata.layers['matrix']


# ## dynamo recipe data
# dyn.pp.recipe_monocle(adata)
# dyn.tl.dynamics(adata, model='stochastic', cores=3)
# # or dyn.tl.dynamics(adata, model='deterministic')
# # or dyn.tl.dynamics(adata, model='stochastic', est_method='negbin')

# ## dynamo reconstruct vector feild in umap space
# dyn.tl.reduceDimension(adata)
# adata.obsm['X_umap'][:, 0] = adata.obs['umap_1']
# adata.obsm['X_umap'][:, 1] = adata.obs['umap_2']

# # adata.obsm['X_umap'] = np.load('zebrafish_H44atoh7_1\Figures_H44_DL_1208_pres\H44_DL_1208_pres_umap_auto_1.npy')

# dyn.pl.umap(adata, color='Cell_type', show_legend='on data')


# #dyn.tl.cell_velocities(adata, method='pearson', other_kernels_dict={'transform': 'sqrt'})
# # dyn.tl.cell_velocities(adata, method='cosine', other_kernels_dict={'transform': 'sqrt'})
# # dyn.tl.cell_velocities(adata, method='fp', other_kernels_dict={'transform': 'sqrt'})

# # You can check the confidence of cell-wise velocity to understand how reliable the recovered velocity is across cells or even correct velocty based on some prior:
# dyn.tl.cell_wise_confidence(adata)
# dyn.tl.confident_cell_velocities(adata, group='Cell_type', lineage_dict={'pre1': ['RGCs', 'GABAergic ACs', 'PRs'], 'PRpre': 'PRs', 'pre2': ['Glycinergic ACs', 'BCs'], 'HCpre': 'HCs'})

# ## dynamo reconstruct vector feild in umap space
# dyn.pl.cell_wise_vectors(adata, color=['Cell_type'], basis='umap', show_legend='on data', quiver_length=6,quiver_size=6, pointsize=0.1, show_arrowed_spines=False)

# dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='umap', show_legend='on data',
#             show_arrowed_spines=True)
# # you can set `verbose = 1/2/3` to obtain different levels of running information of vector field reconstruction
# # M is the number of control points
# dyn.vf.VectorField(adata, basis='umap', n=100, M=1000,pot_curl_div=True)
# dyn.pl.topography(adata, basis='umap', background='white', color=['ntr', 'Cell_type'],streamline_color='black',
#                   show_legend='on data', frontier=True,save_show_or_return='return')
# # black: absorbing fixed points;
# # red: emitting fixed points;
# # blue: unstable fixed points.

# adata = dyn.read_h5ad("zebrafish_H44atoh7_1/data/H44DL1208_pres_processed_data.h5ad")
# ## dynamo reconstruct vector feild in pca space
# dyn.tl.cell_velocities(adata, basis='pca', method='pearson', other_kernels_dict={'transform': 'sqrt'})
# dyn.pl.cell_wise_vectors(adata, color=['Cell_type'], basis='pca', show_legend='on data', quiver_length=6, quiver_size=6, pointsize=0.1, show_arrowed_spines=False)
# dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='pca', show_legend='on data',
#             show_arrowed_spines=True)
# # you can set `verbose = 1/2/3` to obtain different levels of running information of vector field reconstruction
# # M is the number of control points
# dyn.vf.VectorField(adata, basis='pca',n=100,M=1000, pot_curl_div=True,method='SparseVFC')
# dyn.pl.topography(adata, basis='pca',background='white', color=['ntr', 'Cell_type'],streamline_color='black', show_legend='on data', frontier=True)


# # ----------------------------------------------------------------------------------------------
# ###

# ### Dynamo save utility
# # there may be intermediate results stored in adata.uns that can may lead to errors when writing the h5ad object.
# # call dyn.cleanup(adata) first to remove these data objects before saving the adata object.
# dyn.cleanup(adata)
# # # call AnnData write_h5ad to save the entire adata information.
# adata.write_h5ad("zebrafish_H44atoh7_1/data/H44DL1208_pres_processed_data.h5ad")
# # ----------------------------------------------------------------------------------------------


# define the least action path (LAP) class.
class CellLAP:
    
    def __init__(self,adata:Anndata,cell_type:List,start_coord:np.ndarray,end_coord:np.ndarray,startCellType:List=[],fps:List[int]=[]) -> None:
        """
        Initialize the attributes of the class CellLAP

        Parameters
        ----------
            adata:
                Anndata object that has vector field function computed.
            cell_type:
                cell type used to calculate the least action path (LAP).
            start_coord:
                the start coordinates of the LAP.
            end_coord:
                the start coordinates of the LAP.

        Returns
        -------
        the initialized CellLAP object.
        """
        self.adata = adata
        assert len(cell_type)==len(start_coord)==len(end_coord), "The length of the parameters'cell_type', 'start_coord', 'end_coord' is same."
        self.cell_type = cell_type
        # find the cells nearby the start/end coordinates
        if startCellType!=None:
            self.start_indices = self.calculate_indices(start_coord,startCellType)
        else:
            self.start_indices = self.calculate_indices(start_coord,cell_type)
        self.end_indices = self.calculate_indices(end_coord,cell_type)
        # if end_points come from fixed points, i.e., compare different attractors in one cell type
        if len(fps)>0:
            self.fps=fps
    
    # return cells in adata
    def select_cell(self) -> Dict:
        """
        find the indexes in adata for each cell type in self.cell_type.
        """
        ct_indexes = {}
        for ct in self.cell_type:
            ct_indexes[ct] = dyn.tl.select_cell(self.adata, "Cell_type", ct)
        
        return ct_indexes
    
    # calculate start or end indices nearest the given extreme_points
    def calculate_indices(self,extreme_points:np.ndarray,cell_type:List[int],ind_select:int=0) -> np.ndarray:
        """
        calculate start or end cell indices nearest the given extreme_points
        Parameters
        ----------
            extreme_points:
                the coordinates will be used to find the nearest neighbor cells in adata.
            cell_type:
                select the nearest_neighbors belong to the corresponding cell type.
            ind_select:
                because the nearest_neighbors may be not only one, 'ind_select' used to select the fisrt ind_select nearest_neighbors
        Returns
        -------
        the cell indices in adata.
        """
        from dynamo.tools.utils import nearest_neighbors
        indices = []
        for i,index in enumerate(extreme_points):
            ind_=nearest_neighbors(index, self.adata.obsm["X_umap"])
            ind_=ind_[:,self.adata[ind_.reshape(-1),:].obs["Cell_type"].isin([cell_type[i]])]
            indices.append(ind_[:,:ind_select+1])
        return indices
    
    # show start and end points on cell umap
    def plot_start_end_cells(self) -> None:
        """
        plot the start and end cells for each cell type saved in self.cell_type. The background is the adata.obs['Cell_type'] umap.
        """

        from dynamo.plot.scatters import scatters
        scatters(self.adata, "umap",color='Cell_type',pointsize=0.5 ,show_legend='on data',)
        
        # plot start points
        for indices in self.start_indices:
            if np.allclose(indices,self.start_indices[0]):
                plt.scatter(*self.adata[indices[0][0]].obsm["X_umap"].T,c='k',marker='*',label='start')
            else:
                plt.scatter(*self.adata[indices[0][0]].obsm["X_umap"].T,c='k',marker='*')
        
        # plot end points
        for indices in self.end_indices:
            if np.allclose(indices,self.end_indices[0]):
                plt.scatter(*self.adata[indices[0][0]].obsm["X_umap"].T,c='k',label='end')
            else:
                plt.scatter(*self.adata[indices[0][0]].obsm["X_umap"].T,c='k')
        plt.legend(fontsize=15)
        plt.xlabel("UMAP0",fontsize=15)
        plt.ylabel("UMAP1",fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()

    

    # calculate trasition graph
    def calculate_transit_graph(self,min_lap_t:bool=True,top_num:int=5,lap_measures_key:str='pca',save_path:str='./',compared_fps:bool=False,TFs:DataFrame=[],LAPisinCells:bool=False,cluster_genes:bool=False,cluster_dict:dict={}) -> Dict:
        """
        calculate the trasition graph for each cell type along the LAP. 
        Parameters
        ----------
            top_num:
                the number of top genes ranked by MSD_i 
            lap_measures_key:
                the basis is used to calculate the interested measures, e.g., action, distance, etc. Only support 'pca' or 'umap'.
            save_path:
                the path to save results.
            compared_fps:
                whether the defined class is to calculate LAPs for different fixed points in the same cell type.
            LAPisinCells:
                judge whether the calculated LAP has the corresponding cells for given cell type.
        Returns
        -------
            A dictionary saves the trasition graph.
        """
        transition_graph = {}
        lap_measures:DataFrame = pd.DataFrame(columns=['t','action','distance'])
        lap_genesRank:DataFrame=pd.DataFrame()
        for i, start in enumerate(self.start_indices):
            j = i
            end = self.end_indices[j]
            self.mkdir(save_path+self.cell_type[i])
            # lap in umap space
            dyn.pd.least_action(
                self.adata,
                [self.adata.obs_names[start[0]][0]],
                [self.adata.obs_names[end[0]][0]],
                basis="umap",
                adj_key="umap_distances",
                min_lap_t= min_lap_t,
                EM_steps=2,
                n_points=25
            )
            # lap_umap = self.adata.uns['LAP_umap']
            # lap_dict = self.adata.uns['LAP_umap']
            # # lap_dict['prediction'] is the lap
            # # lap_dict['action'] is the least action (similar to time) on lap
            # plot lap
            save_kwargs={"path": save_path+'/'+self.cell_type[i]+'lap_uamp'+str(self.cell_type[i][:3])+str(self.fps[i])+'_1', "prefix": 'scatter', "dpi": None, "ext": 'png', "transparent": True, "close": True, "verbose": True}
            dyn.pl.least_action(self.adata, basis="umap",color='Cell_type',save_show_or_return='save',save_kwargs=save_kwargs)
            # from dynamo.plot.scatters import save_fig, scatters
            # # ax = scatters(self.adata, basis='umap', color='ntr', save_show_or_return="return", ax=None,)
            # ax = scatters(self.adata, basis='umap', color='cell_type', save_show_or_return="return", ax=None,)
            # x=0; y=1
            # from dynamo.plot.utils import map2color
            # for i, j in zip(lap_dict["prediction"], lap_dict["action"]):
            #     ax.scatter(*i[:, [x, y]].T, c=map2color(j))
            #     ax.plot(*i[:, [x, y]].T, c="k")

            # lap in pca space
            lap = dyn.pd.least_action(
                self.adata,
                [self.adata.obs_names[start[0]][0]],
                [self.adata.obs_names[end[0]][0]],
                basis="pca",
                adj_key="cosine_transition_matrix",
                min_lap_t=min_lap_t,
                EM_steps=2,
                n_points=25
            )
            # lap_pca = adata.uns['LAP_pca']

            # plot gene expression along the LAP 
            dyn.pl.kinetic_heatmap(
                self.adata,
                basis="pca",
                mode="lap",
                genes=self.adata.var_names[self.adata.var.use_for_transition],
                project_back_to_high_dim=True,
                show_colorbar=True,
            )
            # dataframe of the gene expression dynamics over time
            kinetics_heatmap = self.adata.uns['kinetics_heatmap']
            if compared_fps:
                kinetics_heatmap.to_csv(save_path+self.cell_type[i]+'/'+str(self.fps[j])+'_kinetics_heatmap_1.csv',index=True)
            else:
                kinetics_heatmap.to_csv(save_path+self.cell_type[i]+'/kinetics_heatmap_1.csv',index=True)

            # The `GeneTrajectory` class can be used to output trajectories for any set of genes of interest
            gtraj = dyn.pd.GeneTrajectory(self.adata)
            # reverse project from PCA back to raw expression space
            # gtraj.X save the original gene expression along the LAP
            gtraj.from_pca(lap.X, t=lap.t)
            # clcalculate the mean square displacement (MSD) for each gene i along the path:
            gtraj.calc_msd()
            traj_msd = self.adata.var['traj_msd']
            traj_msd=traj_msd.sort_values(ascending=False)
            lap_genesRank[self.cell_type[i]+str(self.fps[j])]=traj_msd.index
            lap_genesRank[self.cell_type[i]+str(self.fps[j])+'_msd']=traj_msd.values
            # descending sort
            # descending sort
            ranking = dyn.vf.rank_genes(self.adata, "traj_msd")
            if len(TFs) != 0 :
                print('------------------------------------------------------------')
                print('calculate the intersection betweeen ranked genes and TF_list genes.')
                print('------------------------------------------------------------')
                merged_df = ranking.merge(TFs, on='all')
                intersection = merged_df['all'].unique()
                ranking = ranking[ranking['all'].isin(intersection)]
            ### cluster genes according to given cluster keys, e.g., `acceleration` or `velocity`.
            if cluster_genes:
                print('------------------------------------------------------------')
                print('Cluster genes start, cluster genes according to given cluster keys.')
                print('------------------------------------------------------------')

                from cluster_genes import ClusterGenes
                cluster_genes = ClusterGenes(genes_expr=gtraj.select_gene(ranking['all']),genes_rank=ranking['all'].tolist())
                for top_genes in cluster_dict['top_genes']:
                    for cluster_key in cluster_dict['cluster_key']:
                        for methods in cluster_dict['methods']:
                            corre_clustered = cluster_genes.cal_plot_clusterGenes(top_genes = top_genes, cluster_key=cluster_key,methods=methods,weigh=cluster_dict['weight'],thres=cluster_dict['thres'],thres_method=cluster_dict['thres_method'],top_quantile=cluster_dict['top_quantile'],save_fig=True,save_path=save_path+self.cell_type[i]+'/',fig_fmt='.png')
                            corre_clustered.to_csv(save_path+self.cell_type[i]+'/'+self.cell_type[i][:3]+str(self.fps[j])+'corre_clustered_top_rank_'+str(top_genes)+'_'+cluster_key[:3]+'_'+methods[:4]+'_genes_lap_1.csv')

                print('------------------------------------------------------------')
                print('Cluster genes end.')
                print('------------------------------------------------------------')

            print(start, "->", end)
            genes_list = ranking[:top_num]["all"].to_list()

            # # for gene in genes, plot gene expression on the lap
            # fig_count = 0
            # for m in range(0,len(genes_list),9):
            #     genes = genes_list[m:m+9]
            #     arr = gtraj.select_gene(genes)
            #     # (arr-arr[:,[0]]).sum(axis=1)
            #     # plot original gene expression on the lap
            #     dyn.pl.multiplot(lambda k: [plt.plot(arr[k, :]), plt.title(genes[k])], np.arange(len(genes)))
            #     fig_count += 1
            #     if compared_fps:
            #         plt.suptitle('Top '+str(m+1) +' to '+str(m+len(genes))+ ' Genes, '+self.cell_type[i]+', attractor '+str(self.fps[j]),fontsize=15)
                
            #         plt.savefig(save_path+self.cell_type[i]+'/'+self.cell_type[i][:3]+str(self.fps[j])+'_top_rank_'+str(top_num)+'_genes_lap_'+str(fig_count)+'.png')
            #     else:
            #         plt.suptitle('Top '+str(m+1) +' to '+str(m+len(genes))+ ' Genes, '+self.cell_type[i],fontsize=15)
            #         plt.savefig(save_path+self.cell_type[i]+'/'+self.cell_type[i][:3]+'_top_rank_'+str(top_num)+'_genes_lap_'+str(fig_count)+'.png')
            # plt.show()

            # calculte and plot the predicted phase transition, i.e., expresion velocity and acceleration, in the LAP.
            self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'expression',compared_fps,j,save_path,top_num=20)
            self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'velocity',compared_fps,j,save_path,top_num=20)
            self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'acceleration',compared_fps,j,save_path,top_num=20)
            if top_num>=20:
                step_fig=20
                fig_count=1
                for i_fig in range(20,top_num,step_fig):
                    fig_count+=1
                    self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'expression',compared_fps,j,save_path,top_num=i_fig+step_fig,top_start=i_fig,fig_count=fig_count)
                    self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'velocity',compared_fps,j,save_path,top_num=i_fig+step_fig,top_start=i_fig,fig_count=fig_count)
                    self.calcu_plot_phaseTrans(genes_list,gtraj,self.cell_type[i],'acceleration',compared_fps,j,save_path,top_num=i_fig+step_fig,top_start=i_fig,fig_count=fig_count)
            # # plot the selected genes on the lap
            # genes_selec=['a','b','c']
            # fig_count+=1
            # self.calcu_plot_phaseTrans(genes_selec,gtraj,self.cell_type[i],'acceleration',compared_fps,j,save_path,top_num=len(genes_selec)+1,top_start=0,fig_count=fig_count)
            
            # # calculate the accumulated displacement for each gene i along the LAP
            # acc_disp:DataFrame = pd.DataFrame(data=(gtraj.X-gtraj.X[0]).sum(axis=0),index=gtraj.genes,columns=['acc_disp']) 
            # acc_disp = acc_disp.sort_values(by='acc_disp',axis=0,ascending=False)
            # top5_pos = acc_disp.index.tolist()[:5]
            # top5_neg = acc_disp.index.tolist()[-5:]
            # arr_pos = gtraj.select_gene(top5_pos)
            # arr_neg = gtraj.select_gene(top5_neg)
            # # plot original gene expression on the lap
            # dyn.pl.multiplot(lambda k: [plt.plot(arr_pos[k, :]), plt.title(top5_pos[k])], np.arange(len(top5_pos)))
            # dyn.pl.multiplot(lambda k: [plt.plot(arr_neg[k, :]), plt.title(top5_neg[k])], np.arange(len(top5_neg)))
            # plt.show()

            # calculate some measures along the one lap, e.g., action, distance, etc.
            if compared_fps:
                lap_measures.loc[self.cell_type[i]+str(self.fps[j])] = self.calculate_measures(self.adata.uns['LAP_'+lap_measures_key])
            else:
                lap_measures.loc[self.cell_type[i]] = self.calculate_measures(self.adata.uns['LAP_'+lap_measures_key])
            # judge whether lap is in the cell type 
            print(self.judge_LAPisinCells(cell_type=[self.cell_type[i]]))


            # self.calculate_LapExprVel(genes_list,self.cell_type[i],'umap',compared_fps,j,save_path)
            # np.where(np.sum((adata.uns['VecFld_pca']['X']-lap.X[0])**2,axis=1)<1e-6)[0]
            transition_graph[self.cell_type[i]+str(i+1) + " s->e"] = {
                "lap": lap,
                "LAP_umap": self.adata.uns["LAP_umap"],
                "LAP_pca": self.adata.uns["LAP_pca"],
                "ranking": ranking,
                "gtraj": gtraj,
                # "acc_disp":acc_disp,
            }
        
        lap_measures.to_csv(save_path+'measures_'+lap_measures_key+'_1.csv',index=True)
        lap_genesRank.to_csv(save_path+'lap_genesRank_'+lap_measures_key+'_1.csv',index=True)
    
    # judge whether LAP in adata.cells
    def judge_LAPisinCells(self,basis:str='umap',cell_type:List[str]=None):
        if cell_type==None:
            X_basis:List=self.adata.obsm['X_'+basis].tolist()
        else:
            X_basis=self.adata[self.adata.obs['Cell_type'].isin(cell_type)].obsm['X_'+basis].tolist()
        pred_basis=self.adata.uns['LAP_'+basis]['prediction'][0].tolist()
        lap_flag=pred_basis[0] in X_basis
        if not lap_flag:
            print('Warning: The init point is not belong to '+str(cell_type)+'. You can recalculate the initial point.')
            return lap_flag
        else:
            i=1
            while lap_flag and i<len(pred_basis):
                lap_flag = lap_flag and pred_basis[i] in X_basis
                i+=1
            return lap_flag

    # calculate some measures along the one lap, e.g., action, distance, etc.
    def calculate_measures(self,lap_dict:Dict,) -> List[float]:
        """
        calculate some measures along the one LAP, e.g., action, distance, etc.

        Parameters
        ----------
            lap_dict:
                A dictionary saves the LAP information.
        Returns
        -------
            A dataframe saves the measures.
        """
        lap_results:List[float] = [
            lap_dict['t'][0][-1],
            lap_dict['action'][0][-1],
            np.sqrt(np.sum((lap_dict['prediction'][0][1:] - lap_dict['prediction'][0][:-1])**2,axis=1)).sum()
        ]
        return lap_results
    
    # calculte and plot the predicted phase transition, i.e. velocity and acceleration,in the LAP.
    def calcu_plot_phaseTrans(self,genes_list:List[str],gtraj:object,cell_type:str,calcu_measure:str,compared_fps:bool,j_fp:int,save_path:str,top_num:int,top_start:int=0,fig_count:int=1):
        """
        calculte and plot the predicted phase transition, i.e., expresion velocity and acceleration, in the LAP.
        Parameters
        ----------
            genes_list:
                the 'top_num' genes selected on the LAP.
            gtraj:
                the genes' trajectory object
            cell_type:
                cell type of the LAP.
            calcu_measure:
                which measure is used to calculate, only support 'expression', 'velocity' or 'acceleration'.
            compared_fps:
                whether the defined class is to calculate LAPs for different fixed points in the same cell type.
            j_fp:
                index of the fixed point in the terminal of LAP.
            save_path:
                the path to save results.
            top_num:
                the number of top genes, this parameter is used to save figure.

        Returns
        -------
           Nothing but plot expression, velocity or acceleration figure.
        """
        genes_list = genes_list[top_start:top_num]
        arr=gtraj.select_gene(genes_list)
        if calcu_measure == 'expression':
            arr_mea = arr
        elif calcu_measure == 'velocity':
            # predicted velocity
            # v(n-1) = X(n) - X(n-1)
            arr_mea = arr[:,1:]-arr[:,:-1]
        elif calcu_measure == 'acceleration':
            # predicted acceleration
            # d(n-1) = X(n) - 2X(n-1) + X(n-2)
            arr_mea = arr[:,2:]-2*arr[:,1:-1]+arr[:,:-2]
        else:
            raise ValueError('"calcu_measure" only support "expression",  "velocity" or "acceleration".')
        markers=['-o','-*','-d','-v','->']
        
        plt.figure(figsize=[6.4*2,4.8*2])
        if calcu_measure == 'expression':
            for m in range(len(arr_mea)):
                plt.plot(range(len(arr_mea[m])),arr_mea[m],markers[int(m//5)%len(markers)],label=genes_list[m])
        elif calcu_measure == 'velocity':
            for m in range(len(arr_mea)):
                plt.plot(range(1,len(arr_mea[m])+1),arr_mea[m],markers[int(m//5)],label=genes_list[m])
        elif calcu_measure == 'acceleration':
            for m in range(len(arr_mea)):
                plt.plot(range(1,len(arr_mea[m])+1),arr_mea[m],markers[int(m//5)],label=genes_list[m])
            plt.plot(range(1,len(arr_mea[m])+1),np.zeros(len(range(1,len(arr_mea[m])+1))),'--',color='gray')

        plt.xlabel('lap_index',fontsize=15)
        plt.ylabel(calcu_measure,fontsize=15)
        plt.legend(fontsize=10,loc='upper left')

        if compared_fps:
            j = j_fp
            plt.title('Top '+str(top_start)+' to '+str(top_start+len(genes_list)-1)+ ' Genes, predicted '+calcu_measure+', '+cell_type+', attractor '+str(self.fps[j]),fontsize=15)
            plt.savefig(save_path+cell_type+'/'+cell_type[:3]+str(self.fps[j])+'_top_rank_'+str(top_num)+'_genes_'+calcu_measure[:3]+'_lap_'+str(fig_count)+'.pdf')
        else:
            plt.title('Top '+str(top_start)+' to '+str(top_start+len(genes_list)-1)+ ' Genes, predicted '+calcu_measure+', '+cell_type,fontsize=15)
            plt.savefig(save_path+cell_type+'/'+cell_type[:3]+'_top_rank_'+str(top_num)+'_genes_'+calcu_measure[:3]+'_lap_'+str(fig_count)+'.pdf')
            
        plt.show()


    def calculate_LapExprVel(self,genes_list,cell_type:str,basis:str,compared_fps:bool,j_fp:int,save_path:str):
        """
        calculate and plot original gene measures on LAP, according the negibor cells.
        """
        # for i,gene in enumerate(genes_list):
        # adata_new=adata[:,adata.var['use_for_pca']]
        # ind_=nearest_neighbors(index, self.adata.obsm["X_umap"])
        from dynamo.tools.utils import nearest_neighbors
        lap_X=self.adata.uns['LAP_'+basis]['prediction'][0]
        # id_cells=np.zeros(len(lap_X))
        id_cells = []
        neighb_nums = 30
        for j in range(len(lap_X)):
            # id_cells[j] = nearest_neighbors(lap_X[j], self.adata.obsm["X_"+basis],k=neighb_nums).reshape(-1)
            id_cells += nearest_neighbors(lap_X[j], self.adata.obsm["X_"+basis],k=neighb_nums).reshape(-1).tolist()
            # id_cells+=np.where(np.sum((self.adata.uns['VecFld_pca']['X']-lap_X[j])**2,axis=1)<1e-6)[0].tolist() 
        id_cells = np.unique(id_cells).astype(int)
        assert len(id_cells)==len(lap_X)*neighb_nums, "The number of selected cells is not equal to the number of points along LAP."
        # assert all(_==cell_type for _ in self.adata[id_cells].obs['Cell_type']), "There are some other cell type in the LAP"
        adata_new = self.adata[id_cells,genes_list]
        genes_arr = np.zeros((len(lap_X),len(genes_list)))
        for j in range(0,len(id_cells),neighb_nums):
            genes_arr[int(j/neighb_nums)]=adata_new[j:j+neighb_nums][adata_new[j:j+neighb_nums].obs['Cell_type'].isin([cell_type])].X.A.mean(axis=0)

        # ### plot the top 5 genes
        # genes_arr = genes_arr[:,:5]
        # genes_list = genes_list[:5]
        step_fig = 10
        if len(genes_list)<=step_fig:
            self.plot_LapExprVel(genes_arr,genes_list,cell_type,'expression',compared_fps,j_fp,save_path,top_start=0,fig_count=1)
        else:
            fig_count = 0
            for i_fig in range(0,len(genes_list),step_fig):
                fig_count += 1
                self.plot_LapExprVel(genes_arr[:,i_fig:i_fig+step_fig],genes_list[i_fig:i_fig+step_fig],cell_type,'expression',compared_fps,j_fp,save_path,top_start=i_fig,fig_count=fig_count)


    def plot_LapExprVel(self,genes_arr,genes_list,cell_type:str,pltKey:str,compared_fps:bool,j_fp:int,save_path:str,top_start:int=0,fig_count:int=1):
        """
        plot original gene measures on LAP, according the negibor cells.
        """
        markers=['-o','-*','-d','-v','->']
        top_num = top_start+len(genes_list)
        plt.figure(figsize=[6.4*2,4.8*2])
        for m in range(genes_arr.shape[1]):
            plt.plot(range(len(genes_arr)),genes_arr[:,m],markers[int(m//5)%len(markers)],label=genes_list[m])
        plt.xlabel('lap_index',fontsize=15)
        plt.ylabel(pltKey,fontsize=15)
        plt.legend(fontsize=10,loc='upper left')
        if compared_fps:
            j = j_fp
            plt.title('Top '+str(top_start)+' to '+str(top_start+len(genes_list)-1)+ ' Genes, original '+pltKey+', '+cell_type+', attractor '+str(self.fps[j]),fontsize=15)
            plt.savefig(save_path+cell_type+'/'+cell_type[:3]+str(self.fps[j])+'_ori_top_rank_'+str(top_num)+'_genes_'+pltKey[:3]+'_lap_'+str(fig_count)+'.pdf')
        else:
            plt.title('Top '+str(top_start)+' to '+str(top_start+len(genes_list)-1)+ ' Genes, original '+pltKey+', '+cell_type,fontsize=15)
            plt.savefig(save_path+cell_type+'/'+cell_type[:3]+'_ori_top_rank_'+str(top_num)+'_genes_'+pltKey[:3]+'_lap_'+str(fig_count)+'.pdf')
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

    # ---------------------------------------------------------------------------------------------------
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
    
    # ---------------------------------------------------------------------------------------------------
    # file related
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



# ### plot one LAP in each cell type
# cell_type = ['GABAergic ACs','RGCs']
# # start_coord = np.array([[-1.35,0.45],[-1.50,2.7]])
# start_coord = np.vstack((adata.uns['VecFld_umap']['Xss'][60],adata.uns['VecFld_umap']['Xss'][60]))
# startCellType=['RPCs','RPCs']
# end_coord = np.vstack((adata.uns['VecFld_umap']['Xss'][22],adata.uns['VecFld_umap']['Xss'][24]))


# ### plot multi LAP in each cell type, to compare different attractors
# # # BCs parameters
# cell_type = ['BCs','BCs','BCs','BCs']
# start_coord = np.array([[2,0.4],[2,0.4],[2,0.4],[2,0.4]])
# attract_list=[33,37,40,3]
# startCellType=['BCs','BCs','BCs','BCs']


# # load data 
adata = dyn.read_h5ad("zebrafish_H44atoh7_1/data/H44DL1208_pres_processed_data.h5ad")
### Compute neighbor graph based on `umap`
dyn.tl.neighbors(adata, basis="umap", result_prefix="umap")
# umap_neighbors = adata.uns['umap_neighbors']
# dyn.tl.neighbors(adata, basis="umap", result_prefix="pca")


# RGCs parameters
cell_type = ['RGCs','RGCs']
start_coord = np.array([[-1.50,2.7],[-1.50,2.7]])
attract_list=[14,24]
startCellType=['RGCs','RGCs']

end_coord=[]
for att in attract_list:
    end_coord.append(adata.uns['VecFld_umap']['Xss'][att].tolist())
end_coord = np.array(end_coord)
###
lap_adata = CellLAP(adata,cell_type,start_coord,end_coord,startCellType,attract_list)
lap_adata.plot_start_end_cells()
### plot multi LAP in each cell type, to compare different attractors
save_path='zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/lap_with_lineage/diff_fp/'
TF_list: DataFrame = pd.read_csv("zebrafish_H44atoh7_1/TF_Iist_daniorerio.txt", sep=" ",names=['all'])
# if no TF list, then set 'TFs=[]' in the below function 'lap_adata.calculate_transit_graph()', default value of 'TFs' is [].
# if not cluster genes, just set cluster_genes=Flase. you can ignore the set of cluster_dict, because it's default is {}.
# `methods` can be `spearman` or `pearson`. For the large size data, `spearman` is faster than `pearson`.
# auto-generated threshold to judge abs(acc)=0
cluster_dict = {
    'top_genes': [60], 
    'cluster_key': ['velocity'],
    'methods': ['spearman'],
    'weight': 0.5,
    'thres': 5e-3,
    'thres_method':'auto',
    'top_quantile': 0.5,}

# input threshold to judge abs(acc)=0
# cluster_dict = {
#     'top_genes': [60], 
#     'cluster_key': ['velocity'],
#     'methods': ['spearman'],
#     'weight': 0.5,
#     'thres': 5e-3,
#     'thres_method':'input',
#     'top_quantile': 0.5,}

lap_adata.calculate_transit_graph(top_num=60,save_path=save_path,compared_fps=True,TFs=[],cluster_genes=True,cluster_dict=cluster_dict)
print()
