import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dynamo as dyn
import seaborn as sns
import anndata as Anndata
# type set
from typing import Dict, List, Any, Tuple
from pandas import DataFrame

# dyn.dynamo_logger.main_silence()

# # filter warnings for cleaner tutorials
# import warnings

# warnings.filterwarnings('ignore')

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

# # dyn.tl.cell_velocities(adata, method='pearson', other_kernels_dict={'transform': 'sqrt'})
# dyn.tl.cell_velocities(adata, method='cosine', other_kernels_dict={'transform': 'sqrt'})
# # dyn.tl.cell_velocities(adata, method='fp', other_kernels_dict={'transform': 'sqrt'})

# dyn.pl.cell_wise_vectors(adata, color=['Cell_type'], basis='umap', show_legend='on data', quiver_length=6,quiver_size=6, pointsize=0.1, show_arrowed_spines=False)

# dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='umap', show_legend='on data',
#             show_arrowed_spines=True)


# plot input output patten
# -----------------------------------------------------------------------------------------------------------------------
# *******************************************************************************

class ShowMethods:
    def __init__(self,df_tmp:DataFrame) -> None:
        self.df = df_tmp
        self.color_list = ["#00ff00","#aba900","#b83dba","#fff200","#FF6103","#35e3ee","#585858","#bf1010","#072cb0","#f2be1e","#02a229","#f37736","#8A360F",]
        self.dict_color = self.__match_cell_color()

    def __match_cell_color(self)->Dict:
        """
        define a dictionary to used color, of which keys are cell type and values are color
        """
        dict_color:Dict = {}
        for i,ct in enumerate(self.df['Cell_type'].unique()):
            dict_color[ct] = self.color_list[i]
        return dict_color
    
    def __get_df_show(self,cell_type:List[str]=None,key:str='cosine')->DataFrame:
        """
        sort self.df for the given cell_type relying on key.

        Parameters
        ----------
        cell_type: the cell_type used to sort 
        key: the sorted key

        Returns
        -------
        the sorted DataFrame, which is part of self.df  
        """

        df_ = self.df[self.df['Cell_type'].isin(cell_type)]
        df_ = df_.sort_values(by=[key],axis=0)
        return df_
    
    def show_scatter(self,cell_type:List[str]=None,key_sort:str='cosine',num:int=10,key_show:str='cosine',save:bool=False,save_path:str=None,save_key:str='png',background:str='each')->None:
        """
        show the fisrt num pseudotime cells on background cell type umap.

        Parameters
        ----------
        cell_type: the cell type wanted to show pseudotime.
        key_sort: the sorted key in self.df_
        num: show the fisrt num cells 
        key_show: the shown key in self.df_
        background: the background of umap, deciding the cell_type to use. Only support values are 'each' or 'all', the deafult is 'each'.

        Returns
        -------
        Nothing but the num-th 'key_show' cells on background umap.

        """
        self.df_ = self.__get_df_show(cell_type,key_sort)
        plt.figure(figsize=[8,6])
        if background=='all':
            plt.scatter(self.df['umap_0'],self.df['umap_1'],s=250,c='lightsteelblue',marker='o',alpha=0.2)
        elif background=='each':
            plt.scatter(self.df_['umap_0'],self.df_['umap_1'],s=250,c='lightsteelblue',marker='o',alpha=0.2)
        else:
            raise ValueError("background only support 'each' or 'all'.")
        fig_scatter = plt.scatter(self.df_['umap_0'].iloc[:num],self.df_['umap_1'].iloc[:num],c=self.df_[key_show].iloc[:num],s=250,marker='o',cmap='inferno',alpha=0.8)
        plt.title(f'{key_show} of {cell_type[0]}',fontsize=15)
        cb = plt.colorbar(fig_scatter)
        # set tick fontsize of colorbar
        cb.ax.tick_params(labelsize=12) 
        font = {'family': 'serif',
                'color' : 'black',
                'weight': 'normal',
                'size'  : 15,
                }
        cb.set_label(key_show.split('_')[1],fontdict=font)
        plt.xlabel("UMAP0",fontsize=15)
        plt.ylabel("UMAP1",fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if save and save_path:
            plt.savefig(f'{save_path}{key_show}_{cell_type[0]}.{save_key}')
        elif save:
            plt.savefig(f'{key_show}_{cell_type[0]}.{save_key}')
        plt.show()

    def plot_weigt_mat(self,obs:int=0,i_th:int=0,save:bool=False,save_path:str=None,save_key:str='png') -> None:
        
        """
        plot cosine similarities of the obs-th cell, i.e., obs-th row and obs-th column in weight matrix

        Parameters
        ----------
        obs: the serial number of the ploted cell.
        i_th: the sorted number of pseudotime for the ploted cell

        Returns
        -------
        Nothing but some points plot.
        """

        plt.figure(figsize=[14,6])
        plt.subplot(1,2,1)       
        for j,ct in enumerate(self.Cell_type):
            if j<= len(self.Cell_type)//2+1 or ct=='pre2':
                marker = 'o'
            else:
                marker = 'v'
            plt.plot(self.df[self.df['Cell_type']==ct]['obs_row'].index,self.df[self.df['Cell_type']==ct]['obs_row'].values,marker,color=self.dict_color[ct],alpha=0.8,label=ct)
        plt.title(f'row weight',fontsize=15)
        # set the y-axis tick size to 0.01
        y_major_locator = plt.MultipleLocator(0.01)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        plt.xlabel("cell index",fontsize=15)
        plt.ylabel("weight",fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        y_scale = 0.1
        plt.ylim(df['obs_row'].min()*(1+y_scale)-df['obs_row'].max()*y_scale,df['obs_row'].max()*(1+y_scale)-df['obs_row'].min()*y_scale)

        plt.subplot(1,2,2)
        for j,ct in enumerate(self.Cell_type):
            if j<= len(self.Cell_type)//2+1 or ct=='pre2':
                marker = 'o'
            else:
                marker = 'v'
            plt.plot(self.df[self.df['Cell_type']==ct]['obs_column'].index,self.df[self.df['Cell_type']==ct]['obs_column'].values,marker,color=self.dict_color[ct],alpha=0.8,label=ct)
        plt.title(f'column weight',fontsize=15)
        # set the y-axis tick size to 0.01
        y_major_locator = plt.MultipleLocator(0.01)
        ax = plt.gca()
        ax.yaxis.set_major_locator(y_major_locator)
        # plt.legend(fontsize=12)
        plt.legend(fontsize=12,bbox_to_anchor=(1.0,1))
        plt.xlabel("cell index",fontsize=15)
        plt.ylabel("weight",fontsize=15)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        y_scale = 0.1
        plt.ylim(df['obs_column'].min()*(1+y_scale)-df['obs_column'].max()*y_scale,df['obs_column'].max()*(1+y_scale)-df['obs_column'].min()*y_scale)
        plt.subplots_adjust(left=0.086,right=0.86)

        plt.suptitle(f"{self.df.loc[obs,['Cell_type']].values.item()}, {i_th}-th pseudotime",fontsize=15)
        if save and save_path:
            plt.savefig(f'{save_path}weight_mat_plot_{obs}.{save_key}')
        elif save:
            plt.savefig(f'weight_mat_plot_{obs}.{save_key}')
        plt.show()

    def show_velocity_quiver(self,cell_intere_:str,pseudo_num:int=0,row_or_column:str='row',sort_i:int=0,save:bool=False,save_path:str=None,save_key:str='png'):
        """
        show quivers of velocity v_i and \delta_{ij} calculated in weight matrix.

        to better show the meaning of cosine similarities used in 'Cell' article.
        'Cell' article: Mapping transcriptomic vector fields of single cells
        
        Parameters
        ------------
        cell_intere_: the cell type in pseudo_num-th wight array
        pseudo_num: the sorted number of pseudotime cell
        row_or_column: plot 'row weight' or 'column weight' in weight matrix figure
        sort_i: the sorted number of weight of cell_intere_

        Returns
        -------
        Nothing but a quiver plot.
        """

        pseudo_name = self.df_.iloc[pseudo_num].name
        pseudo_cell = self.df_.iloc[pseudo_num]['Cell_type']
        if sort_i==0:
            id_row = self.df[self.df['Cell_type']==cell_intere_]['obs_row'].idxmax()
            id_col = self.df[self.df['Cell_type']==cell_intere_]['obs_column'].idxmin()
        elif sort_i>0:
            id_row = self.df[self.df['Cell_type']==cell_intere_]['obs_row'].nlargest(sort_i+1).index[-1]
            id_col = self.df[self.df['Cell_type']==cell_intere_]['obs_column'].nsmallest(sort_i+1).index[-1]
        if row_or_column=='row':
            plt.figure(figsize=[12,8])

            plt.quiver(self.df['umap_0'], self.df['umap_1'], self.df['vel_umap_0'],self.df['vel_umap_1'],color='lightsteelblue',units='xy',angles='xy',minlength=0,width=0.5,
            headwidth=3,headlength=6,scale=2,alpha=1.0,linewidth=0.3,edgecolor='black')    

            plt.quiver(self.df.loc[pseudo_name]['umap_0'], self.df.loc[pseudo_name]['umap_1'], self.df.loc[pseudo_name]['vel_umap_0'],self.df.loc[pseudo_name]['vel_umap_1'],color=self.dict_color[pseudo_cell],units='xy',angles='xy',minlength=0)   
            # # plot the j-th cell velocity quiver
            # plt.quiver(self.df.loc[id_row]['umap_0'], self.df.loc[id_row]['umap_1'], self.df.loc[id_row]['vel_umap_0'],self.df.loc[id_row]['vel_umap_1'],color=self.dict_color[cell_intere_],units='xy',minlength=0,alpha=0.8,label='wight_mat cell')
            plt.quiver(self.df.loc[pseudo_name]['umap_0'], self.df.loc[pseudo_name]['umap_1'],self.df.loc[id_row]['umap_0']-self.df.loc[pseudo_name]['umap_0'],self.df.loc[id_row]['umap_1']- self.df.loc[pseudo_name]['umap_1'],color='black',units='xy',angles='xy',minlength=0,alpha=0.8,label='$\delta_{ij}$')

            plt.plot(self.df.loc[pseudo_name]['umap_0'], self.df.loc[pseudo_name]['umap_1'],marker='o',markersize=10,color=self.dict_color[pseudo_cell],alpha=0.8,label='psedotime cell')
            plt.plot(self.df.loc[id_row]['umap_0'],self.df.loc[id_row]['umap_1'],marker='o',markersize=10,color=self.dict_color[cell_intere_],alpha=0.8,label='wight_mat cell')

            plt.plot([self.df.loc[pseudo_name]['umap_0'],self.df.loc[id_row]['umap_0']],[self.df.loc[pseudo_name]['umap_1'],self.df.loc[id_row]['umap_1']],'--',color='k',alpha=0.2)

            plt.legend(fontsize=12)
            plt.title(f"row weight,{pseudo_cell}, {pseudo_num}-th pseudotime, {cell_intere_}, {sort_i}-th largest weight",fontsize=15)
            plt.xlabel("UMAP0",fontsize=15)
            plt.ylabel("UMAP1",fontsize=15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)

            # xylim_max = self.df.iloc[[pseudo_name,id_row,id_col]][['umap_0','umap_1']].max().max()
            # xylim_min = self.df.iloc[[pseudo_name,id_row,id_col]][['umap_0','umap_1']].min().min()
            # plt.xlim(xylim_min*1.1-xylim_max*0.1,xylim_max*1.1+xylim_min*0.1)
            # plt.ylim(xylim_min*1.1-xylim_max*0.1,xylim_max*1.1+xylim_min*0.1)

            if save and save_path:
                plt.savefig(f'{save_path}veldelta_quiver_{cell_intere_}_{row_or_column}_{pseudo_num}.{save_key}')
            elif save:
                plt.savefig(f'veldelta_quiver_{cell_intere_}_{row_or_column}_{pseudo_num}.{save_key}')
            plt.show()

        elif row_or_column=='column':
            plt.figure(figsize=[12,8])
            plt.quiver(self.df['umap_0'], self.df['umap_1'], self.df['vel_umap_0'],self.df['vel_umap_1'],color='lightsteelblue',units='xy',angles='xy',minlength=0,width=0.5,
            headwidth=3,headlength=6,scale=2,alpha=1.0,linewidth=0.3,edgecolor='black')   

            # # plot the j-th cell velocity quiver
            # plt.quiver(self.df.loc[pseudo_name]['umap_0'], self.df.loc[pseudo_name]['umap_1'], self.df.loc[pseudo_name]['vel_umap_0'],self.df.loc[pseudo_name]['vel_umap_1'],color=self.dict_color[pseudo_cell],units='xy',angles='xy',minlength=0,label='psedotime cell')   
            plt.quiver(self.df.loc[id_col]['umap_0'], self.df.loc[id_col]['umap_1'], self.df.loc[id_col]['vel_umap_0'],self.df.loc[id_col]['vel_umap_1'],color=self.dict_color[cell_intere_],units='xy',angles='xy',minlength=0,alpha=0.8)

            plt.quiver(self.df.loc[id_col]['umap_0'], self.df.loc[id_col]['umap_1'],self.df.loc[pseudo_name]['umap_0']-self.df.loc[id_col]['umap_0'],self.df.loc[pseudo_name]['umap_1']-self.df.loc[id_col]['umap_1'],color='black',units='xy',angles='xy',minlength=0,alpha=0.8,label='$\delta_{ij}$')

            plt.plot(self.df.loc[pseudo_name]['umap_0'], self.df.loc[pseudo_name]['umap_1'],marker='o',markersize=10,color=self.dict_color[pseudo_cell],alpha=0.8,label='psedotime cell')
            plt.plot(self.df.loc[id_col]['umap_0'],self.df.loc[id_col]['umap_1'],marker='o',markersize=10,color=self.dict_color[cell_intere_],alpha=0.8,label='wight_mat cell')

            plt.plot([self.df.loc[pseudo_name]['umap_0'],self.df.loc[id_col]['umap_0']],[self.df.loc[pseudo_name]['umap_1'],self.df.loc[id_col]['umap_1']],'--',color='k',alpha=0.2)

            plt.legend(fontsize=12)
            plt.title(f"column weight,{pseudo_cell}, {pseudo_num}-th pseudotime, {cell_intere_}, {sort_i}-th smallest weight",fontsize=15)
            plt.xlabel("UMAP0",fontsize=15)
            plt.ylabel("UMAP1",fontsize=15)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            # plt.xlim(xylim_min*1.1-xylim_max*0.1,xylim_max*1.1+xylim_min*0.1)
            # plt.ylim(xylim_min*1.1-xylim_max*0.1,xylim_max*1.1+xylim_min*0.1)

            if save and save_path:
                plt.savefig(f'{save_path}veldelta_quiver_{cell_intere_}_{row_or_column}_{pseudo_num}.{save_key}')
            elif save:
                plt.savefig(f'veldelta_quiver_{cell_intere_}_{row_or_column}_{pseudo_num}.{save_key}')
            plt.show()

            plt.show()



class ShowInOutPutPattern(ShowMethods):
    def __init__(self,df_tmp:DataFrame) -> None:
        super().__init__(df_tmp)
        self.Cell_type = df_tmp['Cell_type'].unique()
    
    def __get_df_show(self,cell_type:List[str]=None,key:str='cosine',num:int=10)->DataFrame:
        df_ = self.df[self.df['Cell_type'].isin(cell_type)]
        df_ = df_.sort_values(by=[key],axis=0)
        return df_.iloc[:num]

    def show_div(self,cell_type:List[str]=None,key_show:str='cosine_div',save:bool=False,save_path:str=None,save_key:str='png') -> None:
        """
        plot divergence for given cell type.

        Parameters
        ----------
        cell_type: the cell type wanted to show divergence.
        key_show: the shown key in self.df_

        Returns
        -------
        Nothing but a divergence ploted on umap.

        """
        self.show_scatter(cell_type=cell_type,key_show=key_show,save=save,save_path=save_path,save_key=save_key)
    
    def show_weight(self,weight_mat:np.matrix=None,cell_type:List[str]=None,key_sort:str='cosine',num:int=10,key_show:str='cosine',save:bool=False,save_path:str=None,save_key:str='png') -> None:
        """
        plot weight matrix for given cell type.

        Parameters
        ----------
        weight_mat: the weight matrix with size of (n_cells, n_cells)
        cell_type: the cell type wanted to show its row and column weights.
        key_show: the shown key in self.df_
        key_sort: the sorted key in self.df_
        num: show the fisrt num cells, which the sort rely on 'key_sort'
        key_show: the shown key in self.df_

        Returns
        -------
        Nothing but ploting the num figures shown the corresponding row and column weights.
        """
        
        self.df_ = self.__get_df_show(cell_type,key_sort,num)
        for i,obs in enumerate(self.df_.index):
            self.df['obs_row'] = weight_mat[obs].A.reshape(-1)
            self.df['obs_column'] = weight_mat[:,obs].A.reshape(-1)
            self.plot_weigt_mat(obs=obs,i_th=i,save=save,save_path=save_path)

    def show_veldelta_quiver(self,weight_mat:np.matrix=None,cell_type:List[str]=None,key_sort:str='cosine',sort_num:int=10,pseudo_num:int=0,cell_interes:List[str]=None,row_or_column:str='row',sort_i:List[int]=[0],save:bool=False,save_path:str=None,save_key:str='pdf'):
        """
        show quivers of velocity v_i and \delta_{ij} calculated in weight matrix.

        to better show the meaning of cosine similarities used in 'Cell' article.
        'Cell' article: Mapping transcriptomic vector fields of single cells
        
        Parameters
        ------------
        weight_mat: the weight matrix with size of (n_cells, n_cells)
        cell_type: the pseudotime cell type wanted to show.
        key_sort: the sorted key in self.df_
        sort_num: take the first 'sort_num' number in self.df_
        pseudo_num: the sorted number of pseudotime cell
        cell_interes: the intersted cell types in pseudo_num-th wight array
        row_or_column: plot the corresponding 'row weight' or 'column weight' in weight matrix figure
        sort_i: the sorted number of weight for cells in cell_interes, with same size as 'cell_interes'

        Returns
        -------
        Nothing but a quiver plot.
        """

        self.df_ = self.__get_df_show(cell_type,key_sort,sort_num)
        pesudo_name = self.df_.iloc[pseudo_num].name
        self.df['obs_row'] = weight_mat[pesudo_name].A.reshape(-1)
        self.df['obs_column'] = weight_mat[:,pesudo_name].A.reshape(-1)

        if len(cell_interes) != len(sort_i):
            raise ValueError("list 'cell_interes' and list 'sort_i' must have the same length.")
        else:
            for j in range(len(cell_interes)):
                self.show_velocity_quiver(cell_intere_=cell_interes[j],pseudo_num=pseudo_num,row_or_column=row_or_column,sort_i=sort_i[j],save=save,save_path=save_path,save_key=save_key)


adata=dyn.read("zebrafish_H44atoh7_1/data/H44DL1208_pres_linux_processed_data.h5ad")

df: DataFrame = pd.read_csv('zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/H44DL1208_adataNew_pres_potential_cos_fp_1.csv')
# df = df.iloc[:,3:]

# ## plot velocit arrow on umap for selcted cells
# x_umap = adata.obsm['X_umap']
# velocity_umap = adata.obsm['velocity_umap']
# np.save('zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/H44_DL_1208_pres_vel_umap_adata_1.npy',velocity_umap)
# df['vel_umap_0'] = velocity_umap[:,0]
# df['vel_umap_1'] = velocity_umap[:,1]
# # umap adata
# df['vel_umap_0'] = velocity_umap[:,0]
# df['vel_umap_1'] = velocity_umap[:,1]
# df['umap_0'] = adata.obs['umap_1'].values
# df['umap_1'] = adata.obs['umap_2'].values
# df.to_csv('zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/H44_DL_1208_pres_potential_cos_fp_1.csv',index=False)


inoutput_pattern1 = ShowInOutPutPattern(df)

### plot divgergence scatter on umap
# ## umap_auto
# save_path = 'zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_auto/in_output_pattern/div/'
## umap_adata
save_path = 'zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/div/'
# df['umap_0'] = adata.obs['umap_1'].values
# df['umap_1'] = adata.obs['umap_2'].values

# cell_type=['RGCs']
for cell_type in df['Cell_type'].unique():
    inoutput_pattern1.show_div(cell_type=[cell_type],key_show='cosine_div',save=True,save_path=save_path)


### plot weight_mat on umap
# cosine_transition_matrix = adata.obsp['cosine_transition_matrix']
# np.save('zebrafish_H44atoh7_1\Figures_H44_DL_1208_pres\cosine_transition_matrix.npy',cosine_transition_matrix)
cosine_transition_matrix = np.load('zebrafish_H44atoh7_1\Figures_H44_DL_1208_pres\cosine_transition_matrix.npy', allow_pickle=True).item()

cell_intere = ['RGCs','GABAergic ACs']
for cell_type in cell_intere:
    save_path = 'zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_auto/in_output_pattern/weight_cosine/'+cell_type+'/'
    inoutput_pattern1.show_weight(weight_mat=cosine_transition_matrix,cell_type=[cell_type],save=True,save_path=save_path)

### plot velocity and delta_{ij} quiver on umap

# pseudo_nums = 2
# cell_pseudo = ['RGCs']
# sort_is = [0,0]
# for cell_type in cell_pseudo:
#     save_path = 'zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/in_output_pattern/vel_delta_quiver/'+cell_type+'/'
#     inoutput_pattern1.show_veldelta_quiver(weight_mat=cosine_transition_matrix,cell_type=[cell_type],pseudo_num=2,cell_interes=['RGCs','Glycinergic ACs'],row_or_column='row',sort_i=[0,0],save=False,save_path=save_path)

print()

# df['CellID'].values
# np.array(adata.obs_names.tolist(),dtype='object')
# (df['CellID'].values==np.array(adata.obs_names.tolist(),dtype='object')).sum()




# -----------------------------------------------------------------------------------------------------------------------
