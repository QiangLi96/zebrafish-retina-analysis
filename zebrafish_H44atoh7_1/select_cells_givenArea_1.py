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

class CellSelect():
    def __init__(self,adata) -> None:
        self.adata = adata
        pass
    
    def plot_selectCells(self,cell_types:list=[],cell_range:list=None,show_fig:bool=True,save_fig:bool=True,save_path:str='',fig_name:str='',fig_ext:str='.png'):
        """
        plot the selected cells in a grid.

        Parameters
        ----------
        cell_types: `list`
            cell types used to select cells.
        cell_range: `list[list[]]`
            boundaries of the rectangle (x_left,x_right,y_up,y_down), to select cells within the specified range of the rectangle.
        save_fig: `bool`, default is `True`
                Whether save figure.
        show_fig: `bool`, default is `True`
            Whether show figure.
        save_path: `str`
            path used to save figure.
        fig_name: `str`, default `''`
            To make the order of the plotted figure same as the ranked genes, we offer the user can set the file name of the figure.
            if fig_name is not inputted, the default file name is 'similarity_'+ref_gene+'_'+test_gene[i].
        fig_ext: `str`, default is `.png`  
            figure format.

        Returns
        ----------   
            Nothing but plot a figure.

        """

        color_list = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']

        plt.figure()
        if len(cell_types)==0:
            cell_types = list(np.unique(self.adata.obs['Cell_type']))
        self.adata_new = self.adata[self.adata.obs['Cell_type'].isin(cell_types)]
        if cell_range!=None:
            self.cells_rectangle(cell_range=cell_range)
        for i,cell_type in enumerate(cell_types):
            X_umap = self.adata_new[self.adata_new.obs['Cell_type'].isin([cell_type])].obsm['X_umap']
            plt.plot(X_umap[:,0],X_umap[:,1],'.',color=color_list[i],markersize=2,alpha=0.6)
        plt.grid(alpha=0.4)

        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(r'$\mathrm{umap}_1$',fontsize=15)
        plt.ylabel(r'$\mathrm{umap}_2$',fontsize=15)
        # plt.legend(fontsize=15)

        if save_fig:
            plt.savefig(save_path+fig_name+fig_ext)

        if show_fig:
            plt.show()

    def cells_rectangle(self,cell_range):
        """
        select cells in some rectangles.

        Parameters
        ----------
        cell_range: `list[list[]]`
            boundaries of the rectangle (x_left,x_right,y_up,y_down), to select cells within the specified range of the rectangle.

        Returns
        ----------   
            Nothing.
        """
        X_umap = self.adata_new.obsm['X_umap']
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

        self.adata_new = self.adata_new[bool_xy]
        







# # ----------------------------------------------------------------------------------------------
# ###
# # load data 
adata = dyn.read_h5ad("zebrafish_H44atoh7_1/data/H44DL1208_pres_processed_data.h5ad")

# dyn.pl.umap(adata, color='Cell_type', show_legend='on data')

select_cell = CellSelect(adata)

cell_types = ['pre1','PRpre','RGCs']
# cell_range: [x_left,x_right, y_down, y_up]
cell_range=[[-6,-4,2.5,5],[-2,0,0,5],[-4,-2,7.5,10]]

save_fig = False
save_path = 'F:/python_work_PyCharm/work_7_20220317/zebrafish_H44atoh7_1/Figures_H44_DL_1208_pres/umap_adata/genes_perturb/'+'atoh7'+'/'
fig_name = 'select_cells_1'
# default cell_range is 'None', length of cell_range is arbitrary, not same as cell_types
select_cell.plot_selectCells(cell_types=cell_types,cell_range=cell_range,show_fig=True,save_fig=save_fig,save_path=save_path,fig_name=fig_name,fig_ext='.png')




# X_umap = self.adata.obsm['X_umap']
# X_umap = X_umap[cell_loc]

# plt.plot(X_umap[:,0],X_umap[:,1],'.',color='b',markersize=2,alpha=0.4)


print()