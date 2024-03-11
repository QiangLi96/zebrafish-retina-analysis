import warnings

warnings.filterwarnings('ignore')

import dynamo as dyn
import xlrd
import numpy as np
import matplotlib.pyplot as plt

dyn.configuration.set_figure_params('dynamo', background='white')
#dyn.__version__
adata = dyn.read("zebrafish_GABAergic/data/GABAergic_H44_48_60_72_D5_D16_M15_M3.h5ad")
# plt.plot(np.array(adata.obs['umap_1']),np.array(adata.obs['umap_2']),'.', markersize=20, alpha=0.6)

umap1 = adata.obs['umap_1']
umap2 = adata.obs['umap_2']


adata.layers['spliced'] = adata.layers['spliced'].astype('int64')
adata.layers['unspliced'] = adata.layers['unspliced'].astype('int64')

adata.X = adata.layers['spliced']
adata.X = adata.X.astype('float32')


# adata.obs['timebatch'] = adata.obs['timebatch'].astype('str')
# adata.obs['timebatch'] = adata.obs['timebatch'].astype('category')


del adata.layers['ambiguous']
del adata.layers['matrix']

adata.obs['Cell_type'] = (adata.obs['Clusters'].copy()).astype(str)
# adata.obsm['X_umap'] = np.vstack((umap1,umap2)).T
dyn.pp.recipe_monocle(adata)

dyn.tl.dynamics(adata, model='stochastic', cores=3)
# or dyn.tl.dynamics(adata, model='deterministic')
# or dyn.tl.dynamics(adata, model='stochastic', est_method='negbin')

dyn.tl.reduceDimension(adata)
# adata.obsm['X_umap'][:, 0] = adata.obs['umap_1']
# adata.obsm['X_umap'][:, 1] = adata.obs['umap_2']


dyn.pl.umap(adata, color='Cell_type', show_legend='on data')

dyn.tl.cell_velocities(adata, method='pearson', other_kernels_dict={'transform': 'sqrt'})
dyn.pl.cell_wise_vectors(adata, color=['Cell_type'], basis='umap', show_legend='on data', quiver_length=6,
        quiver_size=6, pointsize=0.1, show_arrowed_spines=False)

dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='umap', show_legend='on data',
            show_arrowed_spines=True)
# you can set `verbose = 1/2/3` to obtain different levels of running information of vector field reconstruction
dyn.vf.VectorField(adata, basis='umap', M=1000, pot_curl_div=True)
dyn.pl.topography(adata, basis='umap', background='white', color=['ntr', 'Cell_type'],streamline_color='black',
                  show_legend='on data', frontier=True)


dyn.pl.cell_wise_vectors(adata, color=['Cell_type'], basis='pca', show_legend='on data', quiver_length=6,
        quiver_size=6, pointsize=0.1, show_arrowed_spines=False)
dyn.pl.streamline_plot(adata, color=['Cell_type'], basis='pca', show_legend='on data',
            show_arrowed_spines=True)
# you can set `verbose = 1/2/3` to obtain different levels of running information of vector field reconstruction
dyn.vf.VectorField(adata, basis='pca', M=1000, pot_curl_div=True,method='SparseVFC')
dyn.pl.topography(adata, basis='pca',background='white', color=['ntr', 'Cell_type'],streamline_color='black', show_legend='on data', frontier=True)

print()


