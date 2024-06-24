import sys
sys.path.insert(0,'/workspace/WorldPeatland/code/')
from gdal_sheep import *

import rasterio
import xarray as xr
from glob import glob
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature



def run(site, product):

    # these are the QA analytics calculated
    path = f'/wp_data/sites/{site}/MODIS/analytics/*{product}*pct_data_available.tif'
    l = glob(path)

    tiff_file = l[0]

    # Open the TIFF file using rasterio
    with rasterio.open(tiff_file) as src:
        # Read the data
        data = src.read()

        transform = src.transform
        height, width = src.height, src.width # height = lat,y ; width = lon,x

        x = np.arange(width) * transform[0] + transform[2]
        y = np.arange(height) * transform[4] + transform[5]
        y = y[::-1]  # Reverse y for correct orientation

        coords = {
            "y": y,
            "x": x
        }
        attrs = src.meta.copy()

        data_vars = {f'pct_data_available': (("y", "x"), data[i]) for i in range(data.shape[0])}
        ds_pct = xr.Dataset(data_vars, coords=coords, attrs=attrs)

    bd_pct = ds_pct['pct_data_available']


    path = f'/wp_data/sites/{site}/MODIS/analytics/*{product}*_max_gap_length.tif'
    l = glob(path)


    # max gap length
    tiff_file = l[0]


    # Open the TIFF file using rasterio
    with rasterio.open(tiff_file) as src:
        # Read the data
        data = src.read()

        transform = src.transform
        height, width = src.height, src.width # height = lat,y ; width = lon,x

        x = np.arange(width) * transform[0] + transform[2]
        y = np.arange(height) * transform[4] + transform[5]
        y = y[::-1]  # Reverse y for correct orientation

        coords = {
            "y": y,
            "x": x
        }
        attrs = src.meta.copy()

        data_vars = {f'max_gap_length': (("y", "x"), data[i]) for i in range(data.shape[0])}
        ds_max = xr.Dataset(data_vars, coords=coords, attrs=attrs)


    bd_max = ds_max['max_gap_length']

    # plot settings:

    cmap = 'viridis'
    dpi = 250

    '''

    Similar to the TATSSI UI QA visualization 

    https://github.com/GerardoLopez/TATSSI/blob/122074a1e3015e8ade99dfb08d5a3b712b276bda/TATSSI/UI/plots_qa_analytics.py#L498

    '''

    globe=ccrs.Globe(ellipse=None, 
                     semimajor_axis=6371007.181, 
                     semiminor_axis=6371007.181)

    proj = ccrs.Sinusoidal(globe=globe)


    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 6.5),
                    #sharex=True, sharey=True, tight_layout=True, dpi=dpi,
                    sharex=True, sharey=True, dpi=dpi,
                    subplot_kw=dict(projection=proj))

    for _axis in [ax, bx]:
        _axis.coastlines(resolution='10m', color='white')
        _axis.add_feature(cfeature.BORDERS, edgecolor='white')
        _axis.gridlines()


    # plot % data available
    bd_pct.plot.imshow(
        ax=ax, cmap=cmap,
        cbar_kwargs={'orientation':'horizontal',
                     'pad' : 0.005},
    transform=proj)
    
    ax.set_frame_on(False)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.title.set_text('% of data available')


    # plot max gap length
    bd_max.plot.imshow(
            ax=bx, cmap=cmap,
            cbar_kwargs={'orientation':'horizontal',
                         'pad' : 0.005},
            transform=proj
    )

    bx.set_frame_on(False)
    bx.axis('off')
    bx.set_aspect('equal')
    bx.title.set_text('Max gap-length')

    fig.suptitle(f'{site}: {product}')
    fig.canvas.draw()
    plt.tight_layout()

    fig.savefig(f'/workspace/WorldPeatland/visualisation/{site}_{product}_QA_analytics.png', dpi=fig.dpi)
    
    print('Image saved here:', f'/workspace/WorldPeatland/visualisation/{site}_{product}_QA_analytics.png')

    
def main(site):
    
    product_list = ['Lai_500m','Albedo_WSA_Band2','ET_500m','EVI','Fpar_500m','LST_Night_1km', 'LST_Day_1km']
    
    for product in product_list:
        
        run(site, product)

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print("Usage: python script.py <site>") # the user has to input one argument
    else:
        site = sys.argv[1]
        main(site)
