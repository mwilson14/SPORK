import pyart
import numpy as np
import numpy.ma as ma

def gridding(radar,Z0C):
    print('Grid Section')
    grid = pyart.map.grid_from_radars(
    	(radar,),
    	grid_shape=(41, 500, 500),
    	grid_limits=((0, 10000), (-123000.0, 123000.0), (-123000.0, 123000.0)),
    	fields=['differential_reflectivity','reflectivity','KDP','cross_correlation_ratio'],
    	weighting_function='Barnes')
    Zint = int(np.round((Z0C+1000)/250))
    ZDRall = grid.fields['differential_reflectivity']['data']
    REFall = grid.fields['reflectivity']['data']
    CCall = grid.fields['cross_correlation_ratio']['data']
    KDPall = grid.fields['KDP']['data']
    ZDR = grid.fields['differential_reflectivity']['data'][4,:,:]
    REF = grid.fields['reflectivity']['data'][4,:,:]
    KDP = grid.fields['KDP']['data'][4,:,:]
    CC = grid.fields['cross_correlation_ratio']['data'][4,:,:]
    CC_c = grid.fields['cross_correlation_ratio']['data'][Zint,:,:]
    
    ZDRmasked1 = ma.masked_where(REF < 20, ZDR)
    ZDRrmasked1 = ma.masked_where(REFall < 20, ZDRall)
    REFmasked = ma.masked_where(REF < 20, REF)
    REFrmasked = ma.masked_where(REFall[Zint,:,:] < 20, REFall[Zint,:,:])
    #Use a 50 dBZ mask for KDP to only get areas in the storm core. This threshold should be considered more closely
    KDPmasked = ma.masked_where(REF < 50, KDP)
    KDPmasked = ma.filled(KDPmasked, fill_value = -2)
    KDPrmasked = KDPall[Zint,:,:]

    rlons = grid.point_longitude['data']
    rlats = grid.point_latitude['data']
    rlons_2d = rlons[0,:,:]
    rlats_2d = rlats[0,:,:]
    cenlat = radar.latitude['data'][0]
    cenlon = radar.longitude['data'][0]

    return Zint,REF,KDP,CC,CC_c,CCall,ZDRmasked1,ZDRrmasked1,REFmasked,REFrmasked,KDPmasked,KDPrmasked,rlons,rlats,rlons_2d,rlats_2d,cenlat,cenlon
