import numpy as np
import numpy.ma as ma
import pyart

def quality_control(radar1,n,calibration):
    print('Pre-grid Organization Section')
    ni = 0
    tilts = []
    for i in range(radar1.nsweeps):
        radar2 = radar1.extract_sweeps([i])
        #Checking to make sure the tilt in question has all needed data and is the right elevation
        if ((np.max(np.asarray(radar2.fields['differential_reflectivity']['data'])) != np.min(np.asarray(radar2.fields['differential_reflectivity']['data'])))):
            if ((np.mean(radar2.elevation['data']) < .65)):
                if (ni==0):
                    tilts.append(i)
                    n = ni+1

            else:
                tilts.append(i)

    radar = radar1.extract_sweeps(tilts)
    n = n+1

    radar3 = radar.extract_sweeps([0])
    range_i = radar3.range['data']
    # mask out last 10 gates of each ray, this removes the "ring" around the radar.
    radar.fields['differential_reflectivity']['data'][:, -10:] = np.ma.masked
    ref_ungridded_base = radar3.fields['reflectivity']['data']

    #Get 2d ranges
    range_2d = np.zeros((ref_ungridded_base.shape[0], ref_ungridded_base.shape[1]))
    for i in range(ref_ungridded_base.shape[0]):
        range_2d[i,:]=range_i
    
    #Use the calibration factor on the ZDR field
    radar.fields['differential_reflectivity']['data'] = radar.fields['differential_reflectivity']['data']-calibration
    cc_m = radar.fields['cross_correlation_ratio']['data'][:]
    refl_m = radar.fields['reflectivity']['data'][:]
    radar.fields['differential_reflectivity']['data'][:] = ma.masked_where(cc_m < 0.80, radar.fields['differential_reflectivity']['data'][:])
    radar.fields['differential_reflectivity']['data'][:] = ma.masked_where(refl_m < 20.0, radar.fields['differential_reflectivity']['data'][:])

    #Get stuff for QC control rings
    radar_t = radar.extract_sweeps([radar.nsweeps-1])
    last_height = radar_t.gate_altitude['data'][:]
    rlons_h = radar_t.gate_longitude['data']
    rlats_h = radar_t.gate_latitude['data']
    ungrid_lons = radar3.gate_longitude['data']
    ungrid_lats = radar3.gate_latitude['data']
    gate_altitude = radar3.gate_altitude['data'][:]
    
    return radar,n,range_2d,last_height,rlons_h,rlats_h,ungrid_lons,ungrid_lats