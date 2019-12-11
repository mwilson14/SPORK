import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
from pyproj import Geod
from metpy.calc import get_wind_dir, get_wind_speed, get_wind_components

def zdrcol(zdrrc,ZDRrmasked,CC_c,REFrmasked,grad_ffd,grad_mag,KDP,ZDR_sum_stuff,KDPrmasked,depth_levels,forest_loaded_col,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,ref_areas,proj,storm_relative_dir,tracking_ind,object_number):
    col_areas = []
    col_maxdepths = []
    col_depths = []
    col_centroid_lon = []
    col_centroid_lat = []
    col_storm_lon = []
    col_storm_lat = []
    col_masks = []
    if np.max(ZDRrmasked) > 1.0:
        for level in zdrrc.collections:
            for contour_poly in level.get_paths(): 
                for n_contour,contour in enumerate(contour_poly.to_polygons()):
                    contour_a = np.asarray(contour[:])
                    xa = contour_a[:,0]
                    ya = contour_a[:,1]
                    polygon_new = geometry.Polygon([(i[0], i[1]) for i in zip(xa,ya)])
                    if n_contour == 0:
                        polygon = polygon_new
                    else:
                        polygon = polygon.difference(polygon_new)

                pr_area = (transform(proj, polygon).area * units('m^2')).to('km^2')
                boundary = np.asarray(polygon.boundary.xy)
                polypath = Path(boundary.transpose())
                coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T # create an Mx2 array listing all the coordinates in field
                mask_col = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)
                mean_col = np.mean(ZDRrmasked[mask_col])
                mean_cccol = np.mean(CC_c[mask_col])
                mean_Zcol = np.mean(REFrmasked[mask_col])
                mean_graddircol = np.mean(grad_ffd[mask_col])
                mean_gradcol = np.mean(grad_mag[mask_col])
                mean_kdpcol = np.mean(KDP[mask_col])
                col_depth = ZDR_sum_stuff[mask_col]
                mean_kdp_r = np.mean(KDPrmasked[mask_col])

                try:
                    max_depth = np.max(col_depth)
                    mean_depth = np.mean(col_depth)
                except:
                    print('col_depth', col_depth)
                    max_depth = np.nan
                    mean_depth = np.nan
                if (pr_area > 2 * units('km^2')) and (mean_col > 1):
                    g = Geod(ellps='sphere')
                    dist_col = np.zeros((np.asarray(max_lons_c).shape[0]))
                    forw_col = np.zeros((np.asarray(max_lons_c).shape[0]))
                    back_col = np.zeros((np.asarray(max_lons_c).shape[0]))
                    rawangle_col = np.zeros((np.asarray(max_lons_c).shape[0]))
                    for i in range(dist_col.shape[0]):
                                distance_col = g.inv(polygon.centroid.x, polygon.centroid.y,
                                                       max_lons_c[i], max_lats_c[i])
                                dist_col[i] = distance_col[2]/1000.
                                back_col[i] = distance_col[1]
                                if distance_col[1] < 0:
                                    back_col[i] = distance_col[1] + 360
                                forw_col[i] = np.abs(back_col[i] - storm_relative_dir)
                                rawangle_col[i] = back_col[i] - storm_relative_dir
                                #Account for weird angles
                                if forw_col[i] > 180:
                                    forw_col[i] = 360 - forw_col[i]
                                    rawangle_col[i] = (360-forw_col[i])*(-1)
                                rawangle_col[i] = rawangle_col[i]*(-1)

                    if np.min(np.asarray(dist_col)) < 30.0:
                        #Use ML algorithm to eliminate non-arc objects
                        #Get x and y components
                        #Write stuff to csv for training
                        storm_area = ref_areas[np.where(dist_col == np.min(dist_col))[0][0]]
                        # if (((max_lons_c[np.where(dist_col == np.min(dist_col))[0][0]]) in max_lons_c[tracking_ind]) and ((max_lats_c[np.where(dist_col == np.min(dist_col))[0][0]]) in max_lats_c[tracking_ind])):
                        #     plt.text(float(polygon.centroid.x), float(polygon.centroid.y), "%.1f" %(float(object_number)), size = 23, color = 'r')
                        #     #Add a line that writes all of the attibutes of each object to a csv
                        #     with open('Machine_Learning/ML_columns4'+station+str(dt.year)+str(dt.month)+str(dt.day)+str(dt.hour)+str(dt.minute)+'.csv', 'a') as csvfile:
                        #         fieldnames = ['number', 'hour', 'minute','area','distance','angle','mean','max','mean_cc','mean_kdp','mean_Z','mean_graddir','mean_grad', 'max_depth', 'mean_depth', 'col_mean_kdp', 'storm_area']
                        #         writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        #         writer.writerow({'number': object_number, 'hour': hour, 'minute': minute, 'area': pr_area.magnitude, 'distance': np.min(dist_col), 'angle': rawangle_col[np.where(dist_col == np.min(dist_col))[0][0]], 'mean': mean_col, 'max': np.max(ZDRrmasked[mask_col]), 'mean_cc': mean_cccol, 'mean_kdp': mean_kdpcol, 'mean_Z': mean_Zcol, 'mean_graddir': mean_graddircol.magnitude, 'mean_grad': mean_gradcol.magnitude, 'max_depth': max_depth, 'mean_depth': mean_depth, 'col_mean_kdp': mean_kdp_r, 'storm_area': storm_area})
                        #     object_number=object_number+1

                        #Change this to rawangle once that switch has been made in the ML algorithm
                        if (rawangle_col[np.where(dist_col == np.min(dist_col))[0][0]] > 0):
                            directions_raw = 360 - rawangle_col[np.where(dist_col == np.min(dist_col))[0][0]]
                        else:
                            directions_raw = (-1) * rawangle_col[np.where(dist_col == np.min(dist_col))[0][0]]

                        if storm_area < 200:
                            storm_area = 200
                        
                        dist_norm = np.min(dist_col)/np.sqrt(storm_area/np.pi)

                        xc, yc = get_wind_components(dist_norm*units('m/s'), directions_raw * units('degree'))
                        COL_X = np.zeros((1, 15))
                        COL_X[:,0] = pr_area.magnitude
                        COL_X[:,1] = dist_norm
                        COL_X[:,2] = mean_col
                        COL_X[:,3] = np.max(ZDRrmasked[mask_col])
                        COL_X[:,4] = mean_cccol
                        COL_X[:,5] = mean_kdpcol
                        COL_X[:,6] = mean_Zcol
                        COL_X[:,7] = mean_graddircol
                        COL_X[:,8] = mean_gradcol
                        COL_X[:,9] = max_depth
                        COL_X[:,10] = mean_depth
                        COL_X[:,11] = mean_kdp_r
                        COL_X[:,12] = storm_area
                        COL_X[:,13] = xc
                        COL_X[:,14] = yc
                        pred_col = forest_loaded_col.predict(COL_X)
                        if pred_col[0]==1:
                            col_path = polypath
                            col_areas.append((pr_area))
                            col_maxdepths.append(max_depth)
                            col_depths.append(mean_depth)
                            col_centroid_lon.append((polygon.centroid.x))
                            col_centroid_lat.append((polygon.centroid.y))
                            col_storm_lon.append((max_lons_c[np.where(dist_col == np.min(dist_col))[0][0]]))
                            col_storm_lat.append((max_lats_c[np.where(dist_col == np.min(dist_col))[0][0]]))
                            patch = PathPatch(polypath, facecolor='none', alpha=.7, edgecolor = 'cyan', linewidth = 3, zorder=11)
                            ax.add_patch(patch)
                            col_masks.append(mask_col)
                            #Add polygon to placefile
                            f.write('TimeRange: '+str(time_start.year)+'-'+str(month)+'-'+str(d_beg)+'T'+str(h_beg)+':'+str(min_beg)+':'+str(sec_beg)+'Z '+str(time_start.year)+'-'+str(month)+'-'+str(d_end)+'T'+str(h_end)+':'+str(min_end)+':'+str(sec_end)+'Z')
                            f.write('\n')
                            f.write("Color: 066 245 245 \n")
                            f.write('Line: 3, 0, "ZDR Column Outline" \n')
                            for i in range(len(col_path.vertices)):
                                f.write("%.5f" %(col_path.vertices[i][1]))
                                f.write(", ")
                                f.write("%.5f" %(col_path.vertices[i][0]))
                                f.write('\n')
                            f.write("End: \n \n")
                            depth_cmask = np.copy(ZDR_sum_stuff)
                            depth_cmask[~mask_col] = 0
                            try:
                                plt.contourf(rlons[0,:,:], rlats[0,:,:], depth_cmask, depth_levels, cmap=plt.cm.viridis, zorder=10)
                            except:
                                print("empty column")
    return col_areas,col_maxdepths,col_depths,col_centroid_lon,col_centroid_lat,col_storm_lon,col_storm_lat,ax,col_masks,f