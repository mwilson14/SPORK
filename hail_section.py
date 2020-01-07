import numpy as np
from shapely import geometry
from shapely.ops import transform
from metpy.units import atleast_1d, check_units, concatenate, units
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from pyproj import Geod

def hail_objects(hailc,REF_Hail2,ax,f,time_start,month,d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end,rlons,rlats,max_lons_c,max_lats_c,proj):
    #Inputs,
    #REF_Hail2: REFmasked masked where Zdr and CC greater than 1.0
    #hailc: Contour of REF_Hail2 where reflectivity greater than 50.0 dBz
    #ax: Subplot object to be built on with each contour
    #f: Placefile, edited throughout the program
    #time_start: Radar file date and time of scan
    #month: Month of case, supplied by user
    #d_beg,h_beg,min_beg,sec_beg,d_end,h_end,min_end,sec_end: Day, hour, minute, second of the beginning and end of a scan
    #rlons,rlats: Full volume geographic coordinates, longitude and latitude respectively
    #max_lons_c,max_lats_c: Centroid coordinates of storm objects
    #proj: Projection of Earth's surface to be used for accurate area and distance calculations
    hail_areas = []
    hail_centroid_lon = []
    hail_centroid_lat = []
    hail_storm_lon = []
    hail_storm_lat = []
    if np.max(REF_Hail2) > 50.0:
        for level in hailc.collections:
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
                coord_map = np.vstack((rlons[0,:,:].flatten(), rlats[0,:,:].flatten())).T 
                #Create an Mx2 array listing all the coordinates in field
                mask_hail = polypath.contains_points(coord_map).reshape(rlons[0,:,:].shape)

                if pr_area > 2 * units('km^2'):
                    g = Geod(ellps='sphere')
                    dist_hail = np.zeros((np.asarray(max_lons_c).shape[0]))
                    for i in range(dist_hail.shape[0]):
                                distance_hail = g.inv(polygon.centroid.x, polygon.centroid.y,
                                                   max_lons_c[i], max_lats_c[i])
                                dist_hail[i] = distance_hail[2]/1000.
                    if np.min(np.asarray(dist_hail)) < 15.0:
                        hail_path = polypath
                        hail_areas.append((pr_area))
                        hail_centroid_lon.append((polygon.centroid.x))
                        hail_centroid_lat.append((polygon.centroid.y))
                        hail_storm_lon.append((max_lons_c[np.where(dist_hail == np.min(dist_hail))[0][0]]))
                        hail_storm_lat.append((max_lats_c[np.where(dist_hail == np.min(dist_hail))[0][0]]))
                        patch = PathPatch(polypath, facecolor='gold', alpha=.7, edgecolor = 'gold', linewidth = 4)
                        ax.add_patch(patch)
                        #Add polygon to placefile
                        f.write('TimeRange: '+str(time_start.year)+'-'+str(month)+'-'+str(d_beg)+'T'+str(h_beg)+':'+str(min_beg)+':'+str(sec_beg)+'Z '+str(time_start.year)+'-'+str(month)+'-'+str(d_end)+'T'+str(h_end)+':'+str(min_end)+':'+str(sec_end)+'Z')
                        f.write('\n')
                        f.write("Color: 245 242 066 \n")
                        f.write('Line: 3, 0, "Hail Core Outline" \n')
                        for i in range(len(hail_path.vertices)):
                            f.write("%.5f" %(hail_path.vertices[i][1]))
                            f.write(", ")
                            f.write("%.5f" %(hail_path.vertices[i][0]))
                            f.write('\n')
                        f.write("End: \n \n")
    #Returning Variables,
    #hail_areas: Hail core area
    #hail_centroid_lon,hail_centroid_lat: Hail core centroid coordinates
    #hail_storm_lon,hail_storm_lat: Storm object centroids associated with the hail core
    #ax: Subplot object to be built on with each contour
    #f: Placefile, edited throughout the program
    return hail_areas,hail_centroid_lon,hail_centroid_lat,hail_storm_lon,hail_storm_lat,ax,f