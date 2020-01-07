import numpy as np
import numpy.ma as ma
from scipy import ndimage as ndi
from metpy.units import atleast_1d, check_units, concatenate, units
from metpy.calc import get_wind_dir, get_wind_speed, get_wind_components

def grad_mask(Zint,REFmasked,REF,storm_relative_dir,ZDRmasked1,ZDRrmasked1,CC,CCall):
      print('Gradient Analysis and Masking')
      smoothed_ref1 = ndi.gaussian_filter(REFmasked, sigma = 2, order = 0)
      REFgradient = np.asarray(np.gradient(smoothed_ref1))
      REFgradient[0,:,:] = ma.masked_where(REF < 20, REFgradient[0,:,:])
      REFgradient[1,:,:] = ma.masked_where(REF < 20, REFgradient[1,:,:])
      grad_dir1 = get_wind_dir(REFgradient[1,:,:] * units('m/s'), REFgradient[0,:,:] * units('m/s'))
      grad_mag = get_wind_speed(REFgradient[1,:,:] * units('m/s'), REFgradient[0,:,:] * units('m/s'))
      grad_dir = ma.masked_where(REF < 20, grad_dir1)

      #Get difference between the gradient direction and the FFD gradient direction calculated earlier
      srdir = storm_relative_dir
      srirad = np.copy(srdir)*units('degrees').to('radian')
      grad_dir = grad_dir*units('degrees').to('radian')
      grad_ffd = np.abs(np.arctan2(np.sin(grad_dir-srirad), np.cos(grad_dir-srirad)))
      print(grad_ffd)
      print(np.max(grad_ffd))
      print(np.min(grad_ffd))
      grad_ffd = np.asarray(grad_ffd)*units('radian')
      grad_ex = np.copy(grad_ffd)
      grad_ffd = grad_ffd.to('degrees')
      print(grad_ffd)
      print(np.max(grad_ffd))
      print(np.min(grad_ffd))

      #Mask out areas where the difference between the two is too large and the ZDR is likely not in the forward flank
      ZDRmasked2 = ma.masked_where(grad_ffd > 120 * units('degrees'), ZDRmasked1)
      ZDRmasked = ma.masked_where(CC < .60, ZDRmasked2)
      ZDRallmasked = ma.masked_where(CCall < .70, ZDRrmasked1)
      ZDRallmasked = ma.filled(ZDRallmasked, fill_value = -2)
      ZDRrmasked = ZDRallmasked[Zint,:,:]

      #Add a fill value for the ZDR mask so that contours will be closed
      ZDRmasked = ma.filled(ZDRmasked, fill_value = -2)
      ZDRrmasked = ma.filled(ZDRrmasked, fill_value = -2)

      return grad_mag,grad_ffd,ZDRmasked,ZDRallmasked,ZDRrmasked