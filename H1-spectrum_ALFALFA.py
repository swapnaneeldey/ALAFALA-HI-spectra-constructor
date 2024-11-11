#!/usr/bin/env python
# coding: utf-8

# In[398]:


import numpy as np
import astropy
import astropy.io.fits as fits
import matplotlib.pyplot as plt
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
import astropy.units as u
import csv
from astropy.coordinates import Angle
import scipy.integrate as integrate
import pandas as pd


# In[399]:


plt.rcParams['figure.figsize'] = [10, 5]


# # Matching coordinate

# In[400]:


#please change this accordingly
AFLAFLA_cube_data = "/DATA2/jonesmg/blue_blobs_virgo/data/ALFALFA/virgo_cubes/"
catalogue_data = "/DATA1/sdey/BlueBlobs_Virgo/data/Catalogue0907.csv"
ra_array = [i for i in range(1204, 1253, 8)]
dec_array = [i for i in range(5, 20, 2)]


# In[421]:


def get_file_name(ra, dec, BC_pos):
    """
    Duty: Gets the file name associated to the position of the blue blob.
    Parameters: ra --> Right Ascension
                dec --> Declination
                BC_pos --> Astroy Skycoord class output
    """
    target_ra = 1200 + (BC_pos.ra.hour - 12)*60
    if target_ra < 1196 or target_ra > 1260:
        print(f"BC ra = {BC_pos.ra.hour} out of the search location")


    differences = [abs(ra - target_ra) for ra in ra_array]



    # Find the index of the minimum difference
    closest_index = differences.index(min(differences))

    # Get the value in ra_array closest to target_ra
    closest_value_ra = ra_array[closest_index]
    closest_value_ra

    target_dec = BC_pos.dec.deg
    differences = [abs(dec - target_dec) for dec in dec_array]
    # Find the index of the minimum difference
    closest_index = differences.index(min(differences))

    # Get the value in ra_array closest to target_ra
    closest_value_dec = dec_array[closest_index]
    closest_value_dec

    if closest_value_dec < 10:
        fits_file = f"gridbf_{closest_value_ra}+0{closest_value_dec}a.fits"
    else:
        fits_file = f"gridbf_{closest_value_ra}+{closest_value_dec}a.fits"
    return fits_file


# ## Reducing the spectrum
# 

# In[422]:


def gaussian_reduction(ra, dec, fwhm=5/60 * u.deg):
    """
    Duty: Reduce the spectrum using gaussian weighting of each spixel.
    parameters: ra --> Right Ascension
                dec --> Declination
                fwhm --> set the fwhm of the gaussian curve you want.
    """
    
    #get the file name
    BC_pos = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg))
    cube_file = AFLAFLA_cube_data + get_file_name(ra, dec, BC_pos)
    print(cube_file)

    hdu = fits.open(cube_file)
    head = hdu[0].header
    
    #get the beam width which will be used in normalizing the flux
    gaussian_beam = np.pi * head["BMAJ"] * head["BMIN"] / ((head["CDELT2"] ** 2) * 4 * np.log(2))
    
    #making an array of pixels in a frequency slice.
    cube_WCS = wcs.WCS(head, naxis=2)
    pix_array = np.indices((144, 144)).reshape(2, -1).T
    pos = astropy.wcs.utils.pixel_to_skycoord(pix_array[:, 1], pix_array[:, 0], cube_WCS, origin=1, mode='all')
    
    #do gaussian weighting
    data = hdu[0].data
    separation = BC_pos.separation(pos)
    fwhm=5/60 * u.deg
    sigma = fwhm / 2.335
    gauss_value = np.exp(-((separation) ** 2) / (2 * (sigma ** 2))).reshape(144, 144)
    
    new_data = data * gauss_value
    sum_data = np.sum(new_data, axis=(1, 2))
    
    #make the optical velocity array from the radio velocity in fits header.
    velocity_array = np.arange((head["CRVAL3"]), (head["CRVAL3"] + 1024 * head["CDELT3"]), (head["CDELT3"])) / 1000
    velocity_array = velocity_array * (3 * 10 ** 5) / (3 * 10 ** 5 - velocity_array)
    
    #normalizing and calculating noise.
    pixel_spectrum = sum_data / gaussian_beam
    masked_pixel_spectrum = np.delete(pixel_spectrum, range(600, 650))
    masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(0, 50))
    masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(len(masked_pixel_spectrum) - 50, len(masked_pixel_spectrum)))
    rms_value = np.sqrt(np.mean(masked_pixel_spectrum ** 2))
    
    return rms_value, pixel_spectrum, velocity_array


# In[431]:


def square(ra, dec, steps = 7):
    """
    Duty: Reduce the spectrum using weighting of each spixel in a specific distance from target pixel.
    parameters: ra --> Right Ascension
                dec --> Declination
                steps --> size of the square.
    """
    #call open the correct file function
    BC_pos = SkyCoord(ra, dec, unit = (u.hourangle, u.deg))
    cube_file = AFLAFLA_cube_data + get_file_name(ra, dec, BC_pos)
    print(cube_file)
    

    hdu = fits.open(cube_file)

    head = hdu[0].header
    gaussian_beam = np.pi*head["BMAJ"]*head["BMIN"]/((head["CDELT2"]**2)*4*np.log(2))
    print(gaussian_beam)
    
    cube_WCS = wcs.WCS(head,naxis=2)
    pix_x, pix_y = astropy.wcs.utils.skycoord_to_pixel(BC_pos, cube_WCS, origin=1, mode='all')
    print(pix_x,pix_y)
    data = hdu[0].data
    steps = 7
    first_edge = -steps // 2 + 1
    last_edge = steps // 2 + 1
    
    #choosing pixels around the target pixel. 
    pix_array = [[np.floor(pix_x) + i, np.floor(pix_y) + j] for i in range(first_edge, last_edge) for j in range(first_edge, last_edge)]
    sums_data = []
    for i in range(1024):
        data_array = [data[i, int(pix_array[j][1]), int(pix_array[j][0])] for j in range(len(pix_array))]
        sums_data.append(np.sum(data_array))
    
    #change radio velocity to optical velocity
    velocity_array = np.array([i/1000 for i in np.arange((head["CRVAL3"]), (head["CRVAL3"] + 1024*head["CDELT3"]), (head["CDELT3"]))])
    velocity_array = velocity_array*(3*10**5)/(3*10**5 - velocity_array)
    pixel_spectrum = sums_data/gaussian_beam
    print(pixel_spectrum.shape)
    masked_pixel_spectrum = np.copy(pixel_spectrum)
    masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(600, 650))
    masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(0, 50))
    masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(len(masked_pixel_spectrum) - 50, len(masked_pixel_spectrum)))
    #rms calculation
    squared_values = np.square(masked_pixel_spectrum)
    mean_squared = np.mean(squared_values)
    rms_value = np.sqrt(mean_squared)
    
    return rms_value, pixel_spectrum, velocity_array

    


# ## Calculating the Spectrum

# In[424]:


def spectrum(ra, dec, condition, fwhm = None, steps = None):
    """
    it calls appropriate function to get the flux values, velocity array and the noise.
    
    """
    if condition == "Gaussian":
        rms_value, pixel_spectrum, velocity_array = gaussian_reduction(ra, dec, fwhm = fwhm)
    else:
        rms_value, pixel_spectrum, velocity_array = square(ra, dec, steps = steps)
    
    return rms_value, pixel_spectrum, velocity_array


# ## Finding and Plotting the Peaks only

# In[405]:


def peaks(n, rms_value, pixel_spectrum, condition):
    """
    it helps get the position of the peaks in the spectrum. 
    
    """
    pixel_spectrum_peaks_n = [i for i in pixel_spectrum if i > n*rms_value]
    spectrum_peak = []
    indices_of_peaks = []
    old_indices = 0
    # Iterate over each peak value
    for peak_value in pixel_spectrum_peaks_n:
        # Find indices of the current peak value in pixel_spectrum
        indices = np.where(pixel_spectrum == peak_value)[0][0]
        if condition == True:
            if np.abs(indices - old_indices) > 75:
                indices_of_peaks.append(indices)
                spectrum_peak.append(peak_value)
        else:
            indices_of_peaks.append(indices)
            spectrum_peak.append(peak_value)

        old_indices = indices

    return indices_of_peaks, spectrum_peak


# In[406]:


def peaks_only(name, velocity_array, indices_of_peaks, spectrum_peak):
    """
    This function plots the peaks of the spectrum only. This makes it easier to see them.
    """
    plt.vlines([velocity_array[a] for a in indices_of_peaks], ymin=0, ymax=spectrum_peak, colors='g', linestyles='solid')
    plt.title(f"{name} peaks > 3*rms ")
    print(indices_of_peaks)


# # Plotting the whole spectrum
# 

# In[407]:


def full_spectrum(name, rms_value, velocity_array, pixel_spectrum, condition):
    """
    This function is called to plot the whole spectrum
    it takes paramaters which are the output of the 
    spectrum function.
    """
    
    if condition == True:
        indices_of_peaks_all, spectrum_peak_all = peaks(3, rms_value, pixel_spectrum, False)
        velocity_array_no_peaks = [velocity_array[i] for i in range(len(velocity_array)) if i not in indices_of_peaks_all]
        pixel_spectrum_no_peaks = [pixel_spectrum[i] for i in range(len(pixel_spectrum)) if i not in indices_of_peaks_all]
        plt.plot(velocity_array_no_peaks, pixel_spectrum_no_peaks)
        plt.vlines([velocity_array[a] for a in indices_of_peaks_all], ymin=0, ymax=spectrum_peak_all, colors='r', linestyles='solid')
        plt.title(f"{name} H1 highlighted peak spectrum")
    else:
        plt.plot(velocity_array[100:len(pixel_spectrum)-100], pixel_spectrum[100:len(pixel_spectrum)-100])
        plt.title(f"{name} H1 spectrum")
    plt.axhline(rms_value, c = "r",  label = "rms")
    plt.axhline(3*rms_value, c = "g", label = "3*rms")
    plt.axhline(5*rms_value,  c = "b", label = "5*rms")
    plt.axhline(0, c = "k", lw = 1)
    plt.ylim(-3*rms_value, 7*rms_value)
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Intensity (Jy/beam)')
    plt.legend()
    plt.show()
#plt.xlim(-500,500)


# ## Plotting narrow ranges around the peaks

# In[408]:


def spectrum_narrow(name, indices_of_peaks, velocity_array, rms_value, pixel_spectrum):
    """
    It plots peaks +- some spectrum around it.
    """
    for i in indices_of_peaks:
        plt.figure() 
        frequency = velocity_array[i - 200: i + 200]
        peak_range = pixel_spectrum[i - 200: i + 200]
        plt.plot(frequency, peak_range)
        plt.plot(frequency, [rms_value]*len(frequency),label = "rms")
        plt.axhline(0, c = "k", lw = 1)
        plt.ylim(-3*rms_value, 7*rms_value)
        plt.xlabel('Velocity (km/s)')
        plt.ylabel('Intensity (Jy/beam)')
        plt.title(f'plot of {name} peak at {velocity_array[i]} km/s')
        plt.legend()
    


# ## Making Output
# 
# 

# 
# 

# In[425]:


def read_ra_dec_from_csv(csv_file):
    """
    This reads the catalogue of blue blobs which have information regarding their postion.
    """
    ra_dec_data = {}
    pdf = pd.read_csv(csv_file)
    for i in range(len(pdf)):
        if len(pdf["Other Name"].iloc[i]) > 5:
            BC_name = pdf["Other Name"].iloc[i].split(",")[0]
        else:
            BC_name = pdf["Other Name"].iloc[i]
        ra_val = pdf["RA"][i]
        dec_val = pdf["DEC"][i]
        ra_dec_data[BC_name] = (ra_val, dec_val)
    return ra_dec_data
ra_dec_value = read_ra_dec_from_csv(catalogue_data)
ra_dec_value


# In[262]:


#this part of code calls all the function appropriately to automatically make the spectrum for all the Blueblobs. 
print(ra_dec_value.items())
for name, (ra, dec) in ra_dec_value.items():
    print(name, ra, dec)
    rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition = "")
    full_spectrum(name, rms_value, velocity_array, pixel_spectrum, False)
    full_spectrum(name, rms_value, velocity_array, pixel_spectrum, True)
    indices_of_peaks, spectrum_peak = peaks(3, rms_value, pixel_spectrum, True)
    indices_of_peaks_all, spectrum_peak_all = peaks(3, rms_value, pixel_spectrum, False)
    #peaks_only(name, velocity_array, indices_of_peaks_all, spectrum_peak_all)
    spectrum_narrow(name, indices_of_peaks, velocity_array, rms_value, pixel_spectrum)
    plt.show()

    


# ## For specific Blue Blobs

# In[430]:


#one can run this to call a specific blue blob and make spectrum of that. 
### i recommend running this since it doesnt produce too many plots
print(ra_dec_value.items())
for name, (ra, dec) in ra_dec_value.items():
    if name == "BC3":
        #name, ra, dec = 
        print(name, ra, dec)
        rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition = "")
        print(rms_value)
        full_spectrum(name, rms_value, velocity_array, pixel_spectrum, False)
        full_spectrum(name, rms_value, velocity_array, pixel_spectrum, True)
        indices_of_peaks, spectrum_peak = peaks(3, rms_value, pixel_spectrum, True)
        indices_of_peaks_all, spectrum_peak_all = peaks(3, rms_value, pixel_spectrum, False)
        peaks_only(name, velocity_array, indices_of_peaks_all, spectrum_peak_all)
        spectrum_narrow(name, indices_of_peaks, velocity_array, rms_value, pixel_spectrum)
        plt.show()
    else:
        continue


# In[201]:


ra, dec = ra_dec_value["BC3"]
rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition = "")
indices_of_peaks, spectrum_peak = peaks(3, rms_value, pixel_spectrum, True)
print(indices_of_peaks)
i = 325
plt.figure() 
frequency = velocity_array[i - 200: i + 200]
peak_range = pixel_spectrum[i - 200: i + 200]
plt.plot(frequency, peak_range)
plt.plot(frequency, [rms_value]*len(frequency),label = "rms")
plt.axhline(0, c = "k", lw = 1)
plt.ylim(-3*rms_value, 7*rms_value)
plt.xlabel('Velocity (km/s)')
plt.ylabel('Intensity (Jy/beam)')
plt.title(f'plot of {name} peak at {velocity_array[i]} km/s')
plt.axvline(1540)
plt.axvline(1615)
plt.legend()
sub = abs(velocity_array - 1615)
low_lim = min(abs(velocity_array - 1615))
a = np.where(sub == low_lim)[0][0]
new_sub = abs(velocity_array - 1540)
up_lim = min(abs(velocity_array - 1540))
b = np.where(new_sub == up_lim)[0][0]
flux_s = integrate.simps(pixel_spectrum[b:a+2:-1], velocity_array[b:a+2:-1])
print(flux_s)
flux = sum(pixel_spectrum[a+2:b])
print(flux)
pixel_spectrum[a:b]


# In[277]:


"""for name, (ra, dec) in ra_dec_value.items():
    print(name, ra, dec)
    rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition = "")
    full_spectrum(rms_value, velocity_array, pixel_spectrum, False)
    print(rms_value)"""
        


# In[432]:


int_flux_dict = {}
rms_list = []
for name, (ra, dec) in ra_dec_value.items():
    print(name, ra, dec)
    rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition = "")
    flux = 5*rms_value*(np.sqrt(5/30))*30
    int_flux_dict[name] = flux
    rms_list.append(float(rms_value))


# In[443]:


HI_mass_upper_limit = {}
for name, values in int_flux_dict.items():
    HI_mass_upper_limit[name] = '{:.3}'.format(np.log10(values*2.356*(10**5)*(16.5**2)))


# In[444]:


HI_mass_upper_limit


# In[435]:


rms_list


# In[436]:


int_flux_dict


# # Calculating the HI mass upper limits
# 

# In[368]:


def see_spectrum(BC):
    for name, (ra, dec) in ra_dec_value.items():
        if name == BC:  
            print(name, ra, dec)
            rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition="")
            print(rms_value)
            fig, ax = plt.subplots()
            ax.plot(velocity_array[100:len(pixel_spectrum)-100], pixel_spectrum[100:len(pixel_spectrum)-100])
            ax.set_title(f"{name} H1 spectrum")
            ax.axhline(rms_value, color="r", label="rms")
            ax.axhline(3*rms_value, color="g", label="3*rms")
            ax.axhline(5*rms_value, color="b", label="5*rms")
            ax.axhline(0, color="k", linewidth=1)
            ax.set_ylim(-3*rms_value, 7*rms_value)
            ax.set_xlabel('Velocity (km/s)')  
            ax.set_ylabel('Intensity (Jy/beam)')  
            ax.legend()
            return fig, ax, pixel_spectrum, velocity_array


# In[410]:


fig, ax, pixel_spectrum, velocity_array = see_spectrum("BC7")
ax.axvline(x=0, color='r', linestyle='--', label='x = 500')
masked_pixel_spectrum = np.copy(pixel_spectrum)
masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(np.argmax(pixel_spectrum)-50, 650))
masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(0, 50))
masked_pixel_spectrum = np.delete(masked_pixel_spectrum, range(len(masked_pixel_spectrum) - 50, len(masked_pixel_spectrum)))
#rms calculation
squared_values = np.square(masked_pixel_spectrum)
mean_squared = np.mean(squared_values)
rms_value = np.sqrt(mean_squared)
print(rms_value)


# In[375]:


plt.plot(velocity_array, pixel_spectrum)


# In[376]:


np.argmax(pixel_spectrum)


# In[394]:


indices = np.where((velocity_array >= 0) & (velocity_array <= 5))[0]
indices[0]



# In[385]:


velocity_array[632]


# In[395]:


fraud = []
for name, (ra, dec) in ra_dec_value.items():
    print(name)
    rms_value, pixel_spectrum, velocity_array = spectrum(ra, dec, condition="")
    val = np.where((velocity_array >= 0) & (velocity_array <= 5))[0][0]
    ind = np.argmax(pixel_spectrum)
    if abs(val - ind) > 20:
        fraud.append((name, val - ind))


# In[396]:


fraud


# In[ ]:




