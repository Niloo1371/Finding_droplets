# Finding_droplets
This code will try different thresholds for a 2d image and will cut out the regions that contain objects. Also checks if it is a grey or color image.

# Monte Carlo simulation for error estimation -->

# In order to generate the MC files:

- Use the notebook called “Example_notebook_for_analysis.ipynb” in Example folder (note that cv2 in this notebook is not needed so you can comment it out if you want).
- File called “series.py” should be in the same folder since it should be loaded into the “Example_notebook_for_analysis.ipynb” 
- Read different interpolation files for temperature vs time in h5 format for different droplet sizes (the files are in the folder called Temperature_sims_and_interpolation and they are called as “t_calib_xx.h5", xx is the droplet size)
- Running this notebook gives the files in forms of “xxxx_with_types_measuredpos.txt” where xxxx is the date of the experiment.
- These txt files then can be read in the notebook called “Jnuc_fits.ipynb” for further analysis.
