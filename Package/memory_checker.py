from droplets import *

if __name__ == '__main__':
    full_path='/Volumes/LaCie Drive/Lab/Raw_data/Home_lab/200602/Diffusor_30_40mm_gain40/255kHz_48ulpermin-06012020114101-1050.png'
    #do_cutouts('/Volumes/LaCie Drive/Lab/Raw_data/Home_lab/200602/Diffusor_30_40mm_gain40/255kHz_48ulpermin-06012020114101-1050.png',d_um=2.2,smallest_size_um2=10000,closing_kernel=30,threshold_fraction=0.15,show_diagnostics=True)
    do_stuff(full_path)
    #GD_metric(cv2.imread('/Volumes/LaCie Drive/Lab/Raw_data/Home_lab/200602/Diffusor_30_40mm_gain40/255kHz_48ulpermin-06012020114101-1050.png'))
    
