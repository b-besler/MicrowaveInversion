### Script to analyze the difference between sets of microwave inverse images
import os
import numpy as np
from matplotlib import pyplot as plt


# folder_path1 = 'C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\full_arm\\images'
# folder_path2 = 'C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\full_arm_95\\images'
# actual_path1 = 'C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\full_arm\\measurement'
# actual_path2 = 'C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\full_arm_95\\measurement'

pc_weight = [995, 990, 985, 980]
pc_weight_array = np.array([0.5, 1.0, 1.5, 2.0])

reco_image_er_means = np.zeros(len(pc_weight))
reco_image_er_stds = np.zeros(len(pc_weight))
actual_image_er = np.zeros(len(pc_weight))
reco_image_sig_means = np.zeros(len(pc_weight))
reco_image_sig_stds = np.zeros(len(pc_weight))
actual_image_sig = np.zeros(len(pc_weight))

# get baseline reconstruction (i.e. weight = 100%)
baseline_folder = os.path.join('C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\duke_forearm\\images', str(100))
baseline_actual_folder = os.path.join('C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\duke_forearm\\results', str(100))

n_er_baseline = len([f for f in os.listdir(baseline_folder) if f.endswith('.npy') and f.find('er')>0])
n_sig_baseline = len([f for f in os.listdir(baseline_folder) if f.endswith('.npy') and f.find('sig')>0])

baseline_er_images = np.zeros((n_er_baseline, 20, 20))
baseline_sig_images = np.zeros((n_sig_baseline, 20, 20))
print(f"Number of baseline images: {n_er_baseline}")

i = 0
for file in os.listdir(baseline_folder):
    if file.endswith('.npy') and file.find('er') > 0:
        baseline_er_images[i,:,:] = np.load(os.path.join(baseline_folder, file))
        i += 1
        # plt.imshow(np.squeeze(baseline_er_images[0,:,:]))
        # plt.title("Reconstruction - Baseline - Permittivity")
        # plt.colorbar()
        # plt.show()

i = 0
for file in os.listdir(baseline_folder):
    if file.endswith('.npy') and file.find('sig') > 0:
        baseline_sig_images[i,:,:] = np.load(os.path.join(baseline_folder, file))
        i += 1
        # plt.imshow(np.squeeze(baseline_sig_images[0,:,:]))
        # plt.title("Reconstruction - Baseline - Conductivity")
        # plt.colorbar()
        # plt.show()

for file in os.listdir(baseline_actual_folder):
    if file.endswith('.npy') and file.find('sig') > 0:
        actual_baseline_sig = np.load(os.path.join(baseline_actual_folder, file))
        # plt.imshow(actual_baseline_sig)
        # plt.colorbar()
        # plt.title("Actual - Baseline - Conductivity")
        # plt.show()

for file in os.listdir(baseline_actual_folder):
    if file.endswith('.npy') and file.find('er') > 0:
        actual_baseline_er = np.load(os.path.join(baseline_actual_folder, file))
        # plt.imshow(actual_baseline_er)
        # plt.colorbar()
        # plt.title("Actual - Baseline - Permittivity")
        # plt.show()

indx = 0
for weight in pc_weight:
    print(weight)
    var_folder =  os.path.join('C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\duke_forearm\\images', str(weight))
    var_actual_folder = os.path.join('C:\\Users\\brendon.besler\\MicrowaveInversion\\models\\duke_forearm\\results', str(weight))

    # get number of images
    n_er_var = len([f for f in os.listdir(var_folder) if f.endswith('.npy') and f.find('er')>0])
    n_sig_var = len([f for f in os.listdir(var_folder) if f.endswith('.npy') and f.find('sig')>0])

    # initialize array of images

    var_er_images = np.zeros((n_er_var, 20, 20))
    var_sig_images = np.zeros((n_sig_var, 20, 20))
    print(f"Number of variation images: {n_er_var}")
    # load images
            
    i = 0
    for file in os.listdir(var_folder):
        if file.endswith('.npy') and file.find('er') > 0:
            var_er_images[i,:,:] = np.load(os.path.join(var_folder, file))
            i += 1
            # plt.imshow(np.squeeze(baseline_er_images[0,:,:]))
            # plt.title("Reconstruction - 2% Weight Loss - Permittivity")
            # plt.colorbar()
            # plt.show()
            
    i = 0
    for file in os.listdir(var_folder):
        if file.endswith('.npy') and file.find('sig') > 0:
            var_sig_images[i,:,:] = np.load(os.path.join(var_folder, file))
            i += 1
            # plt.imshow(np.squeeze(baseline_sig_images[0,:,:]))
            # plt.title("Reconstruction - 2% Weight Loss - Conductivity")
            # plt.colorbar()
            # plt.show()
            
    # get actual images
    for file in os.listdir(var_actual_folder):
        if file.endswith('.npy') and file.find('sig') > 0:
            actual_var_sig = np.load(os.path.join(var_actual_folder, file))
            # plt.imshow(np.squeeze(actual_var_sig))
            # plt.colorbar()
            # plt.title("Actual - 2% Weight Loss - Conductivity")
            # plt.show()
    for file in os.listdir(var_actual_folder):
        if file.endswith('.npy') and file.find('er') > 0:
            actual_var_er = np.load(os.path.join(var_actual_folder, file))
            # plt.imshow(actual_var_er)
            # plt.colorbar()
            # plt.title("Actual - 2% Weight Loss - Permittivity")
            # plt.show()

    actual_delta_er = actual_baseline_er.mean() - actual_var_er.mean()
    actual_delta_sig = actual_baseline_sig.mean() - actual_var_sig.mean()

    baseline_er_mean = baseline_er_images.mean(axis=(1,2))
    var_er_mean = var_er_images.mean(axis=(1,2))
    baseline_sig_mean = baseline_sig_images.mean(axis=(1,2))
    var_sig_mean = var_sig_images.mean(axis=(1,2))

    mean_difference_er = baseline_er_mean - var_er_mean
    mean_difference_sig = baseline_sig_mean - var_sig_mean

    # plt.plot(baseline_er_mean)
    # plt.title("Permittivity Mean")
    # plt.show()

    # plt.plot(var_er_mean)
    # plt.title("Permittivity Mean")
    # plt.show()

    baseline_er_images= np.squeeze(baseline_er_images)
    baseline_sig_images= np.squeeze(baseline_sig_images)
    var_er_images= np.squeeze(var_er_images)
    var_sig_images= np.squeeze(var_sig_images)

    mean_mean_er = np.mean(mean_difference_er)
    mean_mean_sig = np.mean(mean_difference_sig)

    mean_std_er = np.std(mean_difference_er)
    mean_std_sig = np.std(mean_difference_sig)

    print(f"Actual permittivity difference: {actual_delta_er}")
    print(f"Actual conductivity difference: {actual_delta_sig}")

    print(f"Permittivity mean difference: {mean_mean_er}")
    print(f"Conductivity mean difference: {mean_mean_sig}")

    print(f"Permittivity mean std: {mean_std_er}")
    print(f"Conductivity mean std: {mean_std_sig}")

    reco_images = {
        "baseline_er": baseline_er_images,
        "baseline_sig": baseline_sig_images,
        "var_er":var_er_images,
        "var_sig":var_sig_images
    }

    actual_images = {
        "baseline_er": actual_baseline_er,
        "baseline_sig": actual_baseline_sig,
        "var_er":actual_var_er,
        "var_sig":actual_var_sig,
    }

    for image in reco_images:
        reco_image = reco_images[image]
        actual_image = actual_images[image]

        x_section = reco_image[:,int(reco_image.shape[1]/2)]
        y_section = reco_image[int(reco_image.shape[0]/2), :]
        x = np.linspace(-0.036, 0.036, num = x_section.size)
        y = np.linspace(-0.036, 0.036, num = y_section.size)

        x_section2 = actual_image[:,int(actual_image.shape[1]/2)]
        y_section2 = actual_image[int(actual_image.shape[0]/2), :]
        x2 = np.linspace(-0.036, 0.036, num = x_section2.size)
        y2 = np.linspace(-0.036, 0.036, num = y_section2.size)

        # plt.plot(x,x_section, label = 'Reconstruction')
        # plt.xlabel("x [m]")
        # plt.plot(x2,x_section2, label = 'Actual')
        # plt.title('Cross section through y')
        # plt.legend()
        # plt.show()

        # plt.plot(y,y_section, label = 'Reconstruction')
        # plt.xlabel("y [m]")
        # plt.plot(y2,y_section2, label = 'Actual')
        # plt.title('Cross section through y')
        # plt.legend()
        # plt.show()

    actual_image_er[indx] = actual_delta_er
    actual_image_sig[indx] = actual_delta_sig
    reco_image_er_means[indx] = mean_mean_er
    reco_image_sig_means[indx] = mean_mean_sig
    reco_image_er_stds[indx] = mean_std_er.astype(float)
    reco_image_sig_stds[indx] =mean_std_sig.astype(float)

    indx += 1

print(f"Actual permittivity difference: {actual_image_er}")
print(f"Reconstructed permittivity difference: {reco_image_er_means}")
print(f"Actual conductivity difference: {actual_image_sig}")
print(f"Reconstructed conductivity difference: {reco_image_sig_means}")

correlation_matrix = np.corrcoef(actual_image_er, reco_image_er_means)
correlation_xy = correlation_matrix[0,1]
print(f"R-squared for permittivity: {correlation_xy**2}")

correlation_matrix = np.corrcoef(actual_image_sig, reco_image_sig_means)
correlation_xy = correlation_matrix[0,1]
print(f"R-squared for conductivity: {correlation_xy**2}")

plt.errorbar(pc_weight_array, reco_image_er_means, yerr = reco_image_er_stds, label = 'Reconstructed')
plt.plot(pc_weight_array,actual_image_er, label = 'Actual')
plt.title('Permittivity vs Weight Loss')
plt.ylabel('Permittivity [ ]')
plt.xlabel('Weight Loss [%]')
plt.legend()
plt.grid()
plt.show()

plt.errorbar(pc_weight_array, reco_image_sig_means, yerr = reco_image_sig_stds, label = 'Reconstructed')
plt.plot(pc_weight_array,actual_image_sig, label = 'Actual')
plt.title('Conductivity vs Weight Loss')
plt.ylabel('Conductivity [S/m]')
plt.xlabel('Weight Loss [%]')
plt.legend()
plt.grid()
plt.show()

