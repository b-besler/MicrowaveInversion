import numpy as np
from matplotlib import image
from matplotlib import pyplot as plt
import scipy.ndimage
import scipy.spatial as sp
import scipy.stats
import mwi.util.dispersive_models as er_model

#pixel_value_list = [0.435,0.556,0.707,0.795, 0.784,0.45,0.582,0.838, 0.767,0.828, ]
pixel_value_list = [0.247,0.188,0.612,0.796, 0.98,0.0,0.326,0.839, 0.78,0.635]
material_list = [
    er_model.material_indices['FreeSpace'],
    er_model.material_indices['Muscle'],
    er_model.material_indices['Skin'],
    er_model.material_indices['Fat'],
    er_model.material_indices['Fat'],
    er_model.material_indices['Blood'],
    er_model.material_indices['Blood'],
    er_model.material_indices['Bone(Cortical)'],
    er_model.material_indices['Bone(Cancellous)'],
    er_model.material_indices['Bone(Cancellous)']
]

colors = [(0,0,0),(250,0,0),(0,250,0),(250,250,250),(0,0,250),(125,125,125)]

file_name = 'duke_forearm_mod.png'

image = image.imread(file_name)
print(image[0,0,:])
plt.imshow(image)
plt.show()
image = np.round(image*1000).astype(int)

x = np.unique(image.reshape(image.shape[0]*image.shape[1], image.shape[2]), axis = 0)
print(x)


remap = {
       (   0,    0,    0, 1000): [er_model.material_indices['Fat']],
       (   0,    0,  376, 1000): [er_model.material_indices['Fat']],
       (   0,    0,  522, 1000): [er_model.material_indices['Fat']],
       (   0,    0,  525, 1000): [er_model.material_indices['Fat']],
       (   0,    0,  714, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  796, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  788, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  863, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  855, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  922, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  973, 1000): [er_model.material_indices['Blood']],
       (   0,    0,  980, 1000): [er_model.material_indices['Blood']],
       (   0,  980,    0, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 176,  937,  176, 1000): [er_model.material_indices['Bone(Cancellous)']],
       ( 247,  247,  247, 1000): [er_model.material_indices['FreeSpace']],
       ( 251,  890,  251, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 306,  839,  306, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 353,  784,  353, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 380,    0,    0, 1000): [er_model.material_indices['Fat']],
       ( 380,    0,  796, 1000): [er_model.material_indices['Blood']],
       ( 380,    0,  922, 1000): [er_model.material_indices['Blood']],
       ( 380,  380,  380, 1000): [er_model.material_indices["Skin"]],
       ( 380,  922,    0, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 392,  725,  392, 1000): [er_model.material_indices['Bone(Cancellous)']],
       ( 427,  659,  427, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 439,  439,  439, 1000): [er_model.material_indices['Skin']],
       ( 459,  580,  459, 1000): [er_model.material_indices['Bone(Cortical)']],
       ( 490,  490,  490, 1000): [er_model.material_indices['Bone(Cancellous)']],
       ( 525,    0,    0, 1000): [er_model.material_indices['Muscle']],
       ( 525,    0,  863, 1000): [er_model.material_indices['Blood']],
       ( 525,  380,  380, 1000): [er_model.material_indices['Skin']],
       ( 525,  525,  525, 1000):[er_model.material_indices['Skin']],
       ( 525,  863,    0, 1000):[er_model.material_indices['Bone(Cortical)']],
       ( 561,  561,  561, 1000):[er_model.material_indices['Skin']],
       ( 631,    0,    0, 1000):[er_model.material_indices['Muscle']],
       ( 631,    0,  796, 1000): [er_model.material_indices["Muscle"]],
       ( 631,  631,  631, 1000):[er_model.material_indices["Skin"]],
       ( 631,  796,    0, 1000):[er_model.material_indices["Bone(Cortical)"]],
       ( 655,  655,  655, 1000): [er_model.material_indices['Skin']],
       ( 722,    0,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 722,    0,  722, 1000):[er_model.material_indices["Muscle"]],
       ( 722,  525,  525, 1000):[er_model.material_indices["Muscle"]],
       ( 722,  631,  631, 1000):[er_model.material_indices["Muscle"]],
       ( 722,  722,    0, 1000):[er_model.material_indices["Bone(Cortical)"]],
       ( 722,  722,  722, 1000):[er_model.material_indices["Skin"]],
       ( 737,  737,  737, 1000):[er_model.material_indices["Skin"]],
       ( 796,    0,    0, 1000):[er_model.material_indices['Muscle']],
       ( 796,    0,  525, 1000):[er_model.material_indices['Muscle']],
       ( 796,    0,  631, 1000):[er_model.material_indices['Muscle']],
       ( 796,  380,  380, 1000):[er_model.material_indices["Muscle"]],
       ( 796,  631,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 796,  722,  722, 1000):[er_model.material_indices['Muscle']],
       ( 796,  796,  796, 1000):[er_model.material_indices["Skin"]],
       ( 804,  804,  804, 1000):[er_model.material_indices["Skin"]],
       ( 863,    0,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 863,    0,  525, 1000):[er_model.material_indices['Muscle']],
       ( 863,  380,  380, 1000):[er_model.material_indices['Muscle']],
       ( 863,  525,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 863,  796,  796, 1000): [er_model.material_indices["Skin"]],
       ( 863,  863,  863, 1000): [er_model.material_indices["Skin"]],
       ( 871,  871,  871, 1000):[er_model.material_indices["Skin"]],
       ( 922,    0,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 922,    0,  380, 1000):[er_model.material_indices['Muscle']],
       ( 922,  380,    0, 1000):[er_model.material_indices["Muscle"]],
       ( 922,  922,  922, 1000):[er_model.material_indices["Skin"]],
       ( 925,  925,  925, 1000): [er_model.material_indices["Skin"]],
       ( 980,    0,    0, 1000):[er_model.material_indices['Muscle']],
       ( 980,  380,  380, 1000):[er_model.material_indices['Muscle']],
       ( 980,  631,  631, 1000):[er_model.material_indices['Muscle']],
       ( 980,  722,  722, 1000):[er_model.material_indices['Muscle']],
       ( 980,  863,  863, 1000):[er_model.material_indices['Muscle']],
       ( 980,  980,  980, 1000):[er_model.material_indices["Skin"]]
}

image_out = np.ones((image.shape[0], image.shape[1]))*-1
for k in remap:
    indx1 = image[:,:,0] == k[0]
    indx2 =image[:,:,1] == k[1]
    indx3 = image[:,:,2] == k[2]
    indx = np.logical_and(np.logical_and(indx1, indx2), indx3)
    image_out[indx] = remap[k][0]

print(image_out.shape)
plt.imshow(image_out)
plt.show()

image_out = image_out[23:623, 107:707]

plt.imshow(image_out)
plt.show()

np.save("duke_forearm.npy", image_out)

print(image_out.shape)

scale = 30
print(image_out.shape[0] // scale)
indx = np.linspace(0, image_out.shape[0] // scale -1, num =image_out.shape[0] // scale ).astype(int)
print(indx)
print(indx.size)


image_out2 = image_out[0::scale,0::scale]
print(image_out2.shape)
plt.imshow(image_out2)
plt.title("Subsample - Subselection")
plt.show()



image_out2 = np.zeros((indx.size, indx.size))
for i in indx:
    for j in indx:
        image_out2[i,j] = scipy.stats.mode(np.ravel(image_out[i*scale:(i+1)*scale, j*scale:(j+1)*scale]))[0]
image_out2 = np.ceil(image_out2)
print(image_out2.shape)
plt.imshow(image_out2)
plt.title('Subsample - average')
plt.show()

np.save('duke_forearm_20_20.npy',image_out2)