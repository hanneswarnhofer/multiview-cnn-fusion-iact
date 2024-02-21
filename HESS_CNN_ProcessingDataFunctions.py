import tables
import tables
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf
import random

import fnmatch
import os
#import h5
import glob
import pickle
import sys
import argparse
import h5py
import os.path
import inspect
import json

from datetime import datetime
import time

from ctapipe.io import EventSource
from ctapipe import utils
from ctapipe.instrument.camera import CameraGeometry

from dl1_data_handler.reader import DL1DataReader
from dl1_data_handler.image_mapper import ImageMapper
from sklearn.metrics import roc_curve, auc

class DataManager():
    """ Data class used to manage the HDF5 data files (simulations + Auger data).
        data_path: data_path of HDF5 file, (hint: use blosc compression to ensure adequate decompression speed,
        to mitigate training bottleneck due to a slow data pipeline)
        params:
            data_path = path to HDF5 datset
        optional params:
            stats: data statistics (stats.json - needed for scaling the dataset)
            tasks: list of tasks to be included (default: ['axis', 'core', 'energy', 'xmax'])
            generator_fn: generator function used for looping over data, generator function needs to have indices and
                          shuffle args.
            ad_map_fn: "advanced mapping function" the function used to map the final dataset. Here an additional
                       preprocessing can be implemented which is mapped during training on the
                       cpu (based on tf.data.experimental.map_and_batch)
    """

    def __init__(self, data_path, stats=None, tasks=['axis', 'impact', 'energy', 'classification']):
        ''' init of DataManager class, to manage simulated (CORSIKA/Offline) and measured dataset '''
        current_timestamp = int(time.time())
        np.random.seed(current_timestamp)
        self.data_path = data_path

    def open_ipython(self):
        from IPython import embed
        embed()

    @property
    def is_data(self):
        return self.type == "Data"

    @property
    def is_mc(self):
        return self.type == "MC"

    def get_h5_file(self):
        return h5py.File(self.data_path, "r")

    def walk_tree(self, details=True):
        """ Draw the tree of yout HDF5 file to see the hierachy of your dataset
            params: detail(activate details to see shapes and used compression ops, Default: True)
        """

        def walk(file, iter_str=''):
            try:
                keys = file.keys()
            except AttributeError:
                keys = []

            for key in keys:
                try:
                    if details:
                        file[key].dtype
                        print(iter_str + str(file[key]))
                    else:
                        print(iter_str + key)
                except AttributeError:
                    print(iter_str + key)
                    walk(file[key], "   " + iter_str)

        with h5py.File(self.data_path, "r") as file:
            print("filename:", file.filename)
            for key in file.keys():
                print(' - ' + key)
                walk(file[key], iter_str='   - ')

    def extract_info(self, path):
        with self.get_h5_file() as f:
            data = f[path]
            y = np.stack(data[:].tolist())

        return {k: y[:, i] for i, k in enumerate(data.dtype.names)}, dict(data.dtype.descr)

    def make_mc_data(self):
        return self.extract_info("simulation/event/subarray/shower")

def re_index_ct14(image):
    return image[5:, :, :]

def make_hess_geometry(file=None):
    # quick fix for dl1 data handler to circumvent to use ctapipe
    if file is None:
        with open(os.path.join(os.getcwd(), "geometry2d3.json")) as f: 
            attr_dict = json.load(f)

        data_ct14 = attr_dict["ct14_geo"]
        data_ct5 = attr_dict["ct5_geo"]
    else:
        data_ct14 = file["configuration/instrument/telescope/camera/geometry_0"][:].tolist()
        data_ct5 = file["configuration/instrument/telescope/camera/geometry_1"][:].tolist()

    class Geometry():
        def __init__(self, data):
            self.pix_id, self.pix_x, self.pix_y, self.pix_area = np.stack(data).T.astype(np.float32)
            self.pos_x = self.pix_x
            self.pos_y = self.pix_y

        def get_pix_pos(self):
            return np.column_stack([self.pix_x, self.pix_y]).T

    return Geometry(data_ct14), Geometry(data_ct5)

def get_current_path():
    filename = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(filename))

def rotate(pix_pos, rotation_angle=0):
    rotation_angle = rotation_angle * np.pi / 180.0
    rotation_matrix = np.matrix([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                                [np.sin(rotation_angle), np.cos(rotation_angle)], ], dtype=float)

    pixel_positions = np.squeeze(np.asarray(np.dot(rotation_matrix, pix_pos)))
    return pixel_positions


##########################################################################################
##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
##########################################################################################

def generate_single_data(train_data, train_labels, threshold=60):
    # Dimensions of the input data
    num_events, num_views, height, width, channels = train_data.shape
    
    # Reshape train_data into a single array with shape (num_events * num_views, height, width, channels)
    reshaped_data = train_data.reshape(-1, height, width, channels)
    reshaped_labels = np.repeat(train_labels,4,axis=0)


    # Compute the sum of pixel values along the last axis (assuming your images are grayscale)
    pixel_sums = np.sum(reshaped_data, axis=(1, 2, 3))
    
    # Find indices of images with sums greater than or equal to the threshold
    valid_indices = np.where(pixel_sums >= threshold)
    
    # Filter the reshaped data based on valid_indices
    filtered_data = reshaped_data[valid_indices]
    # Reshape train_labels to match the reshaped data
    filtered_labels = reshaped_labels[valid_indices]

    combined_data_labels = list(zip(filtered_data, filtered_labels))
    # Shuffle the combined array
    np.random.shuffle(combined_data_labels)

    # Split the shuffled array back into filtered_data and filtered_labels
    shuffled_data, shuffled_labels = zip(*combined_data_labels)

    # Convert them back to NumPy arrays
    shuffled_data = np.array(shuffled_data)
    shuffled_labels = np.array(shuffled_labels)

    return shuffled_data, shuffled_labels

def plot_image_2by2(train_data,event_nr,labels,string,dt):
    print("Plotting Example Event. Event Nr: ", event_nr)

    if (string == 'single train' or string == 'single test'): # Choosing four different images (single view data is shuffled before)
        image1 = train_data
        pltimage1 = image1[event_nr]
        pltimage2 = image1[event_nr+1]
        pltimage3 = image1[event_nr+2]
        pltimage4 = image1[event_nr+3]

        label1 = labels[event_nr].ravel()
        label2 = labels[event_nr+2].ravel()
        label3 = labels[event_nr+3].ravel()
        label4 = labels[event_nr+4].ravel()

        if label1 == 1: str_label1 = "Gamma" 
        elif label1 == 0: str_label1 = "Proton" 
        else: str_label1 = "Unknown"
        if label2 == 1: str_label2 = "Gamma" 
        elif label2 == 0: str_label2 = "Proton" 
        else: str_label2 = "Unknown"
        if label3 == 1: str_label3 = "Gamma" 
        elif label3 == 0: str_label3 = "Proton" 
        else: str_label3 = "Unknown"
        if label4 == 1: str_label4 = "Gamma" 
        elif label4 == 0: str_label4 = "Proton" 
        else: str_label4 = "Unknown"
    
    elif (string == 'train' or string=='test' or string == 'mapped'):
        image1 = train_data[:,0,:,:] 
        image2 = train_data[:,1,:,:] 
        image3 = train_data[:,2,:,:] 
        image4 = train_data[:,3,:,:] 

        pltimage1 = image1[event_nr]
        pltimage2 = image2[event_nr]
        pltimage3 = image3[event_nr]
        pltimage4 = image4[event_nr]

        label = labels[event_nr].ravel()

        if label == 1: str_label1 = "Gamma" 
        elif label == 0: str_label1 = "Proton" 
        else: str_label1 = "Unknown"

        str_label2 = str_label1
        str_label3 = str_label1
        str_label4 = str_label1
    else: print("Unknown string specified during plotting, don't know what to do here.")

    fig, ax = plt.subplots(2,2)

    im1 = ax[0,0].imshow(pltimage1[:,:,0], cmap='viridis',vmin=0)
    im2 = ax[0,1].imshow(pltimage2[:,:,0], cmap='viridis',vmin=0)
    im3 = ax[1,0].imshow(pltimage3[:,:,0], cmap='viridis',vmin=0)
    im4 = ax[1,1].imshow(pltimage4[:,:,0], cmap='viridis',vmin=0)

    cbar1 = fig.colorbar(im1, ax=ax[0, 0], orientation='vertical')
    cbar2 = fig.colorbar(im2, ax=ax[0, 1], orientation='vertical')
    cbar3 = fig.colorbar(im3, ax=ax[1, 0], orientation='vertical')
    cbar4 = fig.colorbar(im4, ax=ax[1, 1], orientation='vertical')    

    ax[0, 0].text(0.05, 0.95, str_label1, transform=ax[0, 0].transAxes, color='white', fontsize=12, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    ax[0, 1].text(0.05, 0.95, str_label2, transform=ax[0, 1].transAxes, color='white', fontsize=12, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 0].text(0.05, 0.95, str_label3, transform=ax[1, 0].transAxes, color='white', fontsize=12, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    ax[1, 1].text(0.05, 0.95, str_label4, transform=ax[1, 1].transAxes, color='white', fontsize=12, ha='left', va='top', bbox=dict(facecolor='black', alpha=0.7))
    fig.suptitle(string,fontsize=15)
    #plt.show()
    #time.sleep(3)
    #plt.close()

    #print("Min. and Max. Value for Image 1: ", np.min(pltimage1), " - " , np.max(pltimage1) , ". Sum: ", np.sum(pltimage1))
    #print("Min. and Max. Value for Image 2: ", np.min(pltimage2), " - " , np.max(pltimage2), ". Sum: ", np.sum(pltimage2))
    #print("Min. and Max. Value for Image 3: ", np.min(pltimage3), " - " , np.max(pltimage3), ". Sum: ", np.sum(pltimage3))
    #print("Min. and Max. Value for Image 4: ", np.min(pltimage4), " - " , np.max(pltimage4), ". Sum: ", np.sum(pltimage4))

    str_evnr = '{}'.format(event_nr)
    name = "Test_images/Test_figure_evnr_" + str_evnr + "_" + string + "_" + dt + ".png"
    #fig.savefig(name)

    return im1, im2, im3, im4

def plot_random_images(data, labels, label_str, dt):
    randnum = np.random.randint(0, np.shape(data)[0])
    #print(f"{label_str} {randnum}: Image plotted!")
    splitstring = label_str.split()
    string = splitstring[0].lower()
    plot_image_2by2(data, randnum, labels, string=string, dt=dt)

def plot_single_images(data, labels, label_str, dt):
    randnum = np.random.randint(0, np.shape(data)[0] - 4) 
    #print(f"Single {label_str} {randnum}: Image plotted!")
    splitstring = label_str.split()
    string = 'single ' + splitstring[0].lower()
    plot_image_2by2(data, randnum, labels, string=string, dt=dt)

def create_pdf(train_data, train_labels, test_data, test_labels, single_train_data, single_train_labels, single_test_data, single_test_labels, formatted_datetime,loc):
    
    if loc == 'local':
        pdf_filename = "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/Test_images/example_images_" + formatted_datetime + ".pdf"
    else: pdf_filename = "Test_images/example_images_" + formatted_datetime + ".pdf"

    with pdf.PdfPages(pdf_filename) as pdf_pages:
        # Page 1: Random Train Data Events
        for i in range(6):
            plot_random_images(train_data, train_labels, "Train Data Event", formatted_datetime)
            plt.savefig(pdf_pages, format='pdf')
            plt.close()

        # Page 2: Random Test Data Events
        for i in range(6):
            plot_random_images(test_data, test_labels, "Test Data Event", formatted_datetime)
            plt.savefig(pdf_pages, format='pdf')
            plt.close()

        # Page 3: Single Train Data Events
        for i in range(6):
            plot_single_images(single_train_data, single_train_labels, "Train Data Event", formatted_datetime)
            plt.savefig(pdf_pages, format='pdf')
            plt.close()

        # Page 4: Single Test Data Events
        for i in range(6):
            plot_single_images(single_test_data, single_test_labels, "Test Data Event", formatted_datetime)
            plt.savefig(pdf_pages, format='pdf')
            plt.close()

    print(f"PDF file '{pdf_filename}' created successfully.")

def handle_nan(data,labels):
    # Check for NaN values
    nan_mask = np.isnan(data)
    nan_event_mask = np.any(np.any(np.any(nan_mask, axis=-1), axis=-1), axis=-1)

    events_to_remove = np.where(nan_event_mask)[0]
    # Remove events from data and labels
    data = np.delete(data, events_to_remove, axis=0)
    labels = np.delete(labels, events_to_remove, axis=0)
    
    # Calculate total count and percentage of NaN entries
    total_nan_entries = np.sum(nan_mask)
    total_entries = np.size(data)
    nan_percentage = (total_nan_entries / total_entries) * 100.0
    
    print(f"Total NaN entries: {total_nan_entries}")
    print(f"Percentage of NaN entries: {nan_percentage:.6f}%")
    
    return data , labels

def print_event_composition(labels, event_types=('proton', 'gamma')):
    if labels.shape[1] != 1:
        raise ValueError("The input array should have shape (num_events, 1)")

    event_counts = {event_types[0]: np.sum(labels == 0), event_types[1]: np.sum(labels == 1)}
    print("Event Composition:")
    for event_type, count in event_counts.items():
        print(f"{count} '{event_type}' events")


def load_telescope_data(file_path,amount,label_value):
    dm = DataManager(file_path)
    f = dm.get_h5_file()
    data = [f[f"dl1/event/telescope/images/tel_{i:03d}"] for i in range(1,5)]
    length, = np.shape(data[0])
    print("Events avaialable: ",length)
    del data
    amount = int(amount)
    data_select = [f[f"dl1/event/telescope/images/tel_{i:03d}"][:amount] for i in range(1,5)]
    shape, = np.shape(data_select[0])
    print("Shape: ",shape)
    if label_value == 1: labels = np.ones_like(range(shape))
    elif label_value == 0: labels = np.zeros_like(range(shape))
    else: print("Invalid label_value! Must be 0 or 1!")

    return data_select , labels

def dataloader(num_events, location,use_combined_data=False):
    # Configure FilePaths:
    file_paths = {
        'local': 
            {'gamma': "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5",
            'proton': "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ECAP_HiWi_WorkingDirectory/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"},
        'alex': 
            {'gamma': "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/gamma_diffuse_noZBDT_noLocDist_hybrid_v2.h5",
            'proton': "../../../../wecapstor1/caph/mppi111h/new_sims/dnn/proton_noZBDT_noLocDist_hybrid_v2.h5",
            'old_gamma': "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_gamma_diffuse_hybrid_preselect_20deg_0deg.h5",
            'old_proton': "../../../../wecapstor1/caph/mppi111h/old_dataset/phase2d3_timeinfo_proton_hybrid_preselect_20deg_0deg.h5"}}


    if use_combined_data and location == 'alex':
        # Load half of the events from the old data and half from the new data
        old_gamma_data, old_gamma_labels = load_telescope_data(file_paths['alex']['gamma'], num_events*0.35, label_value=1)
        old_proton_data, old_proton_labels = load_telescope_data(file_paths['alex']['proton'], num_events*0.35, label_value=0)

        new_gamma_data, new_gamma_labels = load_telescope_data(file_paths['alex']['old_gamma'], num_events*0.35, label_value=1)
        new_proton_data, new_proton_labels = load_telescope_data(file_paths['alex']['old_proton'], num_events*0.35, label_value=0)

        gamma_data = [np.concatenate((old_gamma_data[i], new_gamma_data[i]), axis=0) for i in range(4)]
        proton_data = [np.concatenate((old_proton_data[i], new_proton_data[i]), axis=0) for i in range(4)]

        gamma_labels = np.concatenate([old_gamma_labels, new_gamma_labels], axis=0)
        proton_labels = np.concatenate([old_proton_labels, new_proton_labels], axis=0)
    else:
        # Load data as before
        gamma_data, gamma_labels = load_telescope_data(file_paths[location]['gamma'], num_events * 0.7, label_value=1)
        proton_data, proton_labels = load_telescope_data(file_paths[location]['proton'], num_events * 0.7, label_value=0)

    print("Loaded data")

    # Concatenate data and labels
    tel_data = [np.concatenate((gamma_data[i], proton_data[i]), axis=0) for i in range(4)]
    labels = np.concatenate([gamma_labels, proton_labels], axis=0)

    print("Shape 'tel_data': ", np.shape(tel_data))
    print("Shape 'labels': ",np.shape(labels))

    return tel_data , labels

def datamapper(data,labels,num_events,cut_nonzero,threshold_value):

    individual_threshold_value = 0.0001

    # Mapping setup
    geo_ct14, _ = make_hess_geometry()
    ct_14_mapper = ImageMapper(camera_types=["HESS-I"], pixel_positions={"HESS-I": rotate(geo_ct14.get_pix_pos())}, mapping_method={"HESS-I": "axial_addressing"})
    #mapped_images = {i: [] for i in range(4)}

    mapped_images_1, mapped_images_2, mapped_images_3, mapped_images_4 = [], [], [], []
    mapped_images = [mapped_images_1, mapped_images_2, mapped_images_3, mapped_images_4]
    mapped_labels = []

    mapped_labels = []
    valid_events_count = 0
    
    # Random sampling
    random_list = np.random.randint(len(labels), size=2*num_events)
    # Mapping loop
    for event_nr in random_list:
        if valid_events_count == num_events:
            break

        images = [ct_14_mapper.map_image(data[i][event_nr][3][:, np.newaxis], 'HESS-I') for i in range(4)]

        # If a pixel is below the individual pixel threshold, set it to zero
        for img in images: img[img < individual_threshold_value] = 0

        # If the sum over all pixels is lower than threshold, set all pixels to zero
        for img in images: img[:] = np.where(img.sum() < threshold_value, 0, img)

        # Check non-zero count
        non_zero_count = sum(1 for img in images if np.sum(img) > 0)

        # Only use events with at least cut_nonzero usable images
        if non_zero_count >= cut_nonzero:
            mapped_images_1.append(images[0])
            mapped_images_2.append(images[1])
            mapped_images_3.append(images[2])
            mapped_images_4.append(images[3])
            mapped_labels.append(labels[event_nr])
            valid_events_count += 1

    # Convert the lists to arrays
    mapped_images_1 = np.array(mapped_images_1)
    mapped_images_2 = np.array(mapped_images_2)
    mapped_images_3 = np.array(mapped_images_3)
    mapped_images_4 = np.array(mapped_images_4)
    mapped_labels = np.array(mapped_labels)
    print("... Finished Mapping")
    print("Mapped Label Selection: ",mapped_labels[-20:-10])

    # Reshape the final array, so it is present in the same way as MoDAII data
    mapped_images = np.transpose(mapped_images, (1, 0, 2, 3, 4))
    mapped_labels = mapped_labels[:,np.newaxis]

    return mapped_images , mapped_labels

def data_splitter(mapped_images,mapped_labels,plot,formatted_datetime,loc):
    # Overview about the data array
    print(f"{np.shape(mapped_images)[0]} events with 4 images each are available")
    print("Shape of 'mapped_labels': ", np.shape(mapped_labels))
    print("Shape of 'mapped_images': ", np.shape(mapped_images), "\n")

    # Split into random training data (80%) and test data (20%)
    random_selection = np.random.rand(np.shape(mapped_images)[0]) <= 0.8

    train_data = mapped_images[random_selection]
    test_data = mapped_images[~random_selection]
    train_labels = mapped_labels[random_selection]
    test_labels = mapped_labels[~random_selection]

    print(random_selection[0:10])

    # Free memory
    del mapped_images
    del mapped_labels

    len_train = np.shape(train_data)[0]
    len_test = np.shape(test_data)[0]

    print(len_train)
    print(len_test)
    
    # Overvew about the splitting into training and test data
    print("Split into Training and Test Data")
    print("Train data shape:", np.shape(train_data), "-->", round(100 * len_train / (len_train + len_test), 2), "%")
    print("Test data shape:", np.shape(test_data), "-->", round(100 * len_test / (len_train + len_test), 2), "%")
    print("Train labels shape:", np.shape(train_labels))
    print("Test labels shape:", np.shape(test_labels))

    train_data, train_labels = handle_nan(train_data, train_labels)
    test_data, test_labels = handle_nan(test_data, test_labels)

    print("Test Set:")
    print_event_composition(test_labels)
    print("\nTrain Set:")
    print_event_composition(train_labels)

    single_train_data, single_train_labels = generate_single_data(train_data, train_labels)
    single_test_data, single_test_labels = generate_single_data(test_data, test_labels)

    print(np.shape(single_train_data))
    print(np.shape(single_train_labels))

    if plot == 'yes':
        plot_random_images(train_data, train_labels, "Train Data Event", formatted_datetime)
        plot_random_images(test_data, test_labels, "Test Data Event", formatted_datetime)
        plot_single_images(single_train_data, single_train_labels, "Train Data Event", formatted_datetime)
        plot_single_images(single_test_data, single_test_labels, "Test Data Event", formatted_datetime)
    elif plot == 'pdf':
        create_pdf(train_data, train_labels, test_data, test_labels, single_train_data, single_train_labels, single_test_data, single_test_labels, formatted_datetime,loc)


    return single_train_data, single_train_labels, single_test_data, single_test_labels, train_data, train_labels, test_data, test_labels

def create_strings(fnr,num_events,formatted_datetime,batch_size,dropout_rate,reg,num_epochs,fusiontype,transfer,base,normalize,filters_1,learning_rate,sb,eb):
    str_batch_size = '{}'.format(batch_size)
    str_num_events = '{}'.format(num_events)
    str_rate = '{}'.format(dropout_rate)
    str_reg = '{}'.format(reg)
    #str_num_epochs = '{}'.format(num_epochs)
    #str_thr = '{}'.format(sum_threshold)
    #str_cnz = '{}'.format(cut_nonzero)
    str_transfer = '{}'.format(transfer)
    str_base = '{}'.format(base)
    str_norm = '{}'.format(normalize)
    str_filter = '{}'.format(filters_1)
    str_learning = '{}'.format(learning_rate)
    str_sb = '{}'.format(sb)
    str_eb = '{}'.format(eb)
    if base == "resnet":
        str_name = fnr + "_" + str_sb + "to" + str_eb + "_"
    else:
        str_name = fnr + "_"
    name_str = str_name + fusiontype + "_" + str_base  + "-base_" + str_num_events + "events_"  +  str_batch_size  + "bsize_" + str_rate + "drate_" + str_learning + "lrate_" + str_reg + "reg_" + str_filter + "fil_" + str_transfer + "transf_" + str_norm + "_" + formatted_datetime 
    name_single_str =  str_name + "_singleCNN_" + fusiontype + "_" + str_base  + "-base_" + str_num_events + "events_"  +  str_batch_size + "bsize_" + str_rate + "drate_" + str_learning + "lrate_" + str_reg + "reg_" + str_filter + "fil_" + str_transfer + "transf_" + str_norm + "_" + formatted_datetime 

    return name_str, name_single_str

def create_history_plot(history,name_str,base):

    history_name = "HistoryFiles/history_" + name_str + ".pkl"
    with open(history_name, 'wb') as file:
        pickle.dump(history.history, file)

    # Create plots for quick overview
    fig1, ax1 = plt.subplots(1,2, figsize = (9,3))
    ax1[0].plot(history.history['accuracy'], label='Training Data',lw=2,c="darkorange")
    ax1[0].plot(history.history['val_accuracy'], label = 'Validation Data',lw=2,c="firebrick")
    ax1[0].set_xlabel('Epoch')
    ax1[0].set_ylabel('Accuracy')
    ax1[0].set_ylim([0.5, 1])
    ax1[0].legend(loc='lower right')

    ax1[1].plot(history.history['loss'],lw=2,c="darkorange")
    ax1[1].plot(history.history['val_loss'],lw=2,c="firebrick")
    ax1[1].set_ylabel('Loss')
    ax1[1].set_xlabel('Epoch')

    fig1.suptitle(base)

    print("Image created")

    filename_savefig = "Images/Test_Cluster_" + name_str + ".png"
    fig1.savefig(filename_savefig, bbox_inches='tight')

    print("Image saved")

def save_model(model,name_str,loc):
    if loc == 'local':
        model_name_str = "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ModelFiles/modelfile_" +  name_str + ".h5"
    else: model_name_str = "ModelFiles/modelfile_" +  name_str + ".h5"
    model.save(model_name_str)

def plot_roc_curve(model, X_test, y_test,name_str,loc):
    if loc == 'local':
        save_filename = "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ROC/roc_curve_" +  name_str 
    else: save_filename = "ROC/roc_curve_" +  name_str 

    """
    Plot ROC curve for a Keras model.

    Parameters:
    - model: Keras model
    - X_test: Test data
    - y_test: True labels for the test data
    """
    # Predict probabilities on the test set
    y_pred = model.predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)


    # Save each array separately to avoid dimension mismatch
    np.save(save_filename + '_fpr.npy', fpr)
    np.save(save_filename + '_tpr.npy', tpr)
    np.save(save_filename + '_roc_auc.npy', roc_auc)
    print(f"Data saved to {save_filename}")
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    #plt.show()

    if loc == 'local':
        filename_savefig = "../../../mnt/c/Users/hanne/Desktop/Studium Physik/ECAP_HiWi_CNN/ROC_Images/roc_curve_" +  name_str +".png"
    else: filename_savefig = "ROC_Images/roc_curve_" +  name_str + ".png"

    #filename_savefig = "ROC_Images/roc_curve_" + name_str + ".png"
    plt.savefig(filename_savefig, bbox_inches='tight')

    print("ROC Curve saved")



def plot_roc_curve_from_history(history):
    """
    Plot ROC curve from Keras training history.

    Parameters:
    - history: Keras training history object
    """
    plt.figure(figsize=(8, 8))

    # Extract true labels and predicted probabilities
    y_true = history.history['val_labels']  # Replace 'val_labels' with the actual name in your history
    y_pred_proba = history.history['val_predictions']  # Replace 'val_predictions' with the actual name in your history

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


import numpy as np

def calculate_mean_accuracy_in_intensity_bins(model, validation_data, validation_labels,min_intensity,max_intensity, nr_bins,output_file):


    event_intensity = np.sum(validation_data, axis=(1, 2, 3, 4))

    #min_intensity = np.min(event_intensity)
    #max_intensity = np.max(event_intensity)
    bin_width = (max_intensity - min_intensity) / nr_bins
    print(f"Bin Width: {bin_width}")
    # Print some sanity checks
    print(f"Shape of batch_data: {validation_data.shape}")
    print(f"First 10 event intensities: {event_intensity[:10]}")


    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = min_intensity + i * bin_width
            bin_end = min_intensity + (i + 1) * bin_width
            # Calculate the mean intensity of the bin
            bin_center = (bin_start + bin_end) / 2

            # Indices of events within the current intensity bin
            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]



            # Select data and labels for events within the current intensity bin
            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            # Print additional checks
            print(f"Mean Intensity: {bin_center}")
            print(f"Number of events within intensity bin: {len(selected_indices)}")

            if len(selected_indices) == 0:
                print(f"No events in intensity bin: {bin_center}")
                continue
            #print(f"Shape of selected_data: {selected_data.shape}")
            input_data = [selected_data[:, j, :, :, :] for j in range(4)]
            #print(f"Shape of Input Data for Validation: {input_data.shape}")
            # Evaluate the model on the selected data
            results = model.evaluate(input_data, selected_labels, verbose=0)
            print("Results:", results)

            # Extract the accuracy from the results
            accuracy = results[1]  # assuming accuracy is at index 1

            

            # Append results to lists
            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            # Write results to the file
            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    
    return mean_accuracies, bin_centers

def calculate_mean_accuracy_in_intensity_bins_quantile(model, validation_data, validation_labels,nr_bins, output_file):
    event_intensity = np.sum(validation_data, axis=(1, 2, 3, 4))

    bin_edges = np.percentile(event_intensity, np.linspace(0, 100, nr_bins + 1))
    
    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]

            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            if len(selected_indices) == 0:
                continue

            input_data = [selected_data[:, j, :, :, :] for j in range(4)]
            results = model.evaluate(input_data, selected_labels, verbose=0)
            accuracy = results[1]

            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    return mean_accuracies, bin_centers

def calculate_mean_accuracy_in_intensity_bins_log(model, validation_data, validation_labels, min_intensity, max_intensity, nr_bins, output_file):
    event_intensity = np.sum(validation_data, axis=(1, 2, 3, 4))

    bin_edges = np.logspace(np.log10(min_intensity), np.log10(max_intensity), nr_bins + 1)
    
    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]

            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            if len(selected_indices) == 0:
                continue

            input_data = [selected_data[:, j, :, :, :] for j in range(4)]
            results = model.evaluate(input_data, selected_labels, verbose=0)
            accuracy = results[1]

            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    return mean_accuracies, bin_centers

def calculate_mean_accuracy_in_intensity_bins_log_single_view(model, validation_data, validation_labels, min_intensity, max_intensity, nr_bins, output_file):
    event_intensity = np.sum(validation_data, axis=(1, 2, 3))

    bin_edges = np.logspace(np.log10(min_intensity), np.log10(max_intensity), nr_bins + 1)

    # Print some sanity checks
    print(f"Shape of validation_data: {validation_data.shape}")
    print(f"First 10 event intensities: {event_intensity[:10]}")

    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            # Indices of events within the current intensity bin
            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]

            # Select data and labels for events within the current intensity bin
            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            # Print additional checks
            print(f"Mean Intensity: {bin_center}")
            print(f"Number of events within intensity bin: {len(selected_indices)}")

            if len(selected_indices) == 0:
                print(f"No events in intensity bin: {bin_center}")
                continue

            # Evaluate the model on the selected data
            results = model.evaluate(selected_data, selected_labels, verbose=0)
            print("Results:", results)

            # Extract the accuracy from the results
            accuracy = results[1]  # assuming accuracy is at index 1

            # Append results to lists
            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            # Write results to the file
            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    return mean_accuracies, bin_centers

def calculate_mean_accuracy_in_intensity_bins_single_view(model, validation_data, validation_labels, min_intensity, max_intensity, nr_bins, output_file,type,single):
    
    
    if single == True:
        event_intensity = np.sum(validation_data, axis=(1, 2, 3))
    else: 
        event_intensity = np.sum(validation_data, axis=(1, 2, 3, 4))
        
    bin_width = (max_intensity - min_intensity) / nr_bins
    print(f"Bin Width: {bin_width}")
    # Print some sanity checks
    print(f"Shape of validation_data: {validation_data.shape}")
    print(f"First 10 event intensities: {event_intensity[:10]}")

    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = min_intensity + i * bin_width
            bin_end = min_intensity + (i + 1) * bin_width
            # Calculate the mean intensity of the bin
            bin_center = (bin_start + bin_end) / 2

            # Indices of events within the current intensity bin
            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]

            # Select data and labels for events within the current intensity bin
            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            # Print additional checks
            print(f"Mean Intensity: {bin_center}")
            print(f"Number of events within intensity bin: {len(selected_indices)}")

            if len(selected_indices) == 0:
                print(f"No events in intensity bin: {bin_center}")
                continue

            # Evaluate the model on the selected data
            results = model.evaluate(selected_data, selected_labels, verbose=0)
            print("Results:", results)

            # Extract the accuracy from the results
            accuracy = results[1]  # assuming accuracy is at index 1

            # Append results to lists
            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            # Write results to the file
            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    return mean_accuracies, bin_centers

def calculate_mean_accuracy_in_intensity_bins_quantile_single_view(model, validation_data, validation_labels,  nr_bins, output_file):
    event_intensity = np.sum(validation_data, axis=(1, 2, 3))

    bin_edges = np.percentile(event_intensity, np.linspace(0, 100, nr_bins + 1))
    
    mean_accuracies = []
    bin_centers = []

    with open(output_file, 'w') as file:
        file.write("Bin,Mean_Accuracy,Mean_Intensity\n")

        for i in range(nr_bins):
            bin_start = bin_edges[i]
            bin_end = bin_edges[i + 1]
            bin_center = (bin_start + bin_end) / 2

            # Indices of events within the current intensity bin
            selected_indices = np.where(np.logical_and(event_intensity >= bin_start, event_intensity < bin_end))[0]

            # Select data and labels for events within the current intensity bin
            selected_data = validation_data[selected_indices]
            selected_labels = validation_labels[selected_indices]

            # Print additional checks
            print(f"Mean Intensity: {bin_center}")
            print(f"Number of events within intensity bin: {len(selected_indices)}")

            if len(selected_indices) == 0:
                print(f"No events in intensity bin: {bin_center}")
                continue

            # Evaluate the model on the selected data
            results = model.evaluate(selected_data, selected_labels, verbose=0)
            print("Results:", results)

            # Extract the accuracy from the results
            accuracy = results[1]  # assuming accuracy is at index 1

            # Append results to lists
            mean_accuracies.append(accuracy)
            bin_centers.append(bin_center)

            # Write results to the file
            file.write(f"{i},{accuracy:.4f},{bin_center:.2f}\n")

    return mean_accuracies, bin_centers