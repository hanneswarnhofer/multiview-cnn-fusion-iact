
from HESS_CNN_ProcessingDataFunctions import *
from HESS_CNN_CreateModelsFunctions import *

print("Functions Defined.")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=50)
parser.add_argument("-b", "--batch_size", type=int,default=256)
parser.add_argument("-r", "--rate", type=float,default=0.2)
parser.add_argument("-reg", "--regulization", type=float,default=0.00001)
parser.add_argument("-t", "--threshold", type=float,default=60)
parser.add_argument("-c", "--cut", type=int,default=2)
parser.add_argument("-ne", "--numevents", type=int,default=100000) # ca 1.250.000 events available in total but memory problems!
parser.add_argument("-ft","--fusiontype",type=str,default="latefc")
parser.add_argument("-n","--normalize",type=str,default="nonorm")
parser.add_argument("-loc","--location",type=str,default="alex")
parser.add_argument("-transfer","--transfer",type=str,default="no")
parser.add_argument("-base","--base",type=str,default='moda')
parser.add_argument("-lr",'--learningrate',type=float,default=0.0005)
parser.add_argument("-plt",'--plot',type=str,default='no')
parser.add_argument("-fil",'--filter',type=int,default=512)
parser.add_argument("-single",'--single',type=str,default='yes')
parser.add_argument("-comb",'--combine_data',type=bool,default=False)
parser.add_argument("-sb", "--startblock", type=int,default=0)
parser.add_argument("-eb", "--endblock", type=int,default=12)
parser.add_argument("-nbins", "--nr_bins", type=int,default=20)
parser.add_argument("-bmax", "--bin_max", type=int,default=15000)
parser.add_argument("-sval", "--single_val", type=bool,default=False)

args = parser.parse_args()
num_epochs = args.epochs
batch_size = args.batch_size
dropout_rate = args.rate
reg = args.regulization
sum_threshold = args.threshold
cut_nonzero = args.cut
num_events = args.numevents
fusiontype = args.fusiontype
normalize = args.normalize
location = args.location
transfer = args.transfer
base = args.base
learning_rate = args.learningrate
plot = args.plot
filters_1 = args.filter
single = args.single
combine_data = args.combine_data
startblock = args.startblock
endblock = args.endblock
nr_bins = args.nr_bins
bin_max = args.bin_max
single_val = args.single_val

print("############################################################################")
print("\n #####################    FUSIONTYPE: ",fusiontype,"   #######################")
print("\n")
print("\n Epochs: ", num_epochs)
print("\n Batch Size: ", batch_size)
print("\n Regularization: ", reg)
print("\n Events: ", num_events)
print("\n Learning Rate: ", learning_rate)
print("\n Dropout Rate: ", dropout_rate)
print("\n Filters 1: ",filters_1)
print("\n Transfer: ", transfer)
print("\n Threshold: ", sum_threshold)
print("\n Nonzero Cut: ", cut_nonzero)
print("\n Plotting Events: ", plot)
print("\n Base CNN: ", base)
print("\n Single View: ", single)
print("\n Combine Data: ", combine_data)
print("\n")

# Define the appendix to the file, for being able to specify some general changes in the model structure and trace back the changes when comparing the results of t´different models
fnr = "All-MoDA-Base" 


current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d_%H-%M")
#formatted_date = current_datetime.strftime("%-Y-%m-%d")
print("Date-Time: ", formatted_datetime)
name_str, name_single_str = create_strings(fnr,num_events, formatted_datetime,batch_size,dropout_rate,reg,num_epochs,fusiontype,transfer,base,normalize,filters_1,learning_rate,startblock,endblock)

#folder_str = formatted_date + "_" + appendix

data , labels = dataloader(num_events, location,combine_data)
mapped_images , mapped_labels = datamapper(data,labels,num_events,cut_nonzero,sum_threshold)
single_train_data, single_train_labels, single_test_data, single_test_labels, train_data, train_labels, test_data, test_labels = data_splitter(mapped_images,mapped_labels,plot,formatted_datetime,location)


patience = 10
input_shape = (41, 41, 1)
pool_size = 2
kernel_size = 4
decay = learning_rate/num_epochs


print("##################################################################")

#def lr_time_based_decay(epoch, lr):
#        return lr * 1 / (1 + decay * epoch)

my_callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',patience=patience,verbose=1,mode='auto')]

my_loss = BinaryCrossentropy(from_logits=True)
my_opt = keras.optimizers.Adam(learning_rate=learning_rate)

if single == 'yes':

    print("\n #####################   SINGLE VIEW MODEL   #######################")
    print("###### ",base, " ##### ",fusiontype," ######")

    inputs = Input(shape=input_shape)

    if base == 'moda':
        base_cnn = create_base_model_moda(input_shape,kernel_size,dropout_rate,reg,pool_size)(inputs)
    elif base == 'resnet':
        base_cnn = create_base_model_customresnet(input_shape,'SingleModel',dropout_rate,filters_1,startblock,endblock,fusiontype)(inputs)
    else: print("Unknown Base Model Specified! Must be 'moda' or 'resnet' .")

    single_view_model = create_single_model(base_cnn)

    single_view_model.compile(optimizer=my_opt, loss='binary_crossentropy', metrics=['accuracy'])
    #single_view_model.summary()

    save_model(single_view_model,name_single_str,location)
    single_history = single_view_model.fit(single_train_data, single_train_labels, epochs=num_epochs, batch_size=batch_size,validation_data=(single_test_data,single_test_labels),callbacks=my_callbacks)

    #base_cnn_weights = single_view_model.get_layer('lambda').get_weights()
    if transfer == 'yes': 
        single_view_model.save_weights('single_cnn_weights_partial.h5')


    plot_roc_curve(single_view_model, single_test_data,single_test_labels,name_single_str,location)
    base_str_single = base + "_singleview"
    create_history_plot(single_history,name_single_str,base=base_str_single)

    if single_val == True:
        poe_single_str_lin = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "Singlelin.csv"
        poe_single_str_log = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "Singlelog.csv"
        poe_single_str_quan = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "Singlequan.csv"
        # Adjust the intensity bin as needed

        mean_accuracy = calculate_mean_accuracy_in_intensity_bins_single_view(single_view_model, single_test_data, single_test_labels,0,bin_max,nr_bins,poe_single_str_lin)
        mean_accuracy = calculate_mean_accuracy_in_intensity_bins_log_single_view(single_view_model, single_test_data, single_test_labels,150,bin_max,nr_bins,poe_single_str_log)
        mean_accuracy = calculate_mean_accuracy_in_intensity_bins_quantile_single_view(single_view_model, single_test_data, single_test_labels,nr_bins,poe_single_str_quan)


    
model_multi = create_multi_model(base, transfer, fusiontype, input_shape, kernel_size, dropout_rate, reg, pool_size, filters_1,startblock,endblock)
save_model(model_multi,name_str,location)
model_multi.summary()

model_multi.compile(optimizer=my_opt, loss='binary_crossentropy', metrics=['accuracy'])
history = model_multi.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels), callbacks=my_callbacks)
#history = compiled_multi_view_model.fit([train_data[:,i,:,:] for i in range(4)],train_labels,epochs=num_epochs,batch_size=batch_size,validation_data=([test_data[:,i,:,:] for i in range(4)], test_labels))
print("... Finished the Fitting")

plot_roc_curve(model_multi, [test_data[:,i,:,:] for i in range(4)], test_labels,name_str,location)


# Save the history files for later usage in other scripts and plot results
base_str_multi = base + "_multiview_" + fusiontype
create_history_plot(history,name_str,base_str_multi)

batch_data = test_data
batch_labels = test_labels

poe_str_lin = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "lin.csv"
poe_str_log = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "log.csv"
poe_str_quan = "PerformanceOverEnergy/PerformanceOverEnergy_" + formatted_datetime + "_" + fusiontype + "quan.csv"
# Adjust the intensity bin as needed
mean_accuracy = calculate_mean_accuracy_in_intensity_bins(model_multi, batch_data, batch_labels,0,bin_max,nr_bins,poe_str_lin)
mean_accuracy = calculate_mean_accuracy_in_intensity_bins_log(model_multi, batch_data, batch_labels,150,bin_max,nr_bins,poe_str_log)
mean_accuracy = calculate_mean_accuracy_in_intensity_bins_quantile(model_multi, batch_data, batch_labels,nr_bins,poe_str_quan)
print(mean_accuracy)
# print(f"Mean Validation Accuracy in Intensity Bin {intensity_bin}: {mean_accuracy}")


