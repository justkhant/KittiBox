# This Branch...

Changes made to the original repositories allows more input types to be used for training, other than the normal, RGB, Kitti images. Currently, event data, with 9 channels, and grayscale images, with 1 channel, are supported. Demo.py has also been changed so labels can be generated for RGB images. 

## Requirements

The code requires **Tensorflow 1.11.0** and **Python 2**, as well as the following python libraries: 

* matplotlib
* numpy
* Pillow
* scipy
* runcython

Those modules can be installed using: `pip install numpy scipy pillow matplotlib runcython` or `pip install -r requirements.txt`.

## Setup

1. Clone this repository: `git clone https://github.com/justkhant/KittiBox.git`
2. Switch to the`khantk_grasp_merged` branch: `git checkout khantk_grasp_merged' 
2. Go into the `submodules` directory: `cd submodules` 
3. Clone the TensorVision repository in this folder and switch it to the `khantk_grasp_merged` branch:
    - `git clone https://github.com/justkhant/TensorVision.git`
    - `git checkout khantk_grasp_merged`
4. Do the same with the tensorflow_fcn repository: 
    - `git clone https://github.com/justkhant/tensorflow-fcn`
    - `git checkout khantk_grasp_merged`
3. Run `cd submodules/utils && make` to build cython code
4. Now, returning to the KittiBox root directory, download `vgg.16.npy` using: `wget ftp://mi.eng.cam.ac.uk/pub/mttt2/models/vgg16.npy` and place it in a new directory named `DATA`. 
5. [Optional] Download Kitti Object Detection Data 
    1. Retrieve Kitti data url here: [http://www.cvlibs.net/download.php?file=data_object_image_2.zip](http://www.cvlibs.net/download.php?file=data_object_image_2.zip)
    2. Call `python download_data.py --kitti_url URL_YOU_RETRIEVED`
6. [Optional] Run `cd submodules/KittiObjective2/ && make` to build the Kitti evaluation code (see [submodules/KittiObjective2/README.md](submodules/KittiObjective2/README.md) for more information)

Running `demo.py` does not require step 4. and step 5. Those steps are only required if you want to train your own model using `train.py`.

## Training Your Own Data

If you already have your own training data that you want to use, how should the data be structured?

First, create a directory named `training` in the `data` directory. This is where all the relevant training data will be stored. 
For instance, if you have a `image_2` directory containing the images, a `label_2` directory containing labels, and a `calib` directory containing the calibrations, the folder structure should look like this:
```
KittBox/
   ...
   data/
      train.txt
      ...
      training/
         -> image_2 
         -> label_2
         -> calib
```
If you also have event data with the same labels/calib, you can move it into `training` as well:
```
KittBox/
   ...
   data/
      train.txt 
      ...
      training/
         -> events
         image_2 
         label_2
         calib
```
Now, open the `train.txt` file located in the `data` directory. This text file is used to specify the data to be passed into the dataloader. You should see two columns of text, one for the paths to each training image, and one for the paths to each label. By default, the paths should be pointing to images. 
```
training/image_2/000000.png training/label_2/000000.txt
training/image_2/000001.png training/label_2/000001.txt
training/image_2/000002.png training/label_2/000002.txt
training/image_2/000003.png training/label_2/000003.txt
```
If you want to change the data, you can either edit this existing file, or use this as a template to be used to make your own train text files. I prefer the latter, because if you want to switch between different data, it's going to be a pain to keep changing the same text file. It's much simpler to have different ones and change which text file to use instead. 

Make sure the paths to the data are correct. For instance, if I wanted to use my event data instead of images, I would make an `event_train.txt` file like this: 
```
training/events/000000_data.hdf5 training/label_2/000000.txt
training/events/000001_data.hdf5 training/label_2/000001.txt
training/events/000002_data.hdf5 training/label_2/000002.txt
training/events/000003_data.hdf5 training/label_2/000003.txt
```
Honestly there's a lot of flexibility in where you can put the data because you can edit the paths to match wherever it is. But, using the given structure is recommended. Next, is how to specify which `train.txt` file to use:

Open the configuration file `hypes/kittiBox.json`. This is the default config file that contains global parameters that the network uses. As a side note, we can actaully make our own hypes file as well and use that instead, but in our case, it is much easier to just edit the default. 

In the code, go to the following [section]( https://github.com/justkhant/KittiBox/blob/284152f16a0611f87453c080c73f6dcf9983f6ee/hypes/kittiBox.json#L13): 
```
...
"data": {
        "train_file": "../data/train.txt",
        "val_file": "../val.txt",
        "truncate_data": false,
        "eval_cmd": "../submodules/KittiObjective2/./evaluate_object2",
        "label_dir": "KittiBox/training/label_2"
    },
...
```
Change the `train_file` parameter to whatever text file you want to use. For instance, to use my event data, I would change the param to this: 
`"train_file": "../data/event_train.txt"` 

Your data is now ready to train! Note that this txt file switching happens automatically in `train.py`. 

## Training your data

Run `python2 train.py` to train the network using normal RGB Kitti images

Run `python2 train.py --input_type EVENT` to train the network using events from .hdf5 files 

Run `python2 train.py --input_type GRAYSCALE` to train the network using black and white images, by converting the RGB Kitti images. 

Let's take a closer look at this [code](https://github.com/justkhant/KittiBox/blob/284152f16a0611f87453c080c73f6dcf9983f6ee/train.py#L73) here in `train.py` to see how the switching works:

```
 hypes["input_type"] = tf.app.flags.FLAGS.input_type
    if (hypes["input_type"] == 'COLOR'):
        hypes["input_channels"] = 3        
    elif (hypes["input_type"] == 'GRAYSCALE'):
        hypes["input_channels"] = 1
        hypes["dirs"]["output_dir"] = 'RUNS/grayscale_box' 
    elif (hypes["input_type"] == 'EVENT'):
        hypes["input_channels"] = 9
        hypes["input_file"] = '../inputs/event_data_loader.py'
        hypes["data"]["train_file"] = 'data/event_train.txt'
        hypes["dirs"]["output_dir"] = 'RUNS/events_box'
    else:
        logging.error("data_type {} not supported.".format(hypes["input_type"]))
        exit(1)
```
The input_type directly changes the `input_file`, `input_channels`, `train_file`, and `output_dir` params in `hypes/kittiBox.json`. Note that the default parameters in config file is set to `COLOR` mode. 
Here are descriptions of the relevant parameters:
   - The `input_file` param sets the dataloader that we use. Right now, there are two different loaders, one for events and one for images
   - The `input_channels` param changes the number of input channels the first layer of the training network has.   
   - The `train_file` param dictates which text file we use for the training data. The names of the different text files are hardcoded,    but if the text files are named something different, you can go change this directly in this file. 
   - The `log_dir` param dictates the directory where the model is saved it as it trains. Like the `train_file` param, `log_dir` changes automatically as well, but you can directly change the code if you want to save the model in a different directory than the ones that have been hardcoded. 

The `log_dir` plays a important role in the training the network, as I will explain in the following section.

## Continue training

After we begin training with train.py, _kssh_ kicks you out after a certain amout of time. So, we need a way to pick up the training right where it left off. Running `train.py` again would restart the training, so in this case we will use a different script:

Run `python2 submodules/TensorVision/bin/tv-continue.py --logdir RUNS/color_box` to continue training for color images

Run `python2 submodules/TensorVision/bin/tv-continue.py --logdir RUNS/grayscale_box --input_type GRAYSCALE` to continue training for b&w 

Run `python2 submodules/TensorVision/bin/tv-continue.py --logdir RUNS/event_box --input_type EVENT` to continue training for events 

`tv-continue.py` finds the specified box that the model has been saved in and continues the training using previously trained weights and network architecture. Therefore, `logdir` must align with the specific box each input type is save in. If you used custom `log_dir` boxes, then you must match them accordingly. Also, The input type has to be the type first used to train the network.  

##Evaluating Data using a Model
Run `python2 demo.py --input_images path/to/images/ --output_dir path/to/outdir --save_images_dir path/to/savedir` to evaluate images using the pretrained Kitti Detection Model. Using the `--output_dir` and `--save_boxes_dir` will save the images with bounding boxes drawn on them, and save the coordinates of the boxes in text files. 

Here are the flags in more detail: 
1. `--input_images` = Used to specify what the input images are. Right now supports both single image files as well as directories containing multiple images. It also supports an hdf5 file that contains raw images in the following structure: `hdf5["davis"]["left"]["image_raw"]. 
2. `--output_dir` = Use this flag to save images with the bounding boxes drawn on them. The specified directory must exist.  
3. `--save_boxes_dir` = Use this flag to save to boxes to label files. The text files follow the same format as Kitti labels. The specified directory must exist.
4. `--logdir` = Use this flag if you want to evaluate images using a model that you have trained instead of the pretrained Kitti Detection Model. All you have to do is specify the box that the model has been saved in. For instance, if i want to evaluate images using a network trained with grayscale images, I would run `python2 demo.py --logdir RUNS/grayscale_box --input_image path/to/images`. 



