# Facial-and-Speech-Emotion-Recognition
Facial and Speech emotion recognition using deep learning 

Facial expression recognition is the part of Facial recognition which is gaining more importance and the need for it increases tremendously. Though there are methods to identify expressions using machine learning and Artificial Intelligence techniques, this work attempts to use deep learning and image classification method to recognize expressions and classify the expressions according to the images. Deep Learning is used for facial expression recognition with the FER-2013 dataset. The final accuracy of this expression recognition model using the MobileNet v2 Model is 93%. 
Modern speech recognition systems have come a long way since their ancient counterparts. They can recognize speech from multiple speakers and have enormous vocabularies in numerous languages. The first component of speech recognition is, of course, speech. Speech must be converted from physical sound to an electrical signal with a microphone, and then to digital data with an analogue-to-digital converter. Once digitized, several models can be used to transcribe the audio to text. CNN is used for speech recognition with the RAVDESS dataset. The final accuracy of this expression recognition model using the CNN Model is 98%.

<h2>Dataset Used:</h3>

<h3>FER-2013 </h3>
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

<h3>RAVDESS</h3>
The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) contains 7356 files (total size: 24.8 GB). The database contains 24 professional actors (12 female, 12 male), vocalizing two lexically matched statements in a neutral North American accent. Speech includes calm, happy, sad, angry, fearful, surprise, and disgust expressions, and the song contains calm, happy, sad, angry, and fearful emotions. Each expression is produced at two levels of emotional intensity (normal, strong), with an additional neutral expression. All conditions are available in three modality formats: Audio only (16bit, 48kHz .wav), Audio-Video (720p H.264, AAC 48kHz, .mp4), and Video-only (no sound).

<h2>Facial Emotion Recognition</h2>
For Facial Emotion Recognition, we usedFER13 dataset, the images were of dimension 48x48, but the Transfer Learning Model which we are using requires an input of 224x224, so we convert the image dimensions. The initial dataset contained 7 classes but the data for the disgust class was not sufficient so we removed that class, the happy dataset was downsampled. The input shape of the model was required to be 4D, so we converted our dataset accordingly.

The deep learning model used for training purposes is a Transfer Learning Model-MobileNetV2(untrained). The MobileNetV2 contains an initially fully convolution layer with 32 filters, followed by 19 residual bottleneck layers. The output layer of the model has 1000 prediction classes, we require 6, so we drop the output layer using Keras layers and add an output layer with softmax function.

We trained the model with 30 epochs and a batch size of 15, the training took 290 minutes and we got a training accuracy of 93%.

For the live implementation of emotion recognition, we used the cv2 library and haar cascade algorithm to crop out the face during the live session. We get the probability of each emotion prediction live.

<h2>Speech Emotion Recognition</h2>
For speech emotion recognition, at first, the data has to be augmented an RAVDESS dataset has studio-recorded audios but in real life audio we get may not have studio-level quality, so we are adding noise, stretch and pitch. After Augmentation, we get a dataset with 3456 rows and 162 columns.

The labelled data is transformed into a numerical label, using the OneHotEncoder function provided by sklearn. This dataset is then split into 3 parts for training, validating and testing. 

The training and validation set is passed to a convolutional neural network with 9 conv layers. After every 3 conv layers, a dropout layer and max-pooling layer is added to reduce the complexity and overfitting. Kernel size of 5 and strides size is set to 2 with activation function relu. Finally, a dense layer is used with output units set to 6 and activation used is softmax. This model is compiled with adam optimizer and categorical_crossentropy loss function. 
The Training was completed in 28 epochs with an average of 3.5 sec/epoch, the batch size used was 15
