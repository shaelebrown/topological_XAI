# Explainable AI with topology

Neural networks learn complicated, and often unexplainable, patterns in data, and in this demo I show a novel approach for using topological data analysis  to explain these complex patterns. I will post my code once it is clean enough to be comprehensible!

## Data and models

For this example I will be interpreting the predictions of a [facial emotion recognition (FER) transformer model](https://www.google.com/search?client=safari&rls=en&q=motheecreator%2Fvit-Facial-Expression-Recognition&ie=UTF-8&oe=UTF-8) from Hugging Face. This model predicts which of seven emotions (happy, sad, neutral, angry, disgusted, afraid or surprised) is most likely for a given image, and the model was finetuned on several datasets including [this image dataset](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data?resource=download) which we will use as input to the model. Here are a few examples of images and their emotion labels:

![](original_data.png | width=100)

In order to explain the FER model predictions in the context of specific facial I used a [facial image segmentation model](https://huggingface.co/jonathandinu/face-parsing) from Hugging Face to locate facial features. Once I made hypotheses about what calculations the FER model was carrying out, I used an [additional dataset](https://github.com/amrta-coder/LFW-emotion-dataset?tab=readme-ov-file) of masked faces (with happy and neutral emotions) to evaluate my hypotheses.

## Topological approach to explainability

The FER model learned patterns relating images to emotion labels, and these patterns exist within final layer activities. The steps I carried out to explain these complex patterns were:

1. Sampling data from the image dataset and calculated the final layer activities for those images.
2. Using the package [`tdads`](https://tdads.readthedocs.io/en/latest/) I identified significant topological features (loops and voids) that existed within these activity patterns - features which captured complex non-linear patterns learned by the FER model.
3. Destroying these learned patterns by modifying images on these topological features using model gradients.

## Hypotheses

Most of the image modifications were removing colours like red and yellow from the eyes, nose, mouth or hair: 

![](image_modifications.png | width=100)

Therefore I tested the hypotheses that the model had learned relationships between the emotion label and the

1. Eyes/nose/mouth/hair,
2. image colours, and
3. lower face area.

I tested these hypotheses by determining the effects of

1. Covering (i.e. with black pixels) the eyes/nose/mouth/hair from images (using the facial segmentation model),
2. removing image colours by converting to grayscale and decreasing luminosity by 80%, and
3. predicting the emotions of the facemask dataset.

## Results

![](accuracy_decreases.png | width=100)

The first hypothesis testing resulted in a 15% decrease in model accuracy. This was also the case when I did not cover the hair in images, meaning that the model was looking at the eyes, nose and mouth to partially infer emotion.

The second hypothesis testing resulted in a 35% decrease in model accuracy (both for grayscale and decreasing luminance). This means that the model was looking at colour and luminance to partially infer emotion - images property that should not impact model inference. However, I did not find any racial bias, i.e. a significant decrease in accuracy between caucasian and darker-skinned individuals in the dataset.

The third hypothesis resulted in a 55% decrease in model accuracy. I suspect that similar results would have been observed in a dataset of people wearing sunglasses, although I could not find a suitable one.

## Conclusions

This FER model correctly inferred emotion from the eyes, nose and mouth in images, and was therefore susceptible to misclassification when presented with images of masked people. The FER model incorrectly inferred emotion from image luminance/colour, making the model susceptible to misclassification when presented with nighttime images. The model should be improved by finetuning with masked/partially occluded images and darker images to improve generalization. 

This topological framework was a powerful XAI tool in this case, and I'm excited to try it out for other use-cases.
