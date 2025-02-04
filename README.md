Détection de la Tumeur du Cerveau - Brain Tumor Detection using Deep Learning - January 2025


Le but de l'expérience est d'être capable de prédire si il y a tumeur ou non-tumeur du cerveau tout en utilisant des techniques de Deep Learning en analysant les différentes caractéristiques propres à chacun des cas binaires.

Voici ci-dessous un exemple de tumeur au cerveau:
![image](https://github.com/user-attachments/assets/cb266336-b02a-4e67-ad22-d450af87a6a7)


L'intelligence artificielle (IA) joue un rôle de plus en plus crucial dans le domaine médical, notamment pour la lecture et l'interprétation des images issues de l'imagerie par résonance magnétique (IRM). Grâce à des algorithmes sophistiqués et à l'apprentissage profond (deep learning), l'IA peut analyser ces images avec une précision et une rapidité impressionnantes. Elle est capable de détecter des anomalies subtiles, comme des tumeurs, des lésions ou des signes précoces de maladies, qui pourraient échapper à l'œil humain. De plus, l'IA permet de réduire les temps d'attente pour les diagnostics et d'offrir des analyses plus standardisées, limitant ainsi les erreurs humaines. En complément du travail des radiologues, elle devient un outil précieux pour améliorer la prise en charge des patients et optimiser les décisions médicales. En somme, l'IA révolutionne l'imagerie médicale en rendant les diagnostics plus rapides, plus précis et plus accessibles.

Artificial intelligence (AI) is playing an increasingly crucial role in the medical field, particularly in reading and interpreting images from magnetic resonance imaging (MRI). Thanks to sophisticated algorithms and deep learning, AI can analyze these images with impressive precision and speed. It is capable of detecting subtle anomalies, such as tumors, lesions or early signs of disease, which could escape the human eye. In addition, AI helps reduce waiting times for diagnoses and offers more standardized analyses, thus limiting human errors. In addition to the work of radiologists, it is becoming a valuable tool for improving patient care and optimizing medical decisions. In short, AI is revolutionizing medical imaging by making diagnoses faster, more accurate and more accessible.

Voici un exemple de cycle de vie en Deep Learning pour répondre au besoin de l'expérience.


![image](https://github.com/user-attachments/assets/bcdf89c0-7b0a-41ce-b38d-0610bde2c5f6)

NOUS UTILISONS VGG16 POUR L'APPRENTISSAGE PAR TRANSFERT.¶

Le modèle est construit sur VGG16, qui est un réseau neuronal convolutionnel (CNN) pré-entraîné pour la classification d'images.

Tout d'abord, le modèle VGG16 est chargé avec input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet'. La forme d'entrée est définie pour correspondre à la taille des images dans l'ensemble de données, qui est de 128x128 pixels. Le paramètre include_top est défini sur False, ce qui signifie que les couches finales entièrement connectées de VGG16 qui effectuent la classification ne seront pas incluses. Le paramètre weights est défini sur « imagenet », ce qui signifie que le modèle sera pré-entraîné avec un ensemble de données de 1,4 million d'images appelé imagenet

Ensuite, la boucle for layer in base_model.layers: est utilisée pour définir toutes les couches du modèle de base (VGG16) comme non entraînables, afin que les poids de ces couches ne soient pas mis à jour pendant l'entraînement.

Ensuite, les trois dernières couches du modèle VGG16 sont définies comme entraînables en utilisant base_model.layers[-2].trainable = True,base_model.layers[-3].trainable = True et base_model.layers[-4].trainable = True

Après cela, un modèle séquentiel est créé et le modèle VGG16 lui est ajouté avec model.add(base_model).

Ensuite, une couche Flatten est ajoutée au modèle avec model.add(Flatten()) qui remodèle la sortie du modèle VGG16 d'un tenseur 3D à un tenseur 1D, afin qu'elle puisse être traitée par les couches suivantes du modèle.

Ensuite, une couche Dropout est ajoutée avec model.add(Dropout(0.3)) qui est utilisée pour empêcher le surajustement en définissant aléatoirement une fraction d'unités d'entrée à 0 à chaque mise à jour pendant le temps d'entraînement.

Après cela, une couche dense est ajoutée avec 128 neurones et la fonction d'activation relu est ajoutée avec model.add(Dense(128, activation='relu')).

Ensuite, une autre couche Dropout est ajoutée avec model.add(Dropout(0.2))

Enfin, la couche dense de sortie est ajoutée avec un nombre de neurones égal au nombre d'étiquettes uniques et la fonction d'activation 'softmax' est ajoutée avec model.add(Dense(len(unique_labels), activation='softmax')). La fonction d'activation « softmax » est utilisée pour donner une distribution de probabilité sur les classes possibles.

The model is built on top of VGG16, which is a pre-trained convolutional neural network (CNN) for image classification.
First, the VGG16 model is loaded with input_shape=(IMAGE_SIZE,IMAGE_SIZE,3), include_top=False, weights='imagenet'. The input shape is set to match the size of the images in the dataset, which is 128x128 pixels. The include_top parameter is set to False, which means that the final fully-connected layers of VGG16 that perform the classification will not be included. The weights parameter is set to 'imagenet' which means that the model will be pre-trained with a dataset of 1.4 million images called imagenet
Next, the for layer in base_model.layers: loop is used to set all layers of the base_model (VGG16) to non-trainable, so that the weights of these layers will not be updated during training.
Then, the last three layers of the VGG16 model are set to trainable by using base_model.layers[-2].trainable = True,base_model.layers[-3].trainable = True and base_model.layers[-4].trainable = True
After that, a Sequential model is created and the VGG16 model is added to it with model.add(base_model).
Next, a Flatten layer is added to the model with model.add(Flatten()) which reshapes the output of the VGG16 model from a 3D tensor to a 1D tensor, so that it can be processed by the next layers of the model.
Then, a Dropout layer is added with model.add(Dropout(0.3)) which is used to prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.
After that, a dense layer is added with 128 neurons and relu activation function is added with model.add(Dense(128, activation='relu')).
Next, another Dropout layer is added with model.add(Dropout(0.2))
Finally, the output dense layer is added with number of neurons equal to the number of unique labels and 'softmax' activation function is added with model.add(Dense(len(unique_labels), activation='softmax')). The 'softmax' activation function is used to give a probability distribution over the possible classes.

1- Data Collection

![image](https://github.com/user-attachments/assets/ceae7bae-182f-488d-bec1-3ad3b33f705b)

![image](https://github.com/user-attachments/assets/9e38ce9f-ac66-4593-b0c0-e037a87261da)

2- EDA
![image](https://github.com/user-attachments/assets/7ed2a2c1-7d76-4a8b-97d8-a07452f6b451)

![image](https://github.com/user-attachments/assets/63398616-6d99-44ec-9e60-da6921ba8bb8)

![image](https://github.com/user-attachments/assets/d16ef246-953e-497b-9e11-d8a0136ca9f3)

![image](https://github.com/user-attachments/assets/b14aa53b-41b4-42ac-bb51-d5bbfe1eafdd)

Visualisation de 10 images aléatoirement dans le Dataset
![image](https://github.com/user-attachments/assets/57fe690a-f1f6-4f4c-8052-fdeeb87abe49)


3- Data Processing
![image](https://github.com/user-attachments/assets/cb3ec35e-b8ff-4a71-9d19-1c8d4d82554a)

4- Build Model
![image](https://github.com/user-attachments/assets/298f9489-f325-4d4c-996d-38155d6f622f)

5- Compile and Train Model
![image](https://github.com/user-attachments/assets/8618d321-957d-40ba-9831-0adbc0b1adab)

6- Visualisation of the Model Performance
![image](https://github.com/user-attachments/assets/19f35d90-600f-471d-94b1-aae54c8f9e1e)

7- Prediction on Test Data
![image](https://github.com/user-attachments/assets/67b5eb97-1ab2-49f7-85a7-4bc71d5c01d3)

8- Confusion Matrix
![image](https://github.com/user-attachments/assets/d7bbfc81-a805-4c87-b8a6-754defe5d9f6)

9- Courbe ROC par classe
![image](https://github.com/user-attachments/assets/2030cbbf-91d8-4256-aa15-b1e34d024dff)

10- Save and Load Model
![image](https://github.com/user-attachments/assets/f00072bd-d076-4959-96b2-1651493a08aa)

11- Prediction
![image](https://github.com/user-attachments/assets/0f79eb62-d8a4-411e-961c-bc3e7ea84e35)

![image](https://github.com/user-attachments/assets/7bded313-1a93-474a-aca1-a06a40c4ceec)

![image](https://github.com/user-attachments/assets/4e1c4962-c080-4e6d-a362-738692bda24e)

![image](https://github.com/user-attachments/assets/d5d20375-3313-4b21-b842-70bcad32da6c)

