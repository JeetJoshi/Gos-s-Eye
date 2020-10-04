# Gos-s-Eye

Summary:-

The Mother Nature is always somewhat difficult to understand and is always responsible for existence of life on the earth. So in order to deal with Nature, Human has developed many kind of resources which helps us to understand the different phenomenon in a better way. The Disasters on Earth are greatly responsible for their bad effects on Ecosystem, Economies and Lives. The satellites and it's various sensors are the resources developed by us to deal with this kind of natural disasters. They are capable to generate the required data to predict the formation, area of impact and lifespan of many disasters. Still we require a system which is capable to process that data and automatically predict this all things with the input data collected by the satellite and it's sensors with a great efficiency. To tackle this problem, we have created a Solution which is capable to process the data generated by the satellite and predict the disaster from that data. There are many kind of disasters on earth, some of them have more impact on us and some of them has less impact on us. Cyclones are more common phenomena and also is has a huge impact on our ecosystem, economy and lives. So we have targeted that particular phenomena and with the help of modern technologies like Artificial Intelligence, Machine Learning and Deep Learning, We have developed a Model which is capable of process the satellite images and predict the Cyclone automatically. So now, there is no need to do this observations manually. The Model is itself Capable to handle the satellite data, process it and produce important information about the cyclones.

Features :-

Model is capable of directly process the satellite data.
Model is provide the results in the same images by adding a layer of detection.
The accuracy of the model is about 91.63%
Easy to use
Only single click is required to get the required results from the data.
Get the output in less than 8 seconds.
Use of Open-Source keras and Tensorflow Packages at Backend
How we address this challenge

God's Eye

In order to detect hazard we focused on collecting data from NASA/JAXA satellite based data. We have focused on one particular phenomena (cyclone). We then created a Deep Learning model using inception network to predict hazard. This can help government to take immediate action that can save people's life and they can also try to impede hazard. The main benefit of this model is it is showing particular area where calamity is happening.

Types of Hazard

Biological hazards are of organic origin or conveyed by biological vectors, including pathogenic microorganisms, toxins and bioactive substances. Examples are bacteria, viruses or parasites, as well as venomous wildlife and insects, poisonous plants and mosquitoes carrying disease-causing agents.

Environmental hazards  may include chemical, natural and biological hazards. They can be created by environmental degradation or physical or chemical pollution in the air, water and soil. However, many of the processes and phenomena that fall into this category may be termed drivers of hazard and risk rather than hazards in themselves, such as soil degradation, deforestation, loss of biodiversity, salinization and sea-level rise.

Geological or geophysical hazards  originate from internal earth processes. Examples are earthquakes, volcanic activity and emissions, and related geophysical processes such as mass movements, landslides, rockslides, surface collapses and debris or mud flows. Hydro-meteorological factors are important contributors to some of these processes. Tsunamis are difficult to categorize: although they are triggered by undersea earthquakes and other geological events, they essentially become an oceanic process that is manifested as a coastal water-related hazard.

Hydro-meteorological hazards  are of atmospheric, hydrological or oceanographic origin. Examples are tropical cyclones (also known as typhoons and hurricanes); floods, including flash floods; drought; heatwaves and cold spells; and coastal storm surges. Hydro-meteorological conditions may also be a factor in other hazards such as landslides, wildland fires, locust plagues, epidemics and in the transport and dispersal of toxic substances and volcanic eruption material.

Technological hazards originate from technological or industrial conditions, dangerous procedures, infrastructure failures or specific human activities. Examples include industrial pollution, nuclear radiation, toxic wastes, dam failures, transport accidents, factory explosions, fires and chemical spills. Technological hazards also may arise directly as a result of the impacts of a natural hazard event

How It Works

Here we are trying to recognize hazardous calamities such as fire, hurricane, cyclones volcano- eruption and Storms from all around the world with the help of satellite images. We have created a model with deep learning which recognize a particular phenomenon. We have used inception model for the detection of hazard. It tries to predict the phenomena and it also shows a red box around the hazard. First we are collecting live images from the satellite data and then we are dividing the image into 100 parts. After this deep learning model tries to predict particular phenomena from divided photos and at the end it will show red box highlighting the ground at which it is happening

Impact

A hazard is a process, phenomenon or human activity that may cause loss of life, injury or other health impacts, property damage, social and economic disruption or environmental degradation. Hazards may be natural, anthropogenic or socio-natural in origin. It also affects Global warming and climate change. Effective disaster risk reduction requires the consideration of not just what has occurred but of what could occur. Most disasters that could happen have not yet happened.

How we developed this:

Project definition: Automatic detection of cyclone from the images of the satellite.

Data Gathering: The best source of data for this ask is NASA World View that is available on this events official website. For our project we have collected data from this source in the form of snapshot for training, validating and testing our deep learning model.

Creating model: For this task we have created a deep learning model with transfer learning that take images as input and detect whether there is cyclone or possibility of cyclone or not and display these portion of the world in rectangular box on the map of the world. After Gathering data, we have done data pre-processing task on our data for making it appropriate for our deep learning model. In this task we remove noisy data from dataset, most blurry images that create noise and make deep learning model slow while learning.

For this task, first we have separated data for traing, validation and testing (traing data, validation data, testing data) our model in he appropriate amount.After splitting the data we had used VGG-16 model (one of the deep learning model) along with the transfer learning and trained this model on our training data. Using the validation data we had done our validation of our model. After this we had achieved training accuracy of 94.58% and validation accuracy of 82.3%. Then after validating our model we had tested our model on testing set of data and achieve accuracy of 80.62%. It seemed like we had achieved little bit low accuracy on testing.

After splitting the data we had used VGG-19 model (one of the deep learning model) along with the transfer learning and trained this model on our training data. Using the validation data we had done our validation of our model. After this we had achieved training accuracy of 90.83% and validation accuracy of 81.67%. Then after validating our model we had tested our model on testing set of data and achieve accuracy of 83.38%.

After splitting the data we have used InceptionV3 model (one of the deep learning model) along with the transfer learning and trained this model on our training data. Using the validation data we have done our validation of our model. After this we have achieved training accuracy of 96.29% and validation accuracy of 89.57%. Then after validating our model we have tested our model on testing set of data and achieve accuracy of 91.63% . In this model we have almost achieved our goal.

After comparing all the different model we conclude that InceptionV3 model give us the almost accuracy as same as the our goal. Then we conclude InceptionV3 model the best suitable model for our task.

Resources

NASA Worldview https://worldview.earthdata.nasa.gov/

Global Imagery Browse Services (GIBS) https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-components/gibs

JAXA for Earth http://earth.jaxa.jp/en.html

Pre-Trained Weights for Transfer Learning https://keras.io/api/applications/