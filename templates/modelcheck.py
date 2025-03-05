import tensorflow as tf

model = tf.keras.models.load_model('C://Users//Mayuri//Downloads//plant-disease-detect//PlantVillage_model.h5')
print(model.summary())  # Check input layer details
