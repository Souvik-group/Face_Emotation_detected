import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Image Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'archive/train',                # ✅ Corrected path
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",         # FER dataset is grayscale
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'archive/test',                 # ✅ Corrected path
    target_size=(48, 48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical'
)

# ✅ Build CNN Model
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))   # 7 emotions

# ✅ Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,                        # You can increase later
    validation_data=validation_generator,
    validation_steps=len(validation_generator)
)

# ✅ Save Model
model.save("emotion_model.h5")
print("✅ Model trained and saved as emotion_model.h5")
