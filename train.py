from keras.optimizers import SGD
from Cre_Model import resnet
from keras.utils import plot_model

# load train data and validation data
# train data with shape X (?, 1, 16, 20, 33) Y (?, 6)
X_data =
y_data =
# validation data shape same as train
X_data_2 =
y_data_2 =
model = resnet((1, 16, 20, 33), 6)
model.summary()
plot_model(model, to_file='resnet_2.jpg')

epochs=1000
learningRate=0.01
decay=learningRate/epochs
sgd = SGD(lr=learningRate,decay=decay,momentum=0.9,nesterov=False)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping

# every 10 epochs save weights
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',monitor='val_acc',
                              verbose=1,save_best_only=False,save_weights_only=True,mode='auto',
                              period = 10)
#every epoch check validation accuracy scores and save the highest
checkpoint_2 = ModelCheckpoint('best.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',monitor='val_acc',
                              verbose=1,save_best_only=True,save_weights_only=True,mode='auto',
                              period = 1)

# use tensorboard can watch the change in time
tensorboard_ = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#The training will stop if the validation loss does not decrease for 3 epochs.
history = model.fit(X_data,y_data,validation_data=(X_data_2,y_data_2),verbose=1,
                    batch_size=100,epochs=1000,callbacks=[checkpoint, checkpoint_2, tensorboard_])

