from keras.optimizers import SGD
from Cre_Model import resnet
from keras.utils import plot_model
epochs=1000
learningRate=0.01
decay=learningRate/epochs
model = resnet((1, 16, 20, 33), 6)
model.summary()
plot_model(model, to_file='resnet_2.jpg')

sgd = SGD(lr=learningRate,decay=decay,momentum=0.9,nesterov=False)

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping

checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',monitor='val_acc',
                              verbose=1,save_best_only=False,save_weights_only=True,mode='auto',
                              period = 10)
checkpoint_2 = ModelCheckpoint('best.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}.hdf5',monitor='val_acc',
                              verbose=1,save_best_only=True,save_weights_only=True,mode='auto',
                              period = 1)
#Only save weights for that achieves the best validation accuracy scores at the time

# use tensorboard can watch the change in time
tensorboard_ = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0,
                          embeddings_layer_names=None, embeddings_metadata=None)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
#The training will stop if the validation loss does not decrease for 3 epochs.

history = model.fit(X_data,y_data,validation_data=(X_data_2,y_data_2),verbose=1,
                    batch_size=100,epochs=1000,callbacks=[checkpoint, checkpoint_2, tensorboard_])

