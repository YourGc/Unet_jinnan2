# coding:utf-8


from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping,TensorBoard
from keras import backend as K



N_CLS = 5+1
inDir = '/home/n01z3/dataset/dstl'
IMG_SIZE = 640 #8的倍数
SMOOTH = 1e-12
BATCH_SIZE = 8
LogDir = '../logs/20190414'

class Unet():
    def __init__(self,dataset):
        self.IMG_SIZE = IMG_SIZE
        self.batch_size = BATCH_SIZE
        self.smooth = SMOOTH
        self.N_Cls = N_CLS
        self.model = self.get_unet()
        self.dataset = dataset
        self.log_dir = LogDir


    def get_unet(self):
        inputs = Input(shape = (self.IMG_SIZE, self.IMG_SIZE,3))
        conv1 = Conv2D(64, kernel_size = (3,3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, kernel_size = (3,3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, kernel_size = (3,3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, kernel_size = (3,3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, kernel_size = (3,3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, kernel_size = (3,3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, kernel_size = (3,3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, kernel_size = (3,3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, kernel_size = (3,3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, kernel_size = (3,3), activation='relu', padding='same')(conv5)
        up6 = Concatenate()([Conv2DTranspose(filters = 512 , kernel_size=(2, 2) ,strides=(2,2) ,padding='same')(conv5)
                                , conv4])#Cropping2D(cropping=((4,4),(4,4)))(conv4)])

        conv6 = Conv2D(512, kernel_size = (3,3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(512, kernel_size = (3,3), activation='relu', padding='same')(conv6)

        up7 = Concatenate()([Conv2DTranspose(filters = 256 , kernel_size=(2, 2) ,strides=(2,2) ,padding='same')(conv6)
                                , conv3])#Cropping2D(cropping=((16,16),(16,16)))(conv3)])
        conv7 = Conv2D(256, kernel_size = (3,3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(256, kernel_size = (3,3), activation='relu', padding='same')(conv7)

        up8 = Concatenate()([Conv2DTranspose(filters = 128 , kernel_size=(2, 2) ,strides=(2,2) ,padding='same')(conv7)
                                , conv2])#Cropping2D(cropping=((40,40),(40,40)))(conv2)])
        conv8 = Conv2D(128, kernel_size = (3,3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(128, kernel_size = (3,3), activation='relu', padding='same')(conv8)

        up9 = Concatenate()([Conv2DTranspose(filters=64, kernel_size=(2, 2), strides=(2, 2), padding='same')(conv8)
                                , conv1])#Cropping2D(cropping=((88, 88), (88, 88)))(conv1)])
        conv9 = Conv2D(64, kernel_size = (3,3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(64, kernel_size = (3,3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(self.N_Cls, kernel_size = (1,1) , padding='same', activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)
        model.compile(optimizer=Adam(), loss='binary_crossentropy',
                      metrics=[self.jaccard_coef, self.jaccard_coef_int, 'accuracy'])
        model.summary()
        return model

    def jaccard_coef(self,y_true, y_pred):

        intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)

        return K.mean(jac)

    def jaccard_coef_int(self,y_true, y_pred):
        y_pred_pos = K.round(K.clip(y_pred, 0, 1))

        intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
        jac = (intersection + self.smooth) / (sum_ - intersection + self.smooth)
        return K.mean(jac)

    #learning from https://github.com/qqwweee/keras-yolo3/tree/master/yolo3
    def train(self):
        logging = TensorBoard(log_dir= self.log_dir)
        checkpoint = ModelCheckpoint(self.log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                     monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)


        self.model.fit_generator(generator = self.dataset.tranGenerator(),
                            steps_per_epoch=max(1, self.dataset.dataSize // self.batch_size),
                            validation_data=self.dataset.valGenerator(),
                            validation_steps=max(1, self.dataset.dataSize // self.batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint])
        self.model.save_weights(self.log_dir + 'trained_weights_stage_1.h5')

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

        self.model.fit_generator(generator = self.dataset.tranGenerator,
                            steps_per_epoch=max(1, self.dataset.dataSize // self.batch_size),
                            validation_data=self.dataset.valGenerator,
                            validation_steps=max(1, self.dataset.dataSize // self.batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[logging, checkpoint,reduce_lr,early_stopping])
        self.model.save_weights(self.log_dir + 'trained_final.h5')


