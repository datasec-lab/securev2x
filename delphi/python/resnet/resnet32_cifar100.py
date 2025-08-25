#!/usr/bin/env python
import argparse
import os
import pickle
import resnet32_model
from os import path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

num_classes = 100

LR_SCHEDULE = [
    (0.1, 91), (0.01, 136), (0.001, 182)
]

class Cifar100Model():
    def _read_data(self):
        from tensorflow.keras.datasets import cifar100
        # The data, split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()

        # Convert class vectors to binary class matrices.
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)

        x_train = x_train.astype("float32") / 255
        x_test = x_test.astype("float32") / 255

        # Split the training set into 20% validation set
        x_val = x_train[:10000]
        y_val = y_train[:10000]
        x_train = x_train[10000:]
        y_train = y_train[10000:]

        return (x_train, y_train), (x_test, y_test), (x_val, y_val)

    def _build_model(self):
        model = resnet32_model.build()
        opt = tf.keras.optimizers.SGD(lr=self.lr, momentum=self.mom, clipvalue=2.0)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=opt,
            metrics=["accuracy"])
        self.model = model

    def _setup(self, config):
        self.config = config
        self.train_data, self.test_data, self.val_data = self._read_data()
        self.reset = False
        self.lr = config["lr"]
        self.mom = config["mom"]

        self.epoch = 0
        self.performance = 0
        self.path = "./pretrained/model_checkpoints"
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        self._build_model()

    def _train(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        x_val, y_val = self.val_data

        aug_gen = ImageDataGenerator(
            featurewise_center=False,
            samplewise_center=False,
            featurewise_std_normalization=False,
            samplewise_std_normalization=False,
            zca_whitening=False,
            rotation_range=0,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=False,
        )
        aug_gen.fit(x_train)
        gen = aug_gen.flow(x_train, y_train, batch_size=self.config["batch_size"])

        def learning_decay(current_epoch):
            initial_lr = self.lr
            lr = initial_lr
            for mult, start_epoch in LR_SCHEDULE:
                if current_epoch >= start_epoch:
                  lr = initial_lr * mult
                else:
                  break
            return lr

        lrate = LearningRateScheduler(learning_decay)
        self.model.fit_generator(
            generator=gen,
            steps_per_epoch=40000 // self.config["batch_size"],
            epochs=self.epoch + self.config["epochs"],
            validation_data=(x_val, y_val),
            callbacks=[lrate],
            initial_epoch=self.epoch)

        self.epoch += self.config["epochs"]
        loss, accuracy = self.model.evaluate(x_val, y_val, verbose=0)
        _, test_accuracy = self.model.evaluate(x_test, y_test, verbose=0)
        
        self._save(self.path)
        
        return {
            "accuracy": accuracy,
            "learning_rate": self.lr,
            "loss": loss,
            "accuracy_test": test_accuracy,
            "momentum": self.mom,
            "path": self.path,
            "epoch": self.epoch,
        }

    def _save(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        to_save = {
                'epoch': self.epoch,
                }
        with open(file_path + "_state", "wb") as f:
            f.write(pickle.dumps(to_save))
        self.model.save_weights(file_path, save_format='h5')
        return file_path

if __name__ == "__main__":
    config = {
        "epochs": 2,
        "batch_size": 128,
        "mom": 0.9,
        "lr": 0.01,
    }

    model_instance = Cifar100Model()
    model_instance._setup(config)
    results = model_instance._train()