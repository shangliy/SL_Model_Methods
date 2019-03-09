"""
Basic nets
"""

def bilstm_test():
    model = Sequential()
    input_shape = (149, 40)
    model.add(Bidirectional(LSTM(units=20, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    model.add(TimeDistributed(Dense(1, activation='sigmoid')))
    # model.add(Dense(1, activation='sigmoid'))

    # LSTM参数个数计算：ht-1与xt拼接、隐藏单元数、四个门的bias
    #                    （20+40）*units*4+20*4
    #
    #
    batch_size = 64
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_training, Y_training,
              batch_size=batch_size,
              epochs=30,
              validation_data=(x_test, y_test),
              verbose=1)