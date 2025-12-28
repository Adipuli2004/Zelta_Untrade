import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import warnings
warnings.filterwarnings("ignore")

# Defining Constants

WINDOW_SIZE=6
COLUMN_NAME = "close"
HORIZON=1
BATCH_SIZE = 1024
N_EPOCHS = 5000
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 30
INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = INPUT_SIZE + HORIZON

class NBeatsBlock(tf.keras.layers.Layer):
  "Class to implement one block of the Nbeats Architechture"

  def __init__(self,input_size: int, theta_size: int, horizon: int, n_neurons: int,n_layers: int,**kwargs):
    super().__init__(**kwargs)
    self.input_size = input_size
    self.theta_size = theta_size
    self.horizon = horizon
    self.n_neurons = n_neurons
    self.n_layers = n_layers
    self.hidden = [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]
    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear", name="theta")

  def call(self, inputs):
    x = inputs
    for layer in self.hidden:
      x = layer(x)
    theta = self.theta_layer(x)
    backcast, forecast = theta[:, :self.input_size], theta[:, -self.horizon:]
    return backcast, forecast

class NBeatsModel():
    def __init__(self,input_size: int, theta_size: int, horizon: int, n_neurons: int,n_layers: int, n_stacks:int):
       
        nbeats_block_layer = NBeatsBlock(input_size=input_size,theta_size=theta_size, horizon=horizon,
                                          n_neurons=n_neurons, n_layers=n_layers,name="InitialBlock")
        stack_input = layers.Input(shape=input_size, name="stack_input")
        backcast, forecast = nbeats_block_layer(stack_input)
        residuals = layers.subtract([stack_input, backcast], name=f"subtract_0")

        for i in range(n_stacks-1):
            backcast, block_forecast = NBeatsBlock(input_size=input_size,theta_size=theta_size, horizon=horizon,
                                          n_neurons=n_neurons, n_layers=n_layers,name=f"Block_{i+1}")(residuals)
            residuals = layers.subtract([residuals, backcast], name=f"subtract_{i+1}")
            forecast = layers.add([forecast, block_forecast], name=f"add_{i+1}")

        self.model = tf.keras.Model(inputs=stack_input,outputs=forecast,name="N-BEATS")
        self.model.compile(loss="mae",optimizer=tf.keras.optimizers.Adam(0.001),metrics=["mae", "mse"])

    def fit(self, train_dataset, val_dataset, epochs: int,):
       
       self.model.fit(train_dataset,
                      epochs=epochs,
                      verbose=0,
                      validation_data=val_dataset,
                      callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                                tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)]
                    )
    
    
    def predict(self, input_data):
        prediction = self.model.predict(input_data)
        return tf.squeeze(prediction)
       
def process_data(data: pd.DataFrame, column: str, window_size:int, batch:int):
    
    # Creating Lagged Features
    data_col = pd.DataFrame(data[column])
    data_col_nbeats = data_col.copy()
    for i in range(window_size):
        data_col_nbeats[f"{column}+{i+1}"] = data_col_nbeats[column].shift(periods=i+1) 

    data_nbeats = data_col_nbeats.dropna()

    X = data_nbeats.dropna().drop(column, axis=1)
    y = data_nbeats.dropna()[column]
    features_dataset = tf.data.Dataset.from_tensor_slices(X)
    labels_dataset = tf.data.Dataset.from_tensor_slices(y)

    split_size = int(len(X) * 0.8)
    X_val, y_val = X[split_size:], y[split_size:]
    val_features_dataset = tf.data.Dataset.from_tensor_slices(X_val)
    val_labels_dataset = tf.data.Dataset.from_tensor_slices(y_val)
    
    train_dataset = tf.data.Dataset.zip((features_dataset, labels_dataset))
    val_dataset = tf.data.Dataset.zip((val_features_dataset, val_labels_dataset))
    test_dataset = tf.data.Dataset.from_tensor_slices(X)
    
    train_dataset = train_dataset.batch(batch).prefetch(tf.data.AUTOTUNE)
    val_dataset = train_dataset.batch(batch).prefetch(tf.data.AUTOTUNE)
    test_dataset = train_dataset.batch(batch).prefetch(tf.data.AUTOTUNE)

    return train_dataset,val_dataset,test_dataset

def main():
    data = pd.read_csv('deep_learning_approach/data/btcusdt_1h_zelta.csv')
    train_dataset,val_dataset,features_dataset = process_data(data=data, column=COLUMN_NAME, window_size=WINDOW_SIZE, batch=BATCH_SIZE)
    model = NBeatsModel(input_size=INPUT_SIZE,
                        theta_size=THETA_SIZE,
                        horizon=HORIZON,
                        n_neurons=N_NEURONS,
                        n_layers=N_LAYERS,
                        n_stacks=N_STACKS)
    model.fit(train_dataset=train_dataset,val_dataset=val_dataset, epochs=N_EPOCHS)
    predictions = model.predict(features_dataset)
    predictions = pd.Series(predictions, index=data[6:].index, name='pred_close')
    predicted_data = pd.concat([data,predictions], axis=1)
    predicted_data.to_csv("deep_learning_approach/predicted_data.csv")

if __name__ == "__main__":
   main()