from shiny import App, reactive, render, ui
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd
from joblib import load
import math

model = load_model('rnnmodel.keras')
scaler = load('scaler.joblib')

# UI
app_ui = ui.page_fluid(
    ui.markdown(
        """
        ## Burnout Symptoms Prediction App
        """
    ),
    ui.page_sidebar(
        ui.sidebar(ui.h3("Baseline"),
                   ui.input_slider("TL0", "Training Load at baseline", 0, 168, 0),
                         ui.input_numeric("EXH0", "Exhaustion at baseline", min=0, max=4, value=0),
                         ui.input_numeric("DEV0", "Devaluation at baseline", min=0, max=4, value=0),
                         ui.input_numeric("RSA0", "Reduced Sense of Accomplishment at baseline", min=0, max=4, value=0),
                         ui.input_numeric("PS0", "Physical Symptoms at baseline", min=7, max=126, value=7),
                         ui.input_numeric("IS0", "Illness Symptoms at baseline", min=0, max=70, value=0),
                         ui.input_numeric("PSQI0", "Sleep Disruptions at baseline", min=0, max=21, value=0),
                         ui.input_numeric("LS0", "Life Satisfaction at baseline", min=5, max=35, value=5),
                         ui.h3("Timepoint 1"),
                         ui.input_slider("TL1", "Training Load at timepoint 1", 0, 168, 0),
                         ui.input_numeric("EXH1", "Exhaustion at timepoint 1", min=0, max=4, value=0),
                         ui.input_numeric("DEV1", "Devaluation at timepoint 1", min=0, max=4, value=0),
                         ui.input_numeric("RSA1", "Reduced Sense of Accomplishment at timepoint 1", min=0, max=4, value=0),
                         ui.input_numeric("PS1", "Physical Symptoms at timepoint 1", min=7, max=126, value=7),
                         ui.input_numeric("IS1", "Illness Symptoms at timepoint 1", min=0, max=70, value=0),
                         ui.input_numeric("PSQI1", "Sleep Disruptions at timepoint 1", min=0, max=21, value=0),
                         ui.input_numeric("LS1", "Life Satisfaction at timepoint 1", min=5, max=35, value=5),
                         ui.input_action_button("btn", "Predict")),
        ui.markdown(
            """
            ## Model Prediction
            """
        ),
        ui.output_text_verbatim("txt", placeholder=True),),
    )

# Server
def server(input, output, session):
    @reactive.Effect
    @reactive.event(input.btn)
    def data():
        #data to dataframe
        testset = pd.DataFrame([[input.EXH0(), input.EXH1(), input.DEV0(), input.DEV1(), 
                                 input.RSA0(), input.RSA1(), input.TL0(), input.TL1(),
                                 input.PS0(), input.PS1(), input.LS0(), input.LS1(), 
                                 input.PSQI0(), input.PSQI1(), input.IS0(), input.IS1()]], 
                                 columns=['EXH_1', 'EXH_2', 'DEV_1', 'DEV_2', 'RSA_1', 'RSA_2', 'trainload_1', 'trainload_2', 
                                          'PS_1', 'PS_2', 'LS_1', 'LS_2', 'PSQIg_1', 'PSQIg_2', 'WURSSg_1', 'WURSSg_2'], dtype=float)
        #scale and reshape data
        testset = scaler.transform(testset)
        testset = testset.reshape((testset.shape[0], 2, 8))
        testset = tf.data.Dataset.from_tensor_slices(testset).batch(1)
        #predictions
        for data in testset:
            predictions = model(data)
            print(predictions)
        predexh = math.floor(predictions[0][0]*100)/100
        preddev = math.floor(predictions[0][1]*100)/100
        predrsa = math.floor(predictions[0][2]*100)/100
        #output
        @output
        @render.text
        @reactive.event(input.btn)
        def txt():
            return f'Predicted exhaustion is: {predexh} \nPredicted devaluation is {preddev} \nPredicted reduced sense of accomplishment is {predrsa}'

# App
app = App(app_ui, server)
