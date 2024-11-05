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
        # **Burnout Symptoms Prediction App**
        """
    ),
    ui.page_navbar(
        ui.nav_panel("Home", 
                     ui.markdown(
                         """
                         #### Intention
                        This app was developed with the intention of providing coaches with a tool to predict burnout risk in their athletes.
                        The model takes a number of imput variables at a baseline measurement and a two to three-month follow-up timepoint (timepoint 1).
                        These measurements are then used to predict burnout risk six-months post baseline.
                        #### Input Variables
                        The variables used for prediction include training load (hrs/week), burnout symptoms (measured by the ABQ and scored 0-4), 
                        physical symptoms (measured by the physical symptoms checklist), illness symptoms (measured by the WURSS-g),
                        sleep disruptions (measured by the PSQI), and life satisfaction (measured by the SWLS).
                        See links below for links to each questionnaire.
                        #### Capabilities
                        Note that this is an alpha version and the model will be further developed. 
                        The mean square error for the current model is 0.3, thus the root mean square error is about 0.5.
                        The calculated score for each symptom may therefore deviate by a value of +/- 0.5.
                        Use this app for estimation of risk and to determine potential need for preventative strategies.
                        #### Links and Further Information
                        The questionnaires the model is based on can be found on psycharchives: https://psycharchives.org/en/item/7b941205-9a80-4e1a-90bd-a6ae21c25877
                        - Page 2: Physical Symptoms Checklist (scored 1-7)
                        - Page 5-7: PSQI (see here for scoring: https://www.goodmedicine.org.uk/files/assessment,%20pittsburgh%20psqi.pdf)
                        - Page 8: WURSS-g (scored 0-7; there is another item on page 9 that is the WURSS-c and can be ignored)
                        - Page 9: SWLS (scored 1-7)
                        - Page 10: ABQ (scored 0-4, note item 1 and 14 are reverse scored)
                        """
                     )),
        ui.nav_panel("Model",
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
                                    ui.input_action_button("btn", "Predict"),
                                    width="50%"),
                    ui.markdown(
                        """
                        ## Model Prediction
                        Please provide input to the model on the left. The loading spinner below will appear until input has been provided.

                        Model prediction will appear below:
                        """
                        ),
                    ui.output_text_verbatim("txt", placeholder=True),
                    ui.markdown(
                        """
                        There are currently no cut-offs for burnout in sport. 
                        However, there are some indicators of potential health concerns.
                        - The athlete scores over 3 on a symptom (based on a scale from 0 to 4)
                        - The athlete scores over 3 on multiple symptoms of burnout
                        
                        Consider an implementation of preventative strategies for athletes who may be at risk for burnout.
                        """
                    )))
    ))

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

