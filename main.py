from flask import Flask, render_template , request, redirect, url_for, make_response,jsonify
from markupsafe import escape
import os
import json
import datetime
from markupsafe import Markup
import time
import transformer_parser


app = Flask(__name__)


def process_json(json_content):
    file_content = ""

    imports = """
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import requests

"""

    file_content = file_content + imports

    if isinstance(json_content, str):
        config = json.loads(json_content)
    else:
        config = json_content

    components = config['architecture']['components']
    connections = config['architecture']['connections']

    connection_map = {}
    for conn in connections:
        connection_map[conn['sourceId']] = conn['targetId']

    data_component = None
    layer_components = []
    
    for comp in components:
        if comp['type'] == 'data-input':
            data_component = comp
        else:
            layer_components.append(comp)

    current_id = data_component['id']
    ordered_layers = []
    
    while current_id in connection_map:
        next_id = connection_map[current_id]
        for comp in layer_components:
            if comp['id'] == next_id:
                ordered_layers.append(comp)
                break
        current_id = next_id

    file_content = file_content + "model = keras.Sequential()\n"

    for i, layer_comp in enumerate(ordered_layers):
        attrs = layer_comp['attributes']
        
        if layer_comp['type'] == 'input-layer':
            continue
        elif layer_comp['type'] == 'dropout-layer':
            dropout_rate = float(attrs.get('rate', 0.2))
            file_content = file_content + f"model.add(layers.Dropout({dropout_rate}))\n"
            continue
            
        neurons = int(attrs['neurons'])
        activation = attrs['activation']
        
        if activation == 'None':
            activation = None
        elif activation == 'ReLU':
            activation = 'relu'
        elif activation == 'Softmax':
            activation = 'softmax'
        elif activation == 'Sigmoid':
            activation = "sigmoid"
        elif activation == "Linear":
            activation = None
        elif activation == "Tanh":
            activation = "tanh"
            
        if i == 1:
            input_neurons = int(ordered_layers[0]['attributes']['neurons'])
            if activation:
                file_content = file_content + f"model.add(layers.Dense({neurons}, activation='{activation}', input_shape=({input_neurons},)))\n"
            else:
                file_content = file_content + f"model.add(layers.Dense({neurons}, activation=None, input_shape=({input_neurons},)))\n"
        else:
            if activation:
                file_content = file_content + f"model.add(layers.Dense({neurons}, activation='{activation}'))\n"
            else:
                file_content = file_content + f"model.add(layers.Dense({neurons}, activation=None))\n"

    train_params = {}
    analytics_params = {}
    webhook_url = ""
    notification_text = ""
    custom_code = ""
    
    for action in config['workflow']['actions']:
        if action['type'] == 'train':
            attrs = action['attributes']
            train_params = {
                'epochs': int(attrs['epochs']),
                'dropout': float(attrs['dropout']),
                'optimizer': attrs['optimizer'].lower()
            }
        elif action['type'] == 'analytics':
            attrs = action['attributes']
            analytics_params = {
                'roc_auc': attrs['rocAuc'] == 'true',
                'f1_score': attrs['f1Score'] == 'true',
                'precision': attrs['precision'] == 'true',
                'recall': attrs['recall'] == 'true',
                'conf_matrix': attrs['confMatrix'] == 'true'
            }
        elif action['type'] == 'notify':
            attrs = action['attributes']
            webhook_url = attrs.get('webhookUrl', '')
            notification_text = attrs.get('notificationText', '')
        elif action['type'] == 'custom':
            attrs = action['attributes']
            custom_code = attrs.get('customCode', '')

    file_content = file_content + f"model.compile(optimizer='{train_params['optimizer']}',loss='sparse_categorical_crossentropy',metrics=['accuracy'])\n"
    file_content = file_content + "model.summary()\n"

    attrs = data_component['attributes']
    filepath = attrs['filepath']
    train_pct = int(attrs['train']) / 100
    test_pct = int(attrs['test']) / 100
    val_pct = int(attrs['val']) / 100
    
    data_code = f"""
df = pd.read_csv('{filepath}')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
    
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size={test_pct}, random_state=42)
    
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size={val_pct}/({1-test_pct}), random_state=42)
    
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

"""

    file_content = file_content + data_code

    analytics_content = f"""
print(f"\\nData splits:")
print(f"Train: {{X_train.shape[0]}} samples")
print(f"Validation: {{X_val.shape[0]}} samples") 
print(f"Test: {{X_test.shape[0]}} samples")

print(f"\\nTraining for {train_params['epochs']} epochs with {train_params['optimizer']} optimizer...")
history = model.fit(
    X_train, y_train,
    epochs={train_params['epochs']},
    validation_data=(X_val, y_val),
    batch_size=32,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"\\nTest Results:")
print(f"Loss: {{test_loss:.4f}}")
print(f"Accuracy: {{test_accuracy:.4f}}")

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(f"\\nAnalytics:")

"""

    file_content = file_content + analytics_content

    if analytics_params.get('f1_score'):
        file_content += """
f1 = f1_score(y_test, y_pred_classes, average='weighted')
print(f"F1 Score: {f1:.4f}")

"""

    if analytics_params.get('precision'):
        file_content += """
precision = precision_score(y_test, y_pred_classes, average='weighted')
print(f"Precision: {precision:.4f}")

"""

    if analytics_params.get('recall'):
        file_content += """
recall = recall_score(y_test, y_pred_classes, average='weighted')
print(f"Recall: {recall:.4f}")

"""

    if analytics_params.get('conf_matrix'):
        file_content += """
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print(f"\\nConfusion Matrix:")
print(conf_matrix)

"""

    if analytics_params.get('roc_auc'):
        file_content += """
try:
    roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')
    print(f"ROC-AUC Score: {roc_auc:.4f}")
except:
    print("ROC-AUC: Cannot compute for this dataset")

"""

    if custom_code:
        file_content += f"""
# Custom Code Execution
try:
    {custom_code}
except Exception as e:
    print(f"Error in custom code: {{e}}")

"""

    if webhook_url:
        model_name = config.get('modelName', '')
        message = notification_text.replace("''", f"'{model_name}'")
        file_content += f"""
# Discord Webhook Notification
try:
    payload = {{
        "content": f"{message} Accuracy: {{test_accuracy:.4f}}, Loss: {{test_loss:.4f}}"
    }}
    requests.post("{webhook_url}", json=payload)
except:
    pass

"""

    file_content += f"""
print(f"\\nModel {config.get('modelName', 'Unknown')} training completed!")
"""

    return file_content

    

@app.route("/", methods=['GET'])
def main_page():
    return render_template('main.html')

@app.route("/learn", methods=['GET'])
def learn_page():
    return render_template('learn.html')

@app.route("/how-to", methods=['GET'])
def how_to_page():
    return render_template('how_to_use.html')

@app.route("/create", methods=['GET'])
def create():
    return render_template('create.html')

@app.route("/mlp-editor/<model_name>", methods=['GET'])
def mlp_editor(model_name):
    return render_template('workspace_mlp.html' , model_name=model_name)

@app.route("/transformer-editor/<model_name>", methods=['GET'])
def transformer_editor(model_name):
    return render_template('workspace_transformers.html' , model_name=model_name)

@app.route("/transformer-editor/prebuilt1/<model_name>", methods=['GET'])
def prebuilt_transformer_1(model_name):
    return render_template('transformers_karpathy.html' , model_name=model_name)

@app.route("/transformer-editor/prebuilt2/<model_name>", methods=['GET'])
def prebuilt_transformer_2(model_name):
    return render_template('transformers_aiaun.html' , model_name=model_name)

@app.route("/generate-code-transformer", methods=['POST'])
def generate_code_transformer():
    raw_data = request.data.decode("utf-8")

    #this is a temporary hack
    #the frontend expects pytorch code as we were planning to make it in pytorch initially
    #however we couldnt and now it uses keras, instead of editing 10 places in the frontend we just changed here
    return jsonify({
            "pytorch_code": transformer_parser.process_json(json.loads(raw_data))
        }) , 200

@app.route("/generate-code-mlp", methods=['POST'])
def generate_code_mlp():
    raw_data = request.data.decode("utf-8")
    

    return jsonify({
            "keras_code": process_json(json.loads(raw_data))
        }) , 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=1000, debug=True) 
