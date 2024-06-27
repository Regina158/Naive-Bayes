from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load datasets
data = pd.read_csv('data_training.csv')
test_data = pd.read_csv('data_testing.csv')

# Combine datasets for fitting label encoders
combined_data = pd.concat([data, test_data], ignore_index=True)

# Initialize LabelEncoder
le_jenis = LabelEncoder()
le_merk = LabelEncoder()
le_satuan = LabelEncoder()
le_keterangan = LabelEncoder()

# Function to encode data
def encode_data(df):
    df_encoded = df.copy()
    df_encoded['Jenis Obat'] = le_jenis.transform(df['Jenis Obat'])
    df_encoded['Merk'] = le_merk.transform(df['Merk'])
    df_encoded['Satuan'] = le_satuan.transform(df['Satuan'])
    if 'Keterangan' in df.columns:
        df_encoded['Keterangan'] = le_keterangan.transform(df['Keterangan'])
    return df_encoded

# Fit the encoders with combined data
le_jenis.fit(combined_data['Jenis Obat'])
le_merk.fit(combined_data['Merk'])
le_satuan.fit(combined_data['Satuan'])
le_keterangan.fit(combined_data['Keterangan'])

# Encode training and testing data
data_encoded = encode_data(data)
test_data_encoded = encode_data(test_data)

# Function to evaluate model
def evaluate_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Prediksi
    y_pred = model.predict(X_test)
    
    # Hitung akurasi
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    # Classification report
    report = classification_report(y_test, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return model, accuracy, cv_scores, report, conf_matrix

# Prepare training data
X = data_encoded.drop(columns=['Keterangan', 'Kode', 'Nama Obat'])
y = data_encoded['Keterangan']

# Evaluate and train the model
model, accuracy, cv_scores, report, conf_matrix = evaluate_model(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'password':
            session['logged_in'] = True
            return redirect(url_for('data_obat'))
        else:
            flash('Invalid Credentials', 'error')
    return render_template('login.html')

@app.route('/data_obat')
def data_obat():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('data_obat.html', data=data.to_dict('records'))

@app.route('/data_testing', methods=['GET', 'POST'])
def data_testing():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global data, data_encoded, X, y, model, test_data, test_data_encoded, combined_data
    
    if request.method == 'POST':
        if 'add_new' in request.form:
            try:
                new_data = pd.DataFrame({
                    'Kode': [request.form['kode']],
                    'Nama Obat': [request.form['nama_obat']],
                    'Jenis Obat': [request.form['jenis_obat']],
                    'Merk': [request.form['merk']],
                    'Stok': [int(request.form['stok'])],
                    'Satuan': [request.form['satuan']]
                })
                test_data = pd.concat([test_data, new_data], ignore_index=True)
                combined_data = pd.concat([data, test_data], ignore_index=True)
                test_data.to_csv('data_testing.csv', index=False)
                
                # Re-fit the encoders with the updated combined data
                le_jenis.fit(combined_data['Jenis Obat'])
                le_merk.fit(combined_data['Merk'])
                le_satuan.fit(combined_data['Satuan'])
                le_keterangan.fit(combined_data['Keterangan'])
                
                test_data_encoded = encode_data(test_data)
                flash('New testing data added successfully', 'success')
            except Exception as e:
                flash(f'Error adding new data: {str(e)}', 'error')
    
    return render_template('data_testing.html', data=test_data.to_dict('records'))

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    
    global data, data_encoded, X, y, model, test_data, test_data_encoded, combined_data
    
    # Get the row index from the form
    row_index = int(request.form['row_index'])
    
    # Ensure test data is re-encoded correctly without fitting again
    test_data_encoded = encode_data(test_data)
    
    # Prediction logic for a single row
    X_test = test_data_encoded.drop(columns=['Kode', 'Nama Obat', 'Keterangan']).iloc[[row_index]]
    prediction_encoded = model.predict(X_test)
    
    # Decode prediction
    prediction = le_keterangan.inverse_transform(prediction_encoded)[0]
    
    test_data.at[row_index, 'Prediksi'] = prediction
    
    # Optionally, save the updated test data
    test_data.to_csv('data_testing.csv', index=False)
    
    return redirect(url_for('hasil_prediksi'))

@app.route('/hasil_prediksi')
def hasil_prediksi():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('hasil_prediksi.html', data=test_data.to_dict('records'))

@app.route('/model_metrics')
def model_metrics():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('model_metrics.html', 
                           accuracy=accuracy, 
                           cv_scores=cv_scores, 
                           report=report, 
                           conf_matrix=conf_matrix.tolist())

if __name__ == '__main__':
    app.run(debug=True)