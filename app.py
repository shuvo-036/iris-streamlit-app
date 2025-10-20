# app.py
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")

# -------------------------
# Title, Description, and Image
# -------------------------
st.image("https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg", caption="Iris Flower", use_column_width=True)

st.title("🌸 Iris Flower Classifier")
st.write("""
This app predicts the species of an Iris flower based on its measurements.
Use the sidebar to input the flower’s features, and see the predicted species instantly!
""")
st.info("🔹 Adjust the sliders in the sidebar to set the flower's measurements.\n🔹 The prediction updates automatically.")

# -------------------------
# Sidebar Inputs
# -------------------------
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
    petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.5)
    data = {
        "Sepal Length": sepal_length,
        "Sepal Width": sepal_width,
        "Petal Length": petal_length,
        "Petal Width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# -------------------------
# Load Dataset and Train Model
# -------------------------
iris = load_iris()
X = iris.data
Y = iris.target

model = RandomForestClassifier()
model.fit(X, Y)

# -------------------------
# Prediction
# -------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

species_dict = {0: "Setosa 🌱", 1: "Versicolor 🌿", 2: "Virginica 🌸"}
predicted_species = species_dict[prediction[0]]

# -------------------------
# Display Prediction
# -------------------------
st.subheader("Prediction Result")
st.success(f"The predicted Iris species is: **{predicted_species}**")

st.subheader("Prediction Probability")
proba_df = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.write(proba_df)

# -------------------------
# Display Input Features
# -------------------------
st.subheader("Input Features")
st.write(input_df)

# -------------------------
# Feature Visualization
# -------------------------
st.subheader("Feature Visualization")
st.bar_chart(input_df.T)

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.write("Developed with ❤️ using Streamlit")
