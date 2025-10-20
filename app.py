
import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier


st.set_page_config(page_title="🌸 Iris Flower Classifier", layout="centered")

st.title("🌸 Iris Flower Classifier")
st.write("""
This app predicts the species of an Iris flower based on its measurements.
Adjust the sliders below to set the flower’s features and see the predicted species instantly!
""")

st.info("🔹 Slide the bars to set the flower's measurements.\n🔹 The prediction updates automatically.")

st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.0)
    sepal_width  = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.5)
    petal_width  = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 0.5)
    data = {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()


iris = load_iris()
X = iris.data
Y = iris.target

model = RandomForestClassifier()
model.fit(X, Y)

# -------------------------
# Make Prediction
# -------------------------
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

species_dict = {0: "Setosa 🌱", 1: "Versicolor 🌿", 2: "Virginica 🌸"}
predicted_species = species_dict[prediction[0]]

st.subheader("Prediction Result")
st.success(f"The predicted Iris species is: **{predicted_species}**")

st.subheader("Prediction Probability")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

# -------------------------
# Display Input Features
# -------------------------
st.subheader("Input Features")
st.write(input_df)

# -------------------------
# Optional: Display Feature Bar Chart
# -------------------------
st.subheader("Feature Visualization")
st.bar_chart(input_df.T)
