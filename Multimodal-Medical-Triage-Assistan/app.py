import streamlit as st
import tempfile
import os
import pandas as pd

from Src.main import MedicalMultimodalSystem

st.set_page_config(
    page_title="Medical Multimodal Triage Assistant",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system():
    return MedicalMultimodalSystem(
        cv_model_rel_path="cv_models/effb0_finetuned_best.keras",
        nlp_model_rel_dir="nlp_model"
    )


system = load_system()

st.title("🩺 Medical Multimodal Triage Assistant")
st.write("Upload a skin lesion image and provide a clinical description.")

with st.sidebar:

    st.header("Patient Information")

    uploaded_image = st.file_uploader(
        "Upload Skin Image",
        type=["jpg", "jpeg", "png"]
    )

    text = st.text_area(
        "Clinical Description",
        height=220,
        placeholder="""Example:

Dermatology case: 35 year old male patient.
Lesion location: trunk.
Diagnosis method: histo.
"""
    )

    predict = st.button(
        "🔍 Analyze",
        use_container_width=True
    )

col1, col2 = st.columns([1,1])

with col1:

    if uploaded_image is not None:
        st.subheader("Uploaded Image")
        st.image(uploaded_image, use_container_width=True)

with col2:

    st.subheader("Prediction Result")

    if predict:

        if uploaded_image is None:
            st.error("Please upload an image.")

        elif text.strip() == "":
            st.error("Please enter the clinical description.")

        else:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:

                tmp.write(uploaded_image.read())
                temp_path = tmp.name

            with st.spinner("Analyzing..."):

                prediction, details = system.predict(
                    image_path=temp_path,
                    text=text
                )

            os.remove(temp_path)

            confidence = max(details.values()) * 100

            st.success(f"Prediction: {prediction}")

            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

            st.divider()

            st.subheader("Class Probabilities")

            for label, prob in sorted(details.items(), key=lambda x: x[1], reverse=True):

                st.write(f"**{label}**")
                st.progress(float(prob))
                st.write(f"{prob*100:.2f}%")

            df = pd.DataFrame(
                {
                    "Class": list(details.keys()),
                    "Probability": [v * 100 for v in details.values()]
                }
            )

            st.divider()

            st.subheader("Probability Chart")

            st.bar_chart(
                df.set_index("Class")
            )