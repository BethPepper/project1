# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================
#sources:
# https://chatgpt.com/share/685a1cd9-fbe8-8002-b8dd-aa0349c93e7a
#https://chatgpt.com/share/685aee73-bc9c-8002-ae26-f7b19274666c
#https://chatgpt.com/share/685b2cab-0d0c-8002-86a4-9245f4a85bd6

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pdfplumber
import docx

# Page Configuration
st.set_page_config(
    page_title="ML Text Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_file(uploaded_file):
    filename = uploaded_file.name.lower()

    if filename.endswith('.pdf'):
        import pdfplumber
        with pdfplumber.open(uploaded_file) as pdf:
            texts = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    texts.append(text)
            return "\n".join(texts)

    elif filename.endswith('.docx'):
        import docx
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    else:
        try:
            return uploaded_file.read().decode('utf-8')
        except UnicodeDecodeError:
            return ""


@st.cache_resource
def load_models():
    models = {}

    try:
        # Load TF-IDF vectorizer
        try:
            models['vectorizer'] = joblib.load('models/tfidf_vectorizer.pkl')
            models['vectorizer_available'] = True
        except FileNotFoundError:
            models['vectorizer_available'] = False

        # SVM model
        try:
            models['svm'] = joblib.load('models/svm_model.pkl')
            models['svm_available'] = True
        except FileNotFoundError:
            models['svm_available'] = False

        # Decision tree model
        try:
            models['decision_tree'] = joblib.load('models/decision_tree_model.pkl')
            models['dt_available'] = True
        except FileNotFoundError:
            models['dt_available'] = False

        # AdaBoost model
        try:
            models['adaboost'] = joblib.load('models/adaboost_model.pkl')
            models['ab_available'] = True
        except FileNotFoundError:
            models['ab_available'] = False

        # Final check
        individual_ready = models.get('vectorizer_available', False) and (
            models.get('svm_available', False) or
            models.get('dt_available', False) or
            models.get('ab_available', False)
        )

        if not individual_ready:
            st.error("No complete model setup found!")
            return None

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def make_prediction(text, model_choice, models):
    if models is None:
        return None, None

    try:
        # Vectorize input text
        vectorizer = models.get('vectorizer')
        if vectorizer is None:
            st.error("Vectorizer not loaded")
            return None, None

        text_vector = vectorizer.transform([text])  # This produces a csr_matrix

        prediction = None
        probabilities = None

        if model_choice == "svm" and models.get('svm_available'):
            prediction = models['svm'].predict(text_vector)[0]
            probabilities = models['svm'].predict_proba(text_vector)[0]

        elif model_choice == "decision_tree" and models.get('dt_available'):
            prediction = models['decision_tree'].predict(text_vector)[0]
            probabilities = models['decision_tree'].predict_proba(text_vector)[0]

        elif model_choice == "adaboost" and models.get('ab_available'):
            prediction = models['adaboost'].predict(text_vector)[0]
            probabilities = models['adaboost'].predict_proba(text_vector)[0]

        if prediction is not None and probabilities is not None:
            class_names = ['Human', 'AI']  # adjust if needed
            prediction_label = class_names[prediction]
            return prediction_label, probabilities
        else:
            return None, None

    except Exception as e:
        st.error(f"Error making prediction: {e}")
        st.error(f"Model choice: {model_choice}")
        st.error(f"Available models: {[k for k, v in models.items() if isinstance(v, bool) and v]}")
        return None, None



def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    if models.get('vectorizer_available') and models.get('svm_available'):
        available.append(("svm", "📈 SVM"))

    if models.get('vectorizer_available') and models.get('dt_available'):
        available.append(("decision_tree", "🌳 Decision Tree"))

    if models.get('vectorizer_available') and models.get('ab_available'):
        available.append(("adaboost", "⚡ AdaBoost"))

    return available

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================

st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("Choose what you want to do:")

page = st.sidebar.selectbox(
    "Select Page:",
    ["🏠 Home", "🔮 Single Prediction", "📁 Batch Processing", "⚖️ Model Comparison", "📊 Model Info", "❓ Help"]
)

# Load models
models = load_models()


# ============================================================================
# HOME PAGE
# ============================================================================

if page == "🏠 Home":
    st.markdown('<h1 class="main-header">🤖 AI vs Human Text Classifier</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to your machine learning web application! This app demonstrates **AI vs Human text detection**
    using three trained models: **SVM**, **Decision Tree**, and **AdaBoost**.
    """)
    
    # App overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔮 Single Prediction
        - Enter or upload text
        - Choose a model
        - Get instant prediction and confidence
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload text/PDF/Word files
        - Process multiple inputs
        - Export results
        """)
    
    with col3:
        st.markdown("""
        ### ⚖️ Model Comparison
        - Compare all models on the same text
        - Visualize agreement/disagreement
        - See probability scores side-by-side
        """)
    
    # Model status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get('svm_available') and models.get('vectorizer_available'):
                st.info("**📈 SVM**\n✅ Available")
            else:
                st.warning("**📈 SVM**\n❌ Not Available")

        with col2:
            if models.get('dt_available') and models.get('vectorizer_available'):
                st.info("**🌳 Decision Tree**\n✅ Available")
            else:
                st.warning("**🌳 Decision Tree**\n❌ Not Available")

        with col3:
            if models.get('ab_available') and models.get('vectorizer_available'):
                st.info("**⚡ AdaBoost**\n✅ Available")
            else:
                st.warning("**⚡ AdaBoost**\n❌ Not Available")

    else:
        st.error("❌ Models not loaded. Please check the model files.")


# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================

elif page == "🔮 Single Prediction":
    st.header("🔮 Make a Single Prediction")
    st.markdown("Enter text below and select a model to see if it was human or AI written.")

    # Create separate session variable to control the text area value
    if "text_input_value" not in st.session_state:
        st.session_state["text_input_value"] = ""

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )

            # Text input field (NOTE: uses a different key now!)
            user_input = st.text_area(
                "Enter your text here:",
                value=st.session_state["text_input_value"],
                height=150,
                key="text_area"
            )

            if user_input:
                st.caption(f"Character count: {len(user_input)} | Word count: {len(user_input.split())}")

            # Example texts
            with st.expander("📝 Try these example texts"):
                examples = [
                    "Artificial intelligence models are capable of generating coherent and complex text.",
                    "I had such a great time at the concert last night. Everything felt so real and personal.",
                    "ChatGPT can assist with coding, writing, and creative projects by generating useful suggestions.",
                    "We took our dog to the lake, and she was so happy splashing in the water all day!"
                ]
                col1, col2 = st.columns(2)
                for i, example in enumerate(examples):
                    with col1 if i % 2 == 0 else col2:
                        if st.button(f"Example {i+1}", key=f"example_{i}"):
                            st.session_state["text_input_value"] = example
                            st.rerun()

            # Prediction button
            if st.button("🚀 Predict", type="primary"):
                if user_input.strip():
                    with st.spinner('Analyzing text...'):
                        prediction, probabilities = make_prediction(user_input, model_choice, models)

                        if prediction and probabilities is not None:
                            # Display prediction
                            col1, col2 = st.columns([3, 1])

                            with col1:
                                if prediction == "Human":
                                    st.success(f"🧑 Prediction: **{prediction} Written**")
                                else:
                                    st.error(f"🤖 Prediction: **{prediction} Written**")

                            with col2:
                                confidence = max(probabilities)
                                st.metric("Confidence", f"{confidence:.1%}")

                            # Probability details
                            st.subheader("📊 Prediction Probabilities")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("🧑 Human", f"{probabilities[0]:.1%}")
                            with col2:
                                st.metric("🤖 AI", f"{probabilities[1]:.1%}")

                            # Bar chart
                            class_names = ['Human', 'AI']
                            prob_df = pd.DataFrame({
                                'Class': class_names,
                                'Probability': probabilities
                            })
                            st.bar_chart(prob_df.set_index('Class'), height=300)
                        else:
                            st.error("Failed to make prediction.")
                else:
                    st.warning("Please enter some text to classify!")
        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================
# BATCH PROCESSING PAGE
# ============================================================================

elif page == "📁 Batch Processing":
    st.header("📁 Upload File for Batch Processing")
    st.markdown("Upload a text, CSV, PDF, or Word document to classify multiple texts.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['txt', 'csv', 'pdf', 'docx'],
                help="Upload a .txt, .csv, .pdf, or .docx file. CSV must have text in first column.",
                key="batch_file_uploader"
            )

            if uploaded_file:
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x),
                    key="batch_model_choice"
                )

                if st.button("📊 Process File", key="process_file_button"):
                    try:
                        filename = uploaded_file.name.lower()

                        if filename.endswith('.txt'):
                            content = uploaded_file.read().decode("utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]

                        elif filename.endswith('.csv'):
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file)
                            if df.shape[1] < 1:
                                st.error("CSV file has no columns.")
                                texts = []
                            else:
                                texts = df.iloc[:, 0].astype(str).tolist()

                        elif filename.endswith('.pdf') or filename.endswith('.docx'):
                            uploaded_file.seek(0)
                            extracted_text = extract_text_from_file(uploaded_file) or ""
                            texts = [para.strip() for para in extracted_text.split('\n') if para.strip()]

                        else:
                            st.error("Unsupported file format.")
                            texts = []

                        if not texts:
                            st.error("No text found in the uploaded file.")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                prediction, probabilities = make_prediction(text, model_choice, models)

                                if prediction is not None and probabilities is not None:
                                    results.append({
                                        'Text': text[:100] + "..." if len(text) > 100 else text,
                                        'Full_Text': text,
                                        'Prediction': prediction,
                                        'Confidence': f"{max(probabilities):.1%}",
                                        'Human_Prob': f"{probabilities[0]:.1%}",
                                        'AI_Prob': f"{probabilities[1]:.1%}"
                                    })

                                progress_bar.progress((i + 1) / len(texts))

                            if results:
                                st.success(f"✅ Processed {len(results)} texts successfully!")
                                results_df = pd.DataFrame(results)

                                st.subheader("📊 Summary Statistics")
                                col1, col2, col3, col4 = st.columns(4)

                                human_count = sum(1 for r in results if r['Prediction'] == 'Human')
                                ai_count = len(results) - human_count
                                avg_conf = np.mean([float(r['Confidence'].strip('%')) for r in results])

                                with col1:
                                    st.metric("Total Processed", len(results))
                                with col2:
                                    st.metric("🧑 Human", human_count)
                                with col3:
                                    st.metric("🤖 AI", ai_count)
                                with col4:
                                    st.metric("Avg Confidence", f"{avg_conf:.1f}%")

                                st.subheader("📋 Results Preview")
                                st.dataframe(results_df[['Text', 'Prediction', 'Confidence']], use_container_width=True)

                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Full Results",
                                    data=csv,
                                    file_name=f"predictions_{model_choice}_{uploaded_file.name}.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.error("No valid texts could be processed.")

                    except Exception as e:
                        st.error(f"Error processing file: {e}")
            else:
                st.info("Please upload a file to get started.")
                with st.expander("📄 Example File Formats"):
                    st.markdown("""
                    **Text File (.txt):**
                    ```
                    This is a sample human-written sentence.
                    Another line of AI-generated content.
                    ```

                    **CSV File (.csv):**
                    ```
                    text
                    "Human-written comment goes here."
                    "AI might generate this text."
                    ```
                    """)
        else:
            st.error("No models available for batch processing.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================

elif page == "⚖️ Model Comparison":
    st.header("⚖️ Compare Models")
    st.markdown("Compare predictions from different models on the same text.")

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            comparison_text = st.text_area(
                "Enter text to compare models:",
                placeholder="Paste text to evaluate across models...",
                height=100
            )

            if st.button("📊 Compare All Models"):

                if not comparison_text.strip():
                    st.warning("Please enter some text to compare.")
                else:
                    st.subheader("🔍 Model Comparison Results")

                    comparison_results = []

                    for model_key, model_name in available_models:
                        prediction, probabilities = make_prediction(comparison_text, model_key, models)

                        if prediction is not None and probabilities is not None:
                            comparison_results.append({
                                'Model': model_name,
                                'Prediction': prediction,
                                'Confidence': f"{max(probabilities):.1%}",
                                'Human %': f"{probabilities[0]:.1%}",
                                'AI %': f"{probabilities[1]:.1%}",
                                'Raw_Probs': probabilities
                            })

                    if comparison_results:
                        comparison_df = pd.DataFrame(comparison_results)
                        st.table(comparison_df[['Model', 'Prediction', 'Confidence', 'Human %', 'AI %']])

                        predictions = [r['Prediction'] for r in comparison_results]
                        if len(set(predictions)) == 1:
                            st.success(f"✅ All models agree: **{predictions[0]} Written**")
                        else:
                            st.warning("⚠️ Models disagree on classification")
                            for result in comparison_results:
                                st.write(f"- **{result['Model']}** predicted: **{result['Prediction']}**")

                        st.subheader("📊 Detailed Probability Comparison")

                        cols = st.columns(len(comparison_results))

                        for i, result in enumerate(comparison_results):
                            with cols[i]:
                                st.write(f"**{result['Model']}**")
                                chart_data = pd.DataFrame({
                                    'Class': ['Human', 'AI'],
                                    'Probability': result['Raw_Probs']
                                })
                                st.bar_chart(chart_data.set_index('Class'))
                    else:
                        st.error("Failed to get predictions from models.")

        elif len(available_models) == 1:
            st.info("Only one model available. Use the Single Prediction page for detailed analysis.")
        else:
            st.error("No models available for comparison.")
    else:
        st.warning("Models not loaded. Please check the model files.")


# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "📊 Model Info":
    st.header("📊 Model Information")

    if models:
        st.success("✅ Models are loaded and ready!")

        # Model details
        st.subheader("🔧 Available Models")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📈 SVM (Support Vector Machine)
            **Type:** Linear Classification Model  
            **Kernel:** Linear  
            **Features:** TF-IDF vectors (unigrams + bigrams)

            **Strengths:**
            - High performance for text classification
            - Good with high-dimensional data
            - Robust to overfitting
            """)

        with col2:
            st.markdown("""
            ### 🌳 Decision Tree
            **Type:** Non-linear Classification Model  
            **Algorithm:** Gini Index or Entropy  
            **Features:** TF-IDF + text features

            **Strengths:**
            - Easy to interpret
            - Handles non-linear patterns
            - Fast to train
            """)

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### ⚡ AdaBoost
            **Type:** Ensemble Model  
            **Base Estimator:** Decision Tree (depth=1)  
            **Features:** TF-IDF + engineered features

            **Strengths:**
            - Combines weak learners for strong performance
            - Effective for imbalanced datasets
            - Probabilistic output
            """)

        with col2:
            st.markdown("""
            ### 🔤 Feature Engineering
            **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)  
            - Max Features: 5,000  
            - N-grams: Unigrams + Bigrams  
            - Stop Words: Removed (English)  
            - Extra Features (optional): Avg sentence length, punctuation %, etc.
            """)

        # File status
        st.subheader("📁 Model Files Status")
        file_status = []

        files_to_check = [
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("svm_model.pkl", "SVM Classifier", models.get('svm_available', False)),
            ("decision_tree_model.pkl", "Decision Tree Classifier", models.get('dt_available', False)),
            ("adaboost_model.pkl", "AdaBoost Classifier", models.get('ab_available', False))
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        # Training information
        st.subheader("📚 Training Information")
        st.markdown("""
        **Project Goal:** Detect whether a text was written by a human or by AI.  
        **Training Data:** Labeled dataset of human-written and AI-generated texts  
        **Preprocessing:**  
        - Clean and normalize text  
        - TF-IDF vectorization  
        - Optional: custom linguistic features  
        **Evaluation:** Accuracy, precision, recall, F1-score, ROC-AUC  
        **Validation:** 5-fold cross-validation  
        """)
    else:
        st.warning("Models not loaded. Please check model files in the `models/` directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")

    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (SVM, Decision Tree, or AdaBoost)
        2. **Enter text** manually or paste it from any document (essay, blog post, etc.)
        3. **Click 'Predict'** to classify the text as AI or Human-written
        4. **View the results:** prediction label, confidence score, and probability breakdown
        5. **Try example texts** by clicking on the sample buttons
        """)

    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.txt file:** One text per line
           - **.csv file:** Text in the first column
           - **.pdf or .docx:** Multi-paragraph files, text will be auto-extracted
        2. **Upload the file** using the uploader
        3. **Select the model** you'd like to use for classification
        4. **Click 'Process File'** to run predictions on all texts
        5. **Download the results** as a CSV file with predictions and confidence scores
        """)

    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** that you'd like to evaluate
        2. **Click 'Compare All Models'** to see what each model predicts
        3. **Check the comparison table** for prediction, confidence, and probability
        4. **See agreement analysis:** Do all models agree on the label?
        5. **Visualize probabilities** for each model side-by-side
        """)

    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Fixes:**

        **Models not loading:**
        - Confirm all model files are in the `models/` folder
        - Ensure the following files exist:
          - `tfidf_vectorizer.pkl` (required for all models)
          - `svm_model.pkl`, `decision_tree_model.pkl`, `adaboost_model.pkl`

        **Prediction issues:**
        - Make sure the input is valid text (not empty or corrupted)
        - Extremely short texts may give unreliable results

        **File upload issues:**
        - Only `.txt`, `.csv`, `.pdf`, and `.docx` files are accepted
        - CSV files must have the text in the **first column**
        - All files must be UTF-8 encoded
        """)

    st.subheader("💻 Project Structure")
    st.code("""
    ai_human_detection_project/
    ├── app.py                       # Main Streamlit app
    ├── requirements.txt            # Required packages
    ├── models/                     # Saved models
    │   ├── tfidf_vectorizer.pkl
    │   ├── svm_model.pkl
    │   ├── decision_tree_model.pkl
    │   └── adaboost_model.pkl
    ├── data/                       # Training/test data
    ├── notebooks/                 # Jupyter notebook work
    └── README.md                  # Documentation
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**AI vs Human Text Detection App**  
Built with Streamlit

**Models:**  
- 📈 Support Vector Machine (SVM)  
- 🌳 Decision Tree  
- ⚡ AdaBoost  

**Framework:** scikit-learn  
**Deployment:** Streamlit Cloud Ready  
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | AI vs Human Text Classification Project<br>
    <small>Developed for the course project: <strong>Intro to Large Language Models / AI Agents</strong></small><br>
    <small>This app detects whether text was written by a human or generated by AI</small>
</div>
""", unsafe_allow_html=True)
