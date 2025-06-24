# STREAMLIT ML CLASSIFICATION APP - DUAL MODEL SUPPORT
# =====================================================
#sources:
# https://chatgpt.com/share/685a1cd9-fbe8-8002-b8dd-aa0349c93e7a


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
    if uploaded_file.name.endswith('.pdf'):
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif uploaded_file.name.endswith('.docx'):
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])
    else:
        return uploaded_file.read().decode('utf-8')

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
    """Make prediction using the selected model"""
    if models is None:
        return None, None

    try:
        prediction = None
        probabilities = None

        if model_choice == "svm":
            if models.get('vectorizer_available') and models.get('svm_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['svm'].predict(X)[0]
                probabilities = models['svm'].predict_proba(X)[0]

        elif model_choice == "decision_tree":
            if models.get('vectorizer_available') and models.get('dt_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['decision_tree'].predict(X)[0]
                probabilities = models['decision_tree'].predict_proba(X)[0]

        elif model_choice == "adaboost":
            if models.get('vectorizer_available') and models.get('ab_available'):
                X = models['vectorizer'].transform([text])
                prediction = models['adaboost'].predict(X)[0]
                probabilities = models['adaboost'].predict_proba(X)[0]

        if prediction is not None and probabilities is not None:
            # Convert to readable format
            class_names = ['Human', 'AI']  # Update label names to match your project
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
    
    if models:
        available_models = get_available_models(models)
        
        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
            )
            
            # Text input
            user_input = st.text_area(
                "Enter your text here:",
                placeholder="Type or paste your text here...",
                height=150,
                key="text_input"
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
                            st.session_state.text_input = example
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
                help="Upload a .txt, .csv, .pdf, or .docx file. CSV must have text in first column."
            )

            if uploaded_file:
                # Model selection
                model_choice = st.selectbox(
                    "Choose model for batch processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(model[1] for model in available_models if model[0] == x)
                )

                if st.button("📊 Process File"):
                    try:
                        # Handle different file types
                        if uploaded_file.name.endswith('.txt'):
                            content = str(uploaded_file.read(), "utf-8")
                            texts = [line.strip() for line in content.split('\n') if line.strip()]
                        elif uploaded_file.name.endswith('.csv'):
                            df = pd.read_csv(uploaded_file)
                            texts = df.iloc[:, 0].astype(str).tolist()
                        elif uploaded_file.name.endswith('.pdf') or uploaded_file.name.endswith('.docx'):
                            extracted_text = extract_text_from_file(uploaded_file)
                            texts = [para.strip() for para in extracted_text.split('\n') if para.strip()]
                        else:
                            st.error("Unsupported file format.")
                            texts = []

                        if not texts:
                            st.error("No text found in file.")
                        else:
                            st.info(f"Processing {len(texts)} texts...")
                            results = []
                            progress_bar = st.progress(0)

                            for i, text in enumerate(texts):
                                prediction, probabilities = make_prediction(text, model_choice, models)

                                if prediction and probabilities is not None:
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

                                # Summary
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

                                # Results preview
                                st.subheader("📋 Results Preview")
                                st.dataframe(results_df[['Text', 'Prediction', 'Confidence']], use_container_width=True)

                                # Download button
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

            if st.button("📊 Compare All Models") and comparison_text.strip():
                st.subheader("🔍 Model Comparison Results")

                comparison_results = []

                for model_key, model_name in available_models:
                    prediction, probabilities = make_prediction(comparison_text, model_key, models)

                    if prediction and probabilities is not None:
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

                    # Probability comparison
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
            st.info("Only one model available. Use Single Prediction page for detailed analysis.")
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
            ### 📈 Logistic Regression
            **Type:** Linear Classification Model
            **Algorithm:** Logistic Regression with L2 regularization
            **Features:** TF-IDF vectors (unigrams + bigrams)
            
            **Strengths:**
            - Fast prediction
            - Interpretable coefficients
            - Good baseline performance
            - Handles sparse features well
            """)
            
        with col2:
            st.markdown("""
            ### 🎯 Multinomial Naive Bayes
            **Type:** Probabilistic Classification Model
            **Algorithm:** Multinomial Naive Bayes
            **Features:** TF-IDF vectors (unigrams + bigrams)
            
            **Strengths:**
            - Fast training and prediction
            - Works well with small datasets
            - Good performance on text classification
            - Natural probabilistic outputs
            """)
        
        # Feature engineering info
        st.subheader("🔤 Feature Engineering")
        st.markdown("""
        **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
        - **Max Features:** 5,000 most important terms
        - **N-grams:** Unigrams (1-word) and Bigrams (2-word phrases)
        - **Min Document Frequency:** 2 (terms must appear in at least 2 documents)
        - **Stop Words:** English stop words removed
        """)
        
        # File status
        st.subheader("📁 Model Files Status")
        file_status = []
        
        files_to_check = [
            ("sentiment_analysis_pipeline.pkl", "Complete LR Pipeline", models.get('pipeline_available', False)),
            ("tfidf_vectorizer.pkl", "TF-IDF Vectorizer", models.get('vectorizer_available', False)),
            ("logistic_regression_model.pkl", "LR Classifier", models.get('lr_available', False)),
            ("multinomial_nb_model.pkl", "NB Classifier", models.get('nb_available', False))
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
        **Dataset:** Product Review Sentiment Analysis
        - **Classes:** Positive and Negative sentiment
        - **Preprocessing:** Text cleaning, tokenization, TF-IDF vectorization
        - **Training:** Both models trained on same feature set for fair comparison
        """)
        
    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# ============================================================================
# HELP PAGE
# ============================================================================

elif page == "❓ Help":
    st.header("❓ How to Use This App")
    
    with st.expander("🔮 Single Prediction"):
        st.write("""
        1. **Select a model** from the dropdown (Logistic Regression or Multinomial Naive Bayes)
        2. **Enter text** in the text area (product reviews, comments, feedback)
        3. **Click 'Predict'** to get sentiment analysis results
        4. **View results:** prediction, confidence score, and probability breakdown
        5. **Try examples:** Use the provided example texts to test the models
        """)
    
    with st.expander("📁 Batch Processing"):
        st.write("""
        1. **Prepare your file:**
           - **.txt file:** One text per line
           - **.csv file:** Text in the first column
        2. **Upload the file** using the file uploader
        3. **Select a model** for processing
        4. **Click 'Process File'** to analyze all texts
        5. **Download results** as CSV file with predictions and probabilities
        """)
    
    with st.expander("⚖️ Model Comparison"):
        st.write("""
        1. **Enter text** you want to analyze
        2. **Click 'Compare All Models'** to get predictions from both models
        3. **View comparison table** showing predictions and confidence scores
        4. **Analyze agreement:** See if models agree or disagree
        5. **Compare probabilities:** Side-by-side probability charts
        """)
    
    with st.expander("🔧 Troubleshooting"):
        st.write("""
        **Common Issues and Solutions:**
        
        **Models not loading:**
        - Ensure model files (.pkl) are in the 'models/' directory
        - Check that required files exist:
          - tfidf_vectorizer.pkl (required)
          - sentiment_analysis_pipeline.pkl (for LR pipeline)
          - logistic_regression_model.pkl (for LR individual)
          - multinomial_nb_model.pkl (for NB model)
        
        **Prediction errors:**
        - Make sure input text is not empty
        - Try shorter texts if getting memory errors
        - Check that text contains readable characters
        
        **File upload issues:**
        - Ensure file format is .txt or .csv
        - Check file encoding (should be UTF-8)
        - Verify CSV has text in the first column
        """)
    
    # System information
    st.subheader("💻 Your Project Structure")
    st.code("""
    streamlit_ml_app/
    ├── app.py                              # Main application
    ├── requirements.txt                    # Dependencies
    ├── models/                            # Model files
    │   ├── sentiment_analysis_pipeline.pkl # LR complete pipeline
    │   ├── tfidf_vectorizer.pkl           # Feature extraction
    │   ├── logistic_regression_model.pkl  # LR classifier
    │   └── multinomial_nb_model.pkl       # NB classifier
    └── sample_data/                       # Sample files
        ├── sample_texts.txt
        └── sample_data.csv
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 App Information")
st.sidebar.info("""
**ML Text Classification App**
Built with Streamlit

**Models:** 
- 📈 Logistic Regression
- 🎯 Multinomial Naive Bayes

**Framework:** scikit-learn
**Deployment:** Streamlit Cloud Ready
""")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666666;'>
    Built with ❤️ using Streamlit | Machine Learning Text Classification Demo | By Maaz Amjad<br>
    <small>As a part of the courses series **Introduction to Large Language Models/Intro to AI Agents**</small><br>
    <small>This app demonstrates sentiment analysis using trained ML models</small>
</div>
""", unsafe_allow_html=True)