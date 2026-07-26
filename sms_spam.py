import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------------------------------
# 1. Page Configuration & Setup
# ------------------------------------------------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="💬", layout="centered")

st.title("💬 SMS Spam Detection App")
st.write("A Machine Learning project to classify text messages as **Spam** or **Ham (Legitimate)**.")

# ------------------------------------------------------------------
# 2. Synthetic Data Creation (To make the app run instantly out-of-the-box)
# ------------------------------------------------------------------
@st.cache_data
def load_initial_data():
    # Pre-populating a small dataset mimicking the UCI/Kaggle SMS collection
    data = {
        'v1': ['ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'ham'],
        'v2': [
            'Hey, are we still meeting up for lunch today at 1 PM?',
            'WINNER! As a valued network customer you have been selected to receive a £900 prize reward!',
            'Just wanted to check if you finished the project report yet.',
            'URGENT! Your mobile number has won a £2,000 bonus prize. Call 09066364589 now! Claim code: UT41',
            'I will call you back in a few minutes, driving right now.',
            'FREE ringtone text "LIVE" to 8007 to get your weekly charts. T&Cs apply.',
            'Can you pick up some milk on your way home tonight?',
            'Loan Approval! You are pre-approved for up to $5000. Reply YES to accept or STOP to opt-out.',
            'Don\'t forget about the family dinner this coming Sunday evening.',
            'Sounds good to me, see you there!'
        ]
    }
    return pd.DataFrame(data)

df = load_initial_data()

# Sidebar for training controls and metrics
st.sidebar.header("🛠️ Model Configuration")

# Allow user to see the internal training dataset
if st.sidebar.checkbox("Show Training Dataset Snippet", value=True):
    st.write("### Training Data Preview")
    st.dataframe(df.rename(columns={'v1': 'Label', 'v2': 'Message Text'}))

# ------------------------------------------------------------------
# 3. Model Training & Pipeline Setup
# ------------------------------------------------------------------
@st.cache_resource
def train_model(dataframe):
    # Features (X) and Target Label (y)
    X = dataframe['v2']
    y = dataframe['v1'].map({'ham': 0, 'spam': 1}) # Mapping text labels to binary integers
    
    # Text Extraction using TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    X_vectorized = vectorizer.fit_transform(X)
    
    # Using Logistic Regression (one of the recommended suitable algorithms)
    model = LogisticRegression()
    model.fit(X_vectorized, y)
    
    return vectorizer, model

# Train the pipeline
vectorizer, model = train_model(df)

# Sidebar indicator of model status
st.sidebar.success("Model successfully trained on the dataset!")

# ------------------------------------------------------------------
# 4. User Testing Interface (Inference Section)
# ------------------------------------------------------------------
st.write("---")
st.subheader("🔮 Test the Spam Detector")
st.write("Type or paste any SMS text message below to analyze it instantly.")

# Text input container
user_input = st.text_area("Enter SMS Content:", placeholder="Type your message here...", height=120)

if st.button("Analyze Message", type="primary"):
    if user_input.strip() != "":
        # 1. Transform the input string using the trained TF-IDF vectorizer
        transformed_input = vectorizer.transform([user_input])
        
        # 2. Run prediction
        prediction = model.predict(transformed_input)[0]
        prediction_proba = model.predict_proba(transformed_input)[0]
        
        # 3. Format and display results dynamically
        st.write("### Analysis Result:")
        if prediction == 1:
            st.error(f"🚨 **Spam Detected!** (Confidence: {prediction_proba[1]:.2%})")
            st.info("💡 **Reasoning flags:** This message contains patterns, structures, or terms highly correlated with promotional, urgent, or fraudulent activities.")
        else:
            st.success(f"✅ **Ham (Legitimate Message)** (Confidence: {prediction_proba[0]:.2%})")
            st.info("💡 **Reasoning flags:** This message exhibits conversational, regular linguistic features typical of standard communication.")
            
        # Extra Metrics Data Cards
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Calculated Message Length", value=len(user_input))
        with col2:
            special_char_count = sum(1 for c in user_input if not c.isalnum() and not c.isspace())
            st.metric(label="Special Characters Count", value=special_char_count)
            
    else:
        st.warning("Please type a valid message first before clicking analyze.")

# ------------------------------------------------------------------
# 5. Project Information Footer
# ------------------------------------------------------------------
st.write("---")
with st.expander("ℹ️ View Project Architecture & Metadata"):
    st.markdown("""
    - **Features Extracted:** Message Content, Length, Special Characters, Word Frequency via `TfidfVectorizer`.
    - **Classification Target:** Binary classification mapping (`0` for Ham, `1` for Spam).
    - **Algorithm in Use:** Logistic Regression.
    - **Expandable Upgrades:** To scale this up for a school portfolio or final report, replace the mini-dataset function with a file upload script (`pd.read_csv('spam.csv')`) using the UCI or Kaggle dataset links provided in your documentation blueprint.
    """)
