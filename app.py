import nltk
nltk.download('stopwords')
import streamlit as st
import joblib
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import PyPDF2
import pandas as pd
import re
import openpyxl
import plotly.graph_objects as go

# ---- Page Config ----
st.set_page_config(page_title="Sentiment Analyzer", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            padding: 0.5rem 1rem;
            text-align: center;
            background: rgba(17, 17, 17, 0.9);
        }
        main .block-container {
            padding-bottom: 4rem;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            color: white;
            text-align: center;
        }
        .positive-card {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        }
        .negative-card {
            background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- Load Model & Vectorizer ----
@st.cache_resource
def load_model():
    model = joblib.load('sentiment_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

# ---- PDF Processing Functions ----
def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            # page.extract_text() can return None, guard it
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None

# ---- Excel Processing Functions ----
def extract_reviews_from_excel(excel_file):
    """Extract reviews from Excel file"""
    try:
        df = pd.read_excel(excel_file)
        
        # Display available columns
        st.info(f"📊 Found columns: {', '.join(df.columns.tolist())}")
        
        # Try to find review column automatically
        review_column = None
        possible_names = ['review', 'text', 'comment', 'feedback', 'description', 'content', 'message']
        
        for col in df.columns:
            if any(name in col.lower() for name in possible_names):
                review_column = col
                break
        
        if review_column is None:
            review_column = df.columns[0]
            st.warning(f"⚠️ Auto-detected first column '{review_column}' as review column. You can change this in settings.")
        else:
            st.success(f"✅ Auto-detected '{review_column}' as review column")
        
        return df, review_column
    except Exception as e:
        st.error(f"Error reading Excel file: {str(e)}")
        return None, None

def split_into_reviews(text, delimiter="\n"):
    """Split text into individual reviews based on delimiter"""
    reviews = [review.strip() for review in text.split(delimiter) if review.strip()]
    
    # If reviews are too long, try to split by sentences
    processed_reviews = []
    for review in reviews:
        if len(review) > 1000:
            sentences = re.split(r'(?<=[.!?])\s+', review)
            processed_reviews.extend([s.strip() for s in sentences if len(s.strip()) > 20])
        elif len(review) > 20:
            processed_reviews.append(review)
    
    return processed_reviews

def analyze_sentiment(text):
    """Analyze sentiment of a single text"""
    X = vectorizer.transform([text])
    
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)[0][1]
    else:
        prob = model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-prob))
    
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob * 100 if prob >= 0.5 else (1 - prob) * 100
    
    return sentiment, confidence, prob

def display_results(reviews, show_wordcloud, show_individual):
    """Display analysis results for reviews"""
    results = []
    positive_count = 0
    negative_count = 0
    all_positive_text = ""
    all_negative_text = ""
    
    progress_bar = st.progress(0)
    total_reviews = len(reviews) if reviews else 1
    for i, review in enumerate(reviews):
        sentiment, confidence, prob = analyze_sentiment(review)
        results.append({
            'Review #': i + 1,
            'Review Text': review[:100] + "..." if len(review) > 100 else review,
            'Full Text': review,
            'Sentiment': sentiment,
            'Confidence': f"{confidence:.2f}%",
            'Confidence_Val': confidence
        })
        
        if sentiment == "Positive":
            positive_count += 1
            all_positive_text += " " + review
        else:
            negative_count += 1
            all_negative_text += " " + review
        
        # update progress as integer 0-100
        progress_bar.progress(int(((i + 1) / total_reviews) * 100))
    
    total = len(reviews)
    positive_percent = (positive_count / total) * 100 if total else 0
    negative_percent = (negative_count / total) * 100 if total else 0
    
    # ---- Display Summary Statistics ----
    st.markdown("---")
    st.subheader("📊 Overall Statistics")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    
    with col_stat1:
        st.markdown(f"""
        <div class="metric-card">
            <h2>{total}</h2>
            <p>Total Reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown(f"""
        <div class="metric-card positive-card">
            <h2>{positive_count}</h2>
            <p>😊 Positive Reviews</p>
            <h3>{positive_percent:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown(f"""
        <div class="metric-card negative-card">
            <h2>{negative_count}</h2>
            <p>😞 Negative Reviews</p>
            <h3>{negative_percent:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # ---- Pie Chart & Bar Chart ----
    st.markdown("---")
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Animated Pie Chart
        fig_pie = go.Figure(
            data=[go.Pie(
                labels=['Positive', 'Negative'],
                values=[positive_count, negative_count],
                hole=0.4,
                marker=dict(colors=['#38ef7d', '#f45c43']),
                textinfo='label+percent',
                insidetextorientation='radial'
            )]
        )
        fig_pie.update_layout(
            title=dict(
                text="Sentiment Distribution",
                x=0.5,
                font=dict(size=20, color='white')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", y=-0.2, x=0.3),
            transition=dict(duration=800, easing="cubic-in-out")
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
    
    with col_chart2:
        # Animated Bar Chart
        fig_bar = go.Figure(
            data=[go.Bar(
                x=['Positive', 'Negative'],
                y=[positive_count, negative_count],
                marker_color=['#38ef7d', '#f45c43'],
                text=[f"{positive_count}", f"{negative_count}"],
                textposition='outside'
            )]
        )
        fig_bar.update_layout(
            title=dict(
                text="Sentiment Count Comparison",
                x=0.5,
                font=dict(size=20, color='white')
            ),
            yaxis_title="Number of Reviews",
            xaxis_title="Sentiment Type",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            transition=dict(duration=800, easing="cubic-in-out")
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar": False})
    
    # ---- Word Clouds ----
    if show_wordcloud:
        st.markdown("---")
        st.subheader("☁️ Word Clouds by Sentiment")
        
        col_wc1, col_wc2 = st.columns(2)
        
        if all_positive_text.strip():
            with col_wc1:
                st.markdown("**Positive Reviews**")
                wc_pos = WordCloud(width=400, height=300, 
                                  background_color='white',
                                  colormap='Greens').generate(all_positive_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wc_pos, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
        
        if all_negative_text.strip():
            with col_wc2:
                st.markdown("**Negative Reviews**")
                wc_neg = WordCloud(width=400, height=300,
                                  background_color='white',
                                  colormap='Reds').generate(all_negative_text)
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(wc_neg, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close(fig)
    
    # ---- Detailed Results Table ----
     # ---- Detailed Results Table ----
        st.markdown("---")
        st.subheader("📋 Detailed Analysis Results")

        # Build dataframe from results (same as before)
        df = pd.DataFrame(results)

        # Make sure the 'Confidence' column exists and parse numeric values robustly
        if 'Confidence' in df.columns:
            # strip percent sign if present and convert to float
            def parse_conf(v):
                try:
                    s = str(v).strip()
                    if s.endswith('%'):
                        s = s[:-1].strip()
                    return float(s)
                except Exception:
                    return 0.0
            df['ConfNum'] = df['Confidence'].apply(parse_conf)
        else:
            # Fallback: no Confidence column — create zero-confidence column
            df['ConfNum'] = 0.0

        # Prepare display columns (keep same order/labels as before)
        display_cols = ['Review #', 'Review Text', 'Sentiment', 'Confidence']

        # Ensure all display columns exist (guard if something's missing)
        for c in display_cols:
            if c not in df.columns:
                df[c] = ""  # harmless default

        df_display = df[display_cols].copy()

        # Styling function: uses df['ConfNum'] via the row's index (row.name)
        def highlight_by_confidence(row):
            # Get numeric confidence from master df using the row index
            conf = float(df.loc[row.name, 'ConfNum'])
            sentiment = str(row['Sentiment']).lower()

            # compute alpha/transparency from confidence (0.35 -> 0.85)
            alpha = 0.35 + (conf / 100.0) * 0.5
            alpha = max(0.2, min(alpha, 0.9))

            # color selection
            if sentiment == 'positive':
                # green tone
                bg = f'rgba(56,239,125,{alpha})'
                text_color = 'black' if alpha < 0.55 else 'white'
            else:
                # negative tone (red)
                bg = f'rgba(244,92,67,{alpha})'
                text_color = 'black' if alpha < 0.55 else 'white'

            # return a style for every column in df_display
            return [f'background-color: {bg}; color: {text_color}; font-weight: 500;' for _ in row.index]

        # Apply the styler
        styled_df = df_display.style.apply(highlight_by_confidence, axis=1)

        # Show styled dataframe in Streamlit
        st.dataframe(styled_df, use_container_width=True, height=400)
  
    # ---- Individual Reviews (Optional) ----
    if show_individual:
        st.markdown("---")
        st.subheader("🔍 Individual Review Details")
        for result in results:
            with st.expander(f"Review #{result['Review #']} - {result['Sentiment']} ({result['Confidence']})"):
                st.write(result['Full Text'])
    
    # ---- Download Results ----
    st.markdown("---")
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name="sentiment_analysis_results.csv",
        mime="text/csv",
    )

# ---- App Header ----
st.title("💬 Advanced Sentiment Analysis App")
st.markdown("Analyze sentiment from **manual text input**, **PDF files**, or **Excel files** containing multiple reviews!")

# ---- Tabs for Different Input Methods ----
tab1, tab2, tab3 = st.tabs(["📝 Text Input", "📄 PDF Upload", "📊 Excel Upload"])

# ============ TAB 1: Manual Text Input ============
with tab1:
    st.header("Manual Text Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙ Settings")
        show_wordcloud_text = st.checkbox("Show wordcloud", value=False, key="wc_text")
        confidence_meter_text = st.checkbox("Show confidence meter", value=True, key="conf_text")
    
    with col1:
        user_input = st.text_area("🖊 Enter your review or text below:", height=200, key="text_input")
        
        if st.button("🔍 Analyze Sentiment", key="analyze_text"):
            if user_input.strip():
                sentiment, conf_percent, prob = analyze_sentiment(user_input)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    st.markdown(f"### {emoji} Sentiment: **{sentiment}**")
                with col_b:
                    st.markdown(f"### 📊 Confidence: **{conf_percent:.2f}%**")
                
                if confidence_meter_text:
                    st.progress(int(conf_percent))
                
                if show_wordcloud_text:
                    st.subheader("☁️ Word Cloud")
                    wc = WordCloud(width=800, height=400, background_color='white').generate(user_input)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("⚠ Please enter some text to analyze.")

# ============ TAB 2: PDF Upload ============
with tab2:
    st.header("PDF Review Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙ Settings")
        delimiter_option = st.selectbox(
            "Review separator",
            ["Single line break", "Double line break", "Custom"],
            key="delimiter"
        )
        
        if delimiter_option == "Custom":
            custom_delimiter = st.text_input("Enter custom delimiter:", value="---")
            delimiter = custom_delimiter
        elif delimiter_option == "Double line break":
            delimiter = "\n\n"
        else:
            delimiter = "\n"
        
        show_wordcloud_pdf = st.checkbox("Show combined wordcloud", value=True, key="wc_pdf")
        show_individual_pdf = st.checkbox("Show individual reviews", value=False, key="show_ind_pdf")
    
    with col1:
        uploaded_file_pdf = st.file_uploader("📤 Upload PDF file with reviews", type=['pdf'], key="pdf_upload")
        
        if uploaded_file_pdf is not None:
            if st.button("🔍 Analyze PDF Reviews", key="analyze_pdf"):
                with st.spinner("Extracting and analyzing reviews..."):
                    pdf_text = extract_text_from_pdf(uploaded_file_pdf)
                    
                    if pdf_text:
                        reviews = split_into_reviews(pdf_text, delimiter)
                        
                        if len(reviews) == 0:
                            st.error("No reviews found in the PDF. Try adjusting the delimiter settings.")
                        else:
                            st.success(f"✅ Found {len(reviews)} reviews in the PDF")
                            display_results(reviews, show_wordcloud_pdf, show_individual_pdf)

# ============ TAB 3: Excel Upload ============
with tab3:
    st.header("Excel Review Analysis")
    
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("⚙ Settings")
        show_wordcloud_excel = st.checkbox("Show combined wordcloud", value=True, key="wc_excel")
        show_individual_excel = st.checkbox("Show individual reviews", value=False, key="show_ind_excel")
    
    with col1:
        uploaded_file_excel = st.file_uploader(
            "📤 Upload Excel file (.xlsx or .xls)", 
            type=['xlsx', 'xls'], 
            key="excel_upload"
        )
        
        if uploaded_file_excel is not None:
            df_excel, detected_column = extract_reviews_from_excel(uploaded_file_excel)
            
            if df_excel is not None:
                # Allow user to select the correct column
                review_column = st.selectbox(
                    "Select the column containing reviews:",
                    options=df_excel.columns.tolist(),
                    index=df_excel.columns.tolist().index(detected_column) if detected_column in df_excel.columns else 0,
                    key="column_select"
                )
                
                # Show preview
                st.subheader("📋 Data Preview")
                st.dataframe(df_excel[[review_column]].head(10), use_container_width=True)
                
                if st.button("🔍 Analyze Excel Reviews", key="analyze_excel"):
                    with st.spinner("Analyzing reviews..."):
                        # Extract reviews from selected column
                        reviews = df_excel[review_column].dropna().astype(str).tolist()
                        reviews = [r.strip() for r in reviews if r.strip() and len(r.strip()) > 10]
                        
                        if len(reviews) == 0:
                            st.error("No valid reviews found in the selected column.")
                        else:
                            st.success(f"✅ Found {len(reviews)} reviews in the Excel file")
                            display_results(reviews, show_wordcloud_excel, show_individual_excel)

# ---- Footer ----
st.markdown(
    """
    <div class="footer">
        <small>Built using Streamlit | Enhanced with PDF & Excel Analysis</small>
    </div>
    """,
    unsafe_allow_html=True,
)
