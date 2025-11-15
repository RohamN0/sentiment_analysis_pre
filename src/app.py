from transformers import AutoModelForSequenceClassification
from model import TextPipeline
import streamlit as st
import pandas as pd
import torch

@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        'roberta-large',
        num_labels=5
    )
    # You can use any model here
    state_dict = torch.load('./model.bin', map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    
    return model


model = load_model()

st.title('📝 Real-Time Review Score Predictor (1–5)')
st.write('Enter a product review and the model will predict the user rating (1–5).')

review = st.text_area('Enter review text:')

if st.button('Predict'):
    if not review.strip():
        st.warning('Please enter text.')
    else:
        pre_status = st.empty()
        pre_status.info('Processing the review!')

        pipeline = TextPipeline(review)
        input_ids, attention_mask = pipeline.process()

        input_ids = input_ids.to(torch.long)
        attention_mask = attention_mask.to(torch.long)

        pre_status.empty()
        st.success('Processing completed!')
        
        with torch.no_grad():
            model_status = st.empty()
            model_status.info('Running model prediction...')

            outputs = model(input_ids, attention_mask)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1).item()

        model_status.empty()

        st.subheader(f'⭐ Predicted Rating: **{predicted + 1}/5**')

        probs_list = [f'{x * 100:.3f}%' for x in probs[0].tolist()]

        st.subheader('📊 Confidence Breakdown')

        df_conf = pd.DataFrame({
            'Rating': [1, 2, 3, 4, 5],
            'Confidence': probs_list
        })

        highlight = lambda r: ['background-color: #00c500' if r.Rating == (predicted + 1) else '' for _ in r]
        st.dataframe(
            df_conf.style.apply(highlight, axis=1),
            hide_index=True
        )   
