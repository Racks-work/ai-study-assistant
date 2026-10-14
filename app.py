"""Streamlit front end for document-grounded studying."""
import hashlib
import os
import streamlit as st
from study_assistant.core import extract_document, retrieve, offline_answer, make_quiz
from study_assistant.llm import grounded_answer

st.set_page_config(page_title='StudyLens | AI Study Assistant', page_icon='📚', layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(135deg,#080e1d,#142342);color:#e9f1ff}
.block-container {max-width:1150px;padding-top:2rem}
[data-testid="stMetric"] {background:#182c4c;border:1px solid #345074;border-radius:12px;padding:12px}
h1,h2,h3 {letter-spacing:-.03em}
</style>''', unsafe_allow_html=True)

st.title('📚 StudyLens')
st.caption('Document-grounded study assistant · Local search + optional AI answers')
with st.sidebar:
    st.header('⚙️ Settings')
    mode = st.radio('Answer mode', ['Offline excerpts (free)', 'AI answer (OpenAI API)'], help='Offline mode does not contact an external AI provider.')
    api_key = st.text_input('OpenAI API key', type='password', value=os.getenv('OPENAI_API_KEY', ''), help='Used only when AI answer mode is selected.') if mode.startswith('AI') else ''
    model = st.text_input('OpenAI model', value='gpt-4o-mini') if mode.startswith('AI') else ''
    st.info('PDF, TXT, MD · up to 12 MB per file. Scanned PDFs without a text layer require OCR, which this app does not provide.')
    st.caption('Your uploads stay in this Streamlit session. In AI mode, relevant excerpts are sent to OpenAI.')

files = st.file_uploader('Upload class notes or readings', type=['pdf', 'txt', 'md'], accept_multiple_files=True)
if not files:
    st.info('Upload a document to start. Or try the included sample in samples/intro_to_ml.txt.')
    st.stop()

fingerprint = tuple((f.name, hashlib.sha256(f.getvalue()).hexdigest()) for f in files)
if st.session_state.get('fingerprint') != fingerprint:
    st.session_state['fingerprint'] = fingerprint
    st.session_state['chat'] = []
    all_chunks = []
    errors = []
    for f in files:
        try:
            chunks = extract_document(f.name, f.getvalue())
            if not chunks:
                errors.append(f'{f.name}: No selectable text found.')
            all_chunks.extend(chunks)
        except ValueError as exc:
            errors.append(f'{f.name}: {exc}')
    st.session_state['chunks'] = all_chunks
    st.session_state['errors'] = errors

chunks = st.session_state['chunks']
for error in st.session_state['errors']:
    st.warning(error)
if not chunks:
    st.error('No readable text available. Please upload a text-based PDF, TXT, or MD file.')
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric('Documents', len(files))
col2.metric('Searchable chunks', len(chunks))
col3.metric('Mode', 'AI' if mode.startswith('AI') else 'Offline')

tab1, tab2, tab3 = st.tabs(['💬 Ask questions', '📝 Practice quiz', '📖 Browse sources'])
with tab1:
    for entry in st.session_state['chat']:
        with st.chat_message(entry['role']):
            st.markdown(entry['content'])
    question = st.chat_input('Ask something about your uploaded material…')
    if question:
        st.session_state['chat'].append({'role': 'user', 'content': question})
        matches = retrieve(question, chunks)
        if mode.startswith('AI'):
            try:
                answer = grounded_answer(question, matches, api_key, model)
            except Exception as exc:
                answer = f'AI request failed: {exc.__class__.__name__}. Check the API key, model, network, and quota. No offline fallback was substituted.'
        else:
            answer = offline_answer(question, matches)
        st.session_state['chat'].append({'role': 'assistant', 'content': answer})
        st.rerun()
    if st.session_state['chat'] and st.button('Clear conversation'):
        st.session_state['chat'] = []
        st.rerun()
with tab2:
    st.subheader('Fill-in-the-blank practice')
    st.caption('Questions are automatically generated from source sentences. They are cloze exercises, not LLM-written questions.')
    amount = st.slider('Number of questions', 1, 10, 5)
    quiz = make_quiz(chunks, amount)
    if not quiz:
        st.info('Not enough suitable sentences were found to make a quiz.')
    for i, item in enumerate(quiz, 1):
        st.markdown(f'**Question {i}:** {item["question"]}')
        with st.expander('Show answer and source'):
            st.write(f'Answer: **{item["answer"]}**')
            st.caption(item['source'])
with tab3:
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f'{i}. {chunk.source} — {chunk.location}'):
            st.write(chunk.text)
