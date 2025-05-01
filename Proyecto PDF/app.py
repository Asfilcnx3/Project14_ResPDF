# Importamos las librerías necesarias
import gradio as gr
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms import HuggingFacePipeline

# Cargar modelo solo una vez
model_id = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.1
)

llm = HuggingFacePipeline(pipeline=pipe)

prompt_es = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Responde en español la respuesta a la pregunta, sin repetir la pregunta ni mostrar contexto.\n\n"
        "Contexto:\n{context}\n\n"
        "Pregunta:\n{question}\n\n"
        "Respuesta:"
    )
)

def chatbot(pdf_file, question):
    # Cargar el PDF
    loader = PyPDFLoader(pdf_file.name)
    pages = loader.load()

    # Dividir en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=25)
    docs = text_splitter.split_documents(pages)

    # Vectorizar
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.from_documents(docs, embedding_model)

    # QA pipeline
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 1}),
        return_source_documents=False,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt_es}
    )

    # Ejecutar respuesta
    respuesta = qa.run(question)
    
    # Extraer la respuesta limpia
    if "Respuesta:" in respuesta:
      respuesta_final = respuesta.split("Respuesta:")[-1].strip()
    else:
      respuesta_final = respuesta.strip()
    return respuesta_final

# Interfaz Gradio
demo = gr.Interface(
    fn=chatbot,
    inputs=[
        gr.File(label="Carga un PDF."),
        gr.Textbox(label="Pregunta algo de tu PDF.")
    ],
    outputs=gr.Textbox(label="Respuesta: "),
    title="Pregunta lo que quieras de tu PDF",
    description="Hazle preguntas a tu PDF usando un modelo Phi-4 + embeddings multilíngües"
)

if __name__ == "__main__":
    demo.launch()