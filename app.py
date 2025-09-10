import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import faiss
load_dotenv()

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

##Aula 1 - Classificação de intenções com IA

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=GOOGLE_API_KEY) #Conexão com o Gemini 2.5

# resp_teste = llm.invoke("Quantos dias de férias eu tenho?")
# print(f'\n {resp_teste.content} \n')

# TRIAGEM_PROMPT = (
#     "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
#     "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
#     "{\n"
#     '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
#     '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
#     '  "campos_faltantes": ["..."]\n'
#     "}\n"
#     "Regras:\n"
#     '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
#     '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
#     '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
#     "Analise a mensagem e decida a ação mais apropriada."
# )

# class TriagemOut(BaseModel):
#     decisao: Literal['AUTO_RESOLVER','PEDIR_INFO','ABRIR_CHAMADO']
#     urgencia: Literal['BAIXA','MEDIA','ALTA']
#     campos_faltantes: List[str] = Field(default_factory=list)

# llm_triagem = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=GOOGLE_API_KEY)

# triagem_chain = llm_triagem.with_structured_output(TriagemOut)

# def triagem(mensagem: str) -> Dict:
#     saida : TriagemOut = triagem_chain.invoke([
#         SystemMessage(content=TRIAGEM_PROMPT),
#         HumanMessage(content=mensagem)
#     ])
#     return saida.model_dump()

# testes = [
#     "Quero saber quantos dias de férias eu tenho",
#     "Meu computador não liga, preciso de ajuda urgente",
#     "Gostaria de abrir um chamado para trocar meu monitor, o que preciso informar?",
#     "Posso reembolsar a internet?",
#     "Posso reembolsar cursos ou treinamentos da Alura?"
# ]

# for msg_teste in testes:
#     print(f'\nMensagem: {msg_teste}')
#     print(f'Resposta: {triagem(msg_teste)}\n')


##Aula 2 - Contruindo a base de conhecimento com RAG

docs =  []

for n in Path("content/").glob("*.pdf"):
    try:
        loader = PyMuPDFLoader(str(n))
        docs.extend(loader.load())
        print(f'Carregado com sucesso: {n.name}')
    except Exception as e:
        print(f'Erro ao carregar {n.name}: {e}')

print(f'\nTotal de documentos carregados: {len(docs)}\n')

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

chunks = splitter.split_documents(docs)
for chunk in chunks:
    print(f'--------------\n{chunk.page_content}\n--------------\n')

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)

vectorstore = faiss.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_treshold":0.3, "k":4})