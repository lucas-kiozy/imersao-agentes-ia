import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import List, Literal, Dict
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')

##Aula 1 - Classificação de intenções com IA

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=GOOGLE_API_KEY) #Conexão com o Gemini 2.5

# resp_teste = llm.invoke("Quantos dias de férias eu tenho?")
# print(f'\n {resp_teste.content} \n')

TRIAGEM_PROMPT = (
    "Você é um triador de Service Desk para políticas internas da empresa Carraro Desenvolvimento. "
    "Dada a mensagem do usuário, retorne SOMENTE um JSON com:\n"
    "{\n"
    '  "decisao": "AUTO_RESOLVER" | "PEDIR_INFO" | "ABRIR_CHAMADO",\n'
    '  "urgencia": "BAIXA" | "MEDIA" | "ALTA",\n'
    '  "campos_faltantes": ["..."]\n'
    "}\n"
    "Regras:\n"
    '- **AUTO_RESOLVER**: Perguntas claras sobre regras ou procedimentos descritos nas políticas (Ex: "Posso reembolsar a internet do meu home office?", "Como funciona a política de alimentação em viagens?").\n'
    '- **PEDIR_INFO**: Mensagens vagas ou que faltam informações para identificar o tema ou contexto (Ex: "Preciso de ajuda com uma política", "Tenho uma dúvida geral").\n'
    '- **ABRIR_CHAMADO**: Pedidos de exceção, liberação, aprovação ou acesso especial, ou quando o usuário explicitamente pede para abrir um chamado (Ex: "Quero exceção para trabalhar 5 dias remoto.", "Solicito liberação para anexos externos.", "Por favor, abra um chamado para o RH.").'
    "Analise a mensagem e decida a ação mais apropriada."
)

class TriagemOut(BaseModel):
    decisao: Literal['AUTO_RESOLVER','PEDIR_INFO','ABRIR_CHAMADO']
    urgencia: Literal['BAIXA','MEDIA','ALTA']
    campos_faltantes: List[str] = Field(default_factory=list)

llm_triagem = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, api_key=GOOGLE_API_KEY)

triagem_chain = llm_triagem.with_structured_output(TriagemOut)

def triagem(mensagem: str) -> Dict:
    saida : TriagemOut = triagem_chain.invoke([
        SystemMessage(content=TRIAGEM_PROMPT),
        HumanMessage(content=mensagem)
    ])
    return saida.model_dump()

testes = [
    "Quero saber quantos dias de férias eu tenho",
    "Meu computador não liga, preciso de ajuda urgente",
    "Gostaria de abrir um chamado para trocar meu monitor, o que preciso informar?",
    "Posso reembolsar a internet?",
    "Posso reembolsar cursos ou treinamentos da Alura?"
]

for msg_teste in testes:
    print(f'\nMensagem: {msg_teste}')
    print(f'Resposta: {triagem(msg_teste)}\n')