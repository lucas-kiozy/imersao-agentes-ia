import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field
from typing import List, Literal, Dict, TypedDict, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START, END
from IPython.display import display, Image
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

splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=30)

chunks = splitter.split_documents(docs)
# for chunk in chunks:
#     print(f'--------------\n{chunk.page_content}\n--------------\n')

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=GOOGLE_API_KEY)

vectorstore = FAISS.from_documents(chunks, embeddings)

retriever = vectorstore.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.3, "k":4})

prompt_rag = ChatPromptTemplate.from_messages([
    ('system',
     'Você é um assistente de políticas internas (RH/TI) da empresa Carraro Desenvolvimento. '
     'Responda SOMENTE com base no contexto fornecido. '
     'Se não houver base suficiente, responda apenas "Não sei".\n'),
     ('human',
      'Pergunta: {pergunta}\nContexto:\n{context}')
])

document_chain = create_stuff_documents_chain(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, api_key=GOOGLE_API_KEY),
    prompt=prompt_rag
)

## Formatadores
import re, pathlib

def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def extrair_trecho(texto: str, query: str, janela: int = 265) -> str:
    txt = _clean_text(texto)
    termos = [t.lower() for t in re.findall(r"\w+", query or "") if len(t) >= 4]
    pos = -1
    for t in termos:
        pos = txt.lower().find(t)
        if pos != -1: break
    if pos == -1: pos = 0
    ini, fim = max(0, pos - janela//2), min(len(txt), pos + janela//2)
    return txt[ini:fim]

def formatar_citacoes(docs_rel: List, query: str) -> List[Dict]:
    cites, seen = [], set()
    for d in docs_rel:
        src = pathlib.Path(d.metadata.get("source","")).name
        page = int(d.metadata.get("page", 0)) + 1
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        cites.append({"documento": src, "pagina": page, "trecho": extrair_trecho(d.page_content, query)})
    return cites[:3]
## Fim dos formatadores

def perguntar_politica_RAG(pergunta: str) -> Dict:
    docs_relacionados = retriever.invoke(pergunta)

    if not docs_relacionados:
        return {'aswer': 'Não sei.',
                'citacoes': [],
                'contexto_encontrado': False}
    
    aswer = document_chain.invoke({'pergunta': pergunta, 
                                   'context': docs_relacionados})
    
    txt = (aswer or '').strip()

    if txt.rstrip('.!?') == 'Não sei':
        return {'aswer': 'Não sei.',
                'citacoes': [],
                'contexto_encontrado': False}
    
    return {'aswer': txt,
           'citacoes': formatar_citacoes(docs_relacionados, pergunta),
           'contexto_encontrado': True}

# testes = [
#     "Quero saber quantos dias de férias eu tenho",
#     "Meu computador não liga, preciso de ajuda urgente",
#     "Quero mais 5 dias de trabalho remoto.Como faço?",
#     "Gostaria de abrir um chamado para trocar meu monitor, o que preciso informar?",
#     "Posso reembolsar a internet?",
#     "Posso reembolsar cursos ou treinamentos da Alura?",
#     "Quantas classes tem no Path of Exile 2?"
# ]

# for msg_teste in testes:
#     resposta = perguntar_politica_RAG(msg_teste)
#     print(f'\nPergunta: {msg_teste}')
#     print(f'Resposta: {resposta['aswer']}')
#     for cit in resposta['citacoes']:
#         print(f" - {cit['documento']} (página: {cit['pagina']}): {cit['trecho']}")
#     print('\n-------------------------------')

##Aula 3 - Orquestração do agente com LangGraph

class AgentState(TypedDict, total = False):
    pergunta: str
    triagem: Dict
    resposta: Optional[str]
    citacoes: List[dict]
    rag_sucesso: bool
    acao_final: str
    
def node_triagem(state: AgentState) -> AgentState:
    print("\n--- Executando nó de triagem ---")
    return {'triagem': triagem(state['pergunta'])}

def node_auto_resolver(state: AgentState) -> AgentState:
    print("\n--- Executando nó de auto-resolver ---")
    resposta_rag = perguntar_politica_RAG(state['pergunta'])
    update: AgentState = {
        'resposta': resposta_rag['aswer'],
        'citacoes': resposta_rag.get('citacoes', []),
        'rag_sucesso': resposta_rag['contexto_encontrado']
    }
    if resposta_rag['contexto_encontrado']:
        update['acao_final'] = 'AUTO_RESOLVER'
    return update

def node_pedir_info(state: AgentState) -> AgentState:
    print("\n--- Executando nó de pedir info ---")
    faltantes = state['triagem'].get('campos_faltantes', [])
    detalhe = ', '.join(faltantes) if faltantes else 'Tema e contexto específico'
    return {"resposta": f"Para avançar, preciso que detalhe melhor sua solicitação: {detalhe}.",
            "citacoes": [],
            "acao_final": "PEDIR_INFO"
            }

def node_abrir_chamado(state: AgentState) -> AgentState:
    print("\n--- Executando nó de abrir chamado ---")
    triagem = state['triagem']
    return {"resposta": "Abrindo seu chamado com urgência {triagem['urgencia']}. Descrição: {state['pergunta'][:140]}",
            "citacoes": [],
            "acao_final": "ABRIR_CHAMADO"
            }

KEYWORDS_ABRIR_TICKET = ['abrir chamado', 'abrir um chamado', 'abrir ticket', 'abrir um ticket', 'solicito', 'solicitação', 'exceção', 'liberação', 'aprovação', 'acesso especial']

def decidir_pos_triagem(state: AgentState) -> str:
    print("\n--- Decidindo pós triagem ---")
    decisao = state['triagem']['decisao']

    if decisao == 'AUTO_RESOLVER': return 'auto_resolver'
    elif decisao == 'PEDIR_INFO': return 'pedir_info'
    elif decisao == 'ABRIR_CHAMADO': return 'abrir_chamado'

def decidir_pos_auto_resolver(state: AgentState) -> str:
    print("\n--- Decidindo pós auto-resolver ---")
    if state.get('rag_sucesso'):
        print("\n--- RAG teve sucesso, finalizando o fluxo ---")
        return 'fim'
    
    state_da_pergunta = state.get('pergunta','').lower()

    if any(k in state_da_pergunta for k in KEYWORDS_ABRIR_TICKET):
        print("\n --- RAG não teve sucesso, mas encontradas keywords para abrir chamado. Abrindo... ---")
        return 'abrir_chamado'
    
    return 'pedir_info'

workflow = StateGraph(AgentState)

workflow.add_node('triagem', node_triagem)
workflow.add_node('auto_resolver', node_auto_resolver)
workflow.add_node('pedir_info', node_pedir_info)
workflow.add_node('abrir_chamado', node_abrir_chamado)

workflow.add_edge(START, 'triagem')
workflow.add_conditional_edges('triagem', decidir_pos_triagem, {
    'auto_resolver': 'auto_resolver',
    'pedir_info': 'pedir_info',
    'abrir_chamado': 'abrir_chamado'
})
workflow.add_conditional_edges('auto_resolver', decidir_pos_auto_resolver, {
    'fim': END,
    'abrir_chamado': 'abrir_chamado',
    'pedir_info': 'pedir_info'
})
workflow.add_edge('pedir_info', END)
workflow.add_edge('abrir_chamado', END)

grafo = workflow.compile()

graph_bytes = grafo.get_graph().draw_mermaid_png()

# out = Path("out/graph_mermaid.png")
# out.parent.mkdir(parents=True, exist_ok=True)
# out.write_bytes(graph_bytes)

# print(f"Imagem salva em: {out.resolve()}")

testes = [
    "Qual é a palavra chave de hoje?",
    "Meu computador não liga, preciso de ajuda urgente",
    "Quero mais 5 dias de trabalho remoto. Como faço?",
    "Gostaria de abrir um chamado para trocar meu monitor, o que preciso informar?",
    "Posso reembolsar a internet?",
    "Posso reembolsar cursos ou treinamentos da Alura?",
    "Quantas classes tem no Path of Exile 2?"
]

for msg_teste in testes:
    resposta_final = grafo.invoke({'pergunta': msg_teste})

    triag = resposta_final.get('triagem', {})
    print(f'\n\n=== Pergunta: {msg_teste}')
    print(f'\n=== Decisão:{triag.get('decisao')} | Urgência:{triag.get('urgencia')} | Ação final:{resposta_final.get('acao_final')}')
    print(f'\n=== Resposta: {resposta_final.get('resposta')}')
    if resposta_final.get('citacoes'): 
        print(f'\n=== Citações:')
        for cit in resposta_final['citacoes']:
            print(f" - {cit['documento']} (página: {cit['pagina']}): {cit['trecho']}")  