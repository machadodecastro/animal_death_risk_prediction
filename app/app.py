# App de Medição de Indice de Descarte

# Imports
import pickle
import numpy as np
import pandas as pd
import logging, io, os, sys
from sklearn.ensemble import GradientBoostingClassifier
from flask import Flask, render_template, flash, request, jsonify

# pip install xgboost
from xgboost import XGBClassifier

# pip install flask_httpauth
from flask_httpauth import HTTPBasicAuth


# Cria a app
app = Flask(__name__)


# Inicializa a autenticação
auth = HTTPBasicAuth()


# Define o modelo como None
modelo_linhagem = None


# Variáveis de entrada
atributos = ['temperatura',
			 'amonia',
			 'idade',
			 'peso',
			 'sexo',
			 'fotoperiodo',
			 'ruido',
			 'luz',
			 'umidade',
			 'infeccoes',
			 'animais_por_gaiola',
			 'linhagem']
 

# Usuários de acesso
usuarios = {"ictb": "ictb", "lab": "lab"}


# Função para obter os usuários
@auth.get_password
def get_pw(username):
    if username in usuarios:
        return usuarios.get(username)
    return None

 # Função para mostrar imagem
def mostra_imagem(linhagem, nivel_descarte):

	# Verifica a linhagem
	if linhagem == 0:
		linhagem_str = 'black'
	else:
		linhagem_str = 'swiss'

	# Retorna a imagem correspondente
	return('/static/imagens/' + linhagem_str + '_' + str(nivel_descarte) + '.jpg')


# Função para carregar o modelo ao inicializar a app
@app.before_request
def startup():
	global modelo_linhagem 

	# Carrega o modelo
	# modelo_linhagem = pickle.load(open("static/modelo/modelo_linhagens.json",'rb'))

	modelo_linhagem = XGBClassifier()
	modelo_linhagem.load_model("../modelo/modelo_linhagens.json")
 

# Função para formatar mensagem de erro
@app.errorhandler(500)
def server_error(e):
    logging.exception('some error')
    return """
    And internal error <pre>{}</pre>
    """.format(e), 500


# Função para executar o processo de atribuição das variáveis
@app.route('/background_process', methods = ['POST', 'GET'])
def background_process():
	temperatura          = float(request.args.get('temperatura'))
	amonia               = float(request.args.get('amonia'))
	idade                = float(request.args.get('idade'))
	peso                 = float(request.args.get('peso'))
	sexo                 = float(request.args.get('sexo'))
	fotoperiodo          = float(request.args.get('fotoperiodo'))
	ruido                = float(request.args.get('ruido'))
	luz                  = float(request.args.get('luz'))
	umidade              = float(request.args.get('umidade'))
	infeccoes            = float(request.args.get('infeccoes'))
	animais_por_gaiola   = float(request.args.get('animais_por_gaiola'))
	linhagem             = int(request.args.get('linhagem'))


	# Cria o dataframe para os novos dados
	novos_dados = pd.DataFrame([[temperatura,
		                         amonia,
		                         idade,
		                         peso,
		                         sexo,
		                         fotoperiodo,
		                         ruido,
		                         luz,
		                         umidade,
		                         infeccoes,
		                         animais_por_gaiola,
		                         linhagem]], 
		                         columns = atributos)


	# Faz as previsões
	previsoes = modelo_linhagem.predict_proba(novos_dados[atributos])

	# Obtém a melhor previsão (maior probabilidade)
	melhor_previsao = [3,6,9][np.argmax(previsoes[0])]

	# Retorna a cor do vinho e a imagem que corresponde à previsão
	return jsonify({'qualidade_prevista':melhor_previsao, 'image_name': mostra_imagem(linhagem, melhor_previsao)})


# Função para carregar a página principal e renderizar a imagem
@app.route("/", methods = ['POST', 'GET'])
@auth.login_required
def index():
	logging.warning("index!")
	return render_template('index.html', qualidade_prevista = 1, image_name = '/static/imagens/imagem.jpg')
 
@app.route("/sobre")
@auth.login_required
def sobre():
	return render_template('sobre.html')

# Executa app
if __name__ == '__main__':
    app.debug = True
    app.run()

