# Vogais Libras - AutoML + XAI

Este projeto utiliza Visão Computacional e Aprendizado de Máquina para classificar as vogais (**A, E, I, O, U**) na Língua Brasileira de Sinais (LIBRAS). O diferencial deste repositório é o uso de **AutoML** para otimização do modelo e **XAI (Explainable AI)** para visualização do que o modelo está "olhando" para tomar a decisão.

## 🚀 Funcionalidades

- **Coleta de Dados Automatizada**: Script para capturar imagens da webcam e organizar o dataset.
- **AutoML (Keras Tuner)**: Busca automática pelos melhores hiperparâmetros (taxa de aprendizado, dropout, unidades densas) usando Otimização Bayesiana.
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Mapa de calor baseado nos gradientes da última camada convolucional.
  - **Occlusion Sensitivity**: Identifica quais partes da imagem são mais importantes ao ocultar pequenas regiões e medir a queda na confiança.
- **Interface em Tempo Real**: Painel com múltiplas visões (Webcam, Grad-CAM, Evidência por Oclusão e Gráficos de Probabilidade).

## 🛠️ Tecnologias Utilizadas

- **Python 3.10+**
- **TensorFlow / Keras**
- **Keras Tuner** (AutoML)
- **OpenCV** (Processamento de imagem e interface)
- **MobileNetV2** (Transfer Learning)

## 📦 Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/GRB-04/vogais-xai-automl.git
   cd vogais-xai-automl
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Fluxo de Trabalho

Siga estes passos para treinar e rodar o projeto do zero:

1. **Coletar Imagens**:
   Rode o script e siga as instruções na tela para capturar mãos fazendo as vogais.
   ```bash
   python collect_data.py
   ```

2. **Preparar Dataset**:
   Organiza e divide as imagens em pastas de treino e validação.
   ```bash
   python split_dataset.py
   ```

3. **Treinar com AutoML**:
   Inicia a busca pelos melhores parâmetros e realiza o fine-tuning automático.
   ```bash
   python train_automl.py
   ```

4. **Executar Inferência (com XAI)**:
   Abre a interface em tempo real com as explicações visuais.
   ```bash
   python realtime_infer_xai_layout.py
   ```

## ⌨️ Comandos na Interface de Inferência

- `q`: Sair do programa.
- `g`: Ligar/Desligar visualização do **Grad-CAM**.
- `o`: Ligar/Desligar visualização de **Oclusão**.

---
Desenvolvido como um exemplo prático de como tornar modelos de Deep Learning mais transparentes e eficientes.
