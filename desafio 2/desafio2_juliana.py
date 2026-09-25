"""
harness_desafio2.py — Desafio 2: Ativação e inicialização em redes profundas (sem normalização)
GBC073 — Inteligência Computacional (FACOM/UFU) — Prof. Marcelo Keese Albertini

O ALUNO ENTREGA, no próprio repositório, o arquivo desafio2/desafio2_nomes.py com duas funções:

    def ativacao(x: torch.Tensor) -> torch.Tensor
        # elemento a elemento, sem parâmetros, sem olhar o lote; a derivada vem do autograd

    @torch.no_grad()
    def inicializar(W: torch.Tensor, b: torch.Tensor,
                    fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None
        # preenche W (fan_out, fan_in) e b (fan_out,) IN-PLACE; camada = 1..n_camadas

O HARNESS constrói um MLP de largura 256 e profundidade L ∈ {4, 16, 48}, SEM BatchNorm,
SEM conexões residuais, alternando Linear -> ativacao; inicializa cada camada com
inicializar(); treina 3 épocas com SGD (momento 0,9, lr 0,05, lote 128, entropia cruzada);
mede a acurácia de teste. Também imprime um "raio-X" da propagação de sinal no passo 0.

Escore: s_t = clip((m - baseline)/(referencia - baseline), 0, 1.25)
        S   = 100 * (0.7*média(s_t) + 0.3*mín(s_t))
Baseline:   tanh + U(-0.05, 0.05).       Referência: ReLU + He normal, viés zero.

Uso:
    python harness_desafio2.py                         # baseline e referência, tarefas públicas
    python harness_desafio2.py desafio2/desafio2_nomes.py   # avalia a submissão
    python harness_desafio2.py desafio2/desafio2_nomes.py --rapido    # 10% dos dados, 1 época: teste de fumaça
    python harness_desafio2.py desafio2/desafio2_nomes.py --ocultas   # professor: usa ocultas_d2.py, se existir
    python harness_desafio2.py --calibrar              # recalcula e grava calibracao_d2.json

Baseline e referência são caros (treinam 9+ redes); ficam em cache em calibracao_d2.json.
Dependências: torch, torchvision (baixa MNIST/FashionMNIST/CIFAR-10 em ./dados).
"""
# Nome: Juliana Rodrigues Gonçalves

import math
import torch

def ativacao(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)

_g = torch.Generator().manual_seed(0)
_E_f2 = ativacao(torch.randn(1_000_000, generator=_g)).pow(2).mean().item()

@torch.no_grad()
def inicializar(W: torch.Tensor, b: torch.Tensor,
                fan_in: int, fan_out: int, camada: int, n_camadas: int) -> None:
    
    if camada == 1:
        desvio = math.sqrt(1.0 / fan_in)
    else:
        desvio = math.sqrt(1.0 / (fan_in * _E_f2))
    
    if camada == n_camadas:
        desvio *= 0.5
        
    W.normal_(0.0, desvio)    
    b.zero_()
