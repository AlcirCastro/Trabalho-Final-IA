import torch
import ltn

# Definição dos quantificadores
Not = ltn.Connective(ltn.fuzzy_ops.NotStandard())
And = ltn.Connective(ltn.fuzzy_ops.AndProd())
Or = ltn.Connective(ltn.fuzzy_ops.OrProbSum())
Implies = ltn.Connective(ltn.fuzzy_ops.ImpliesReichenbach())
Equiv = ltn.Connective(ltn.fuzzy_ops.Equiv(ltn.fuzzy_ops.AndProd(), ltn.fuzzy_ops.ImpliesReichenbach()))
Forall = ltn.Quantifier(ltn.fuzzy_ops.AggregPMeanError(p=2), quantifier="f")
Exists = ltn.Quantifier(ltn.fuzzy_ops.AggregPMean(p=6), quantifier="e")

# Predicados espaciais
def O(bb1, bb2):
    # Predicate to check if two bounding boxes overlap
    return Or(p(bb1, bb2), p(bb2, bb1))

def PO(bb1, bb2):
    # Predicate to check if two bounding boxes partially overlap
    return Exists([ltn.Variable("A", bb1), ltn.Variable("B", bb2)], And(p(bb1, bb2), p(bb2, bb1)))

def D(bb1, bb2):
    # Predicate to check if two bounding boxes are disjoint
    return Not(Or(p(bb1, bb2), p(bb2, bb1)))

p = ltn.Predicate(func=lambda A, B: torch.all(torch.stack([
    torch.gt(A[:, 0] - B[:, 0], 0),                 # Verifica se x de A é maior que x de B
    torch.gt(A[:, 1] - B[:, 1], 0),                 # Verifica se y de A é maior que y de B
    torch.lt(A[:, 0] + A[:, 2] - (B[:, 0] + B[:, 2]), 0),# Verifica se lado direito de A está à esquerda de B
    torch.lt(A[:, 1] + A[:, 3] - (B[:, 1] + B[:, 3]), 0) # Verifica se lado inferior de A está acima de B
])))


left = ltn.Predicate(func=lambda A, D: torch.gt(A[:, 0] - (D[:, 0] + D[:, 2]), 0))

def above(A, C):
    # Predicate to check if A is above C
    return p(A, C)

def inSideRight(E, A):
    # Predicado para verificar se E está à direita de A
    return And(p(A, E), Not(left(A, E)))

def inSideLeft(I, A):
    # Predicado para verificar se I está à esquerda de A
    return And(p(A, I), Not(left(A, I)))

def inside(A, B):
    return Exists([A], p(A, B))

def inAbove(C, A):
    # Predicate to check if C is inside and above A
    return And(p(A, C), above(C, A))

def interpret_output(output):
    # Se o valor do resultado for maior que 0.5, retorna True, caso contrário, retorna False
    return output.value > 0.5

# Bounding boxes fornecidas
bbox1 = torch.tensor([[0.15, 0.09, 0.74, 0.91]])  # x, y, w, h bb
bbox2 = torch.tensor([[0.19, 0.12, 0.26, 0.86]])  # x, y, w, h pessoa

# Consulta ao predicado inSideLeft
A = ltn.Variable("A", bbox1)
B = ltn.Variable("B", bbox2)
output = inSideLeft(A, B)

# Interpretar e imprimir o resultado
print(interpret_output(output))
